# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import cv2
import draccus
import grpc
import torch

from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.async_inference.helpers import (
    FPSTracker,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, heatmap_to_rgb, blend_heatmap_on_image
import torch.nn.functional as F
import numpy as np
import rerun as rr


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # ---- RTC state ----
        # 存的是"上一次预测的 chunk"，但是是 postprocess（反归一化）之前的
        # 版本 —— 因为 RTC 的 denoise_step 是在模型内部的归一化空间里做
        # guidance 计算的，不能直接用发给 client 的、已经反归一化的动作。
        # 形状: (chunk_len, action_dim)，已经去掉了 batch 维。
        self.last_raw_action_chunk: torch.Tensor | None = None
        # 这个 chunk 对应的起始 timestep（也就是当时那次推理用的观测的
        # timestep），用来在下一次推理时算 "consumed = 新观测timestep -
        # 这个值"，consumed 同时就是 inference_delay，也用来定位
        # leftover 该从 chunk 的第几步开始切。
        self.last_raw_action_chunk_start_timestep: int | None = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

        # 新 client 连进来，RTC 的"历史 chunk"就作废了，必须清空，
        # 不然会拿上一个 client/episode 的动作去 guide 这一个 episode。
        self.last_raw_action_chunk = None
        self.last_raw_action_chunk_start_timestep = None

    def reset_rtc_state(self) -> None:
        """Empty RTC memory，not the obs queue,
        to prevent the RTC from using the memory of the previous episode.
        """
        self.last_raw_action_chunk = None
        self.last_raw_action_chunk_start_timestep = None
        self.logger.debug("[RTC] episode 边界重置：清空 last_raw_action_chunk")

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        policy_class = get_policy_class(self.policy_type)

        start = time.perf_counter()
        self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        self.policy.to(self.device)

        # print policy structure
        for name, module in self.policy.named_modules():
            print(name, type(module))

        from torchinfo import summary
        summary(self.policy)

        for name, p in self.policy.named_parameters():
            if p.requires_grad:
                print(name)

        # Load preprocessor and postprocessor, overriding device to match requested device
        device_override = {"device": self.device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            },
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        try:
            receive_time = time.time()  # comparing timestamps so need time.time()
            start_deserialize = time.perf_counter()
            received_bytes = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, self.logger.name
            )  # blocking call while looping over request_iterator
            timed_observation = pickle.loads(received_bytes)  # nosec
            deserialize_time = time.perf_counter() - start_deserialize

            self.logger.info(f"Received observation #{timed_observation.get_timestep()}")

            obs_timestep = timed_observation.get_timestep()
            obs_timestamp = timed_observation.get_timestamp()

            # Calculate FPS metrics
            fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

            self.logger.debug(
                f"Received observation #{obs_timestep} | "
                f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
                f"Target: {fps_metrics['target_fps']:.2f} | "
                f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Server timestamp: {receive_time:.6f} | "
                f"Client timestamp: {obs_timestamp:.6f} | "
                f"Deserialization time: {deserialize_time:.6f}s"
            )

            if not self._enqueue_observation(
                timed_observation  # wrapping a RawObservation
            ):
                self.logger.debug(f"Observation #{obs_timestep} has been filtered out") # because it is not a must_go

            return services_pb2.Empty()

        except grpc.RpcError as e:
            # client disconnected / channel closed while streaming -> handle gracefully
            self.logger.info(f"gRPC client stream closed while receiving observations: {e}")
            return services_pb2.Empty()
        except Exception as e:
            self.logger.exception(f"Error while receiving observations: {e}")
            return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            # log_rerun_data(observation=obs.get_observation(), compress_images=True)

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            self.last_processed_obs = obs  # prevent twice #0 obs

            start_time = time.perf_counter()
            action_chunk = self._predict_action_chunk(obs)
            inference_time = time.perf_counter() - start_time

            # 如果模型保存了 attention map，就取出来并上传到 rerun
            try:
                attn = None
                model = getattr(self.policy, "model", None)
                if model is not None:
                    vlm_expert = getattr(model, "vlm_with_expert", None)
                    if vlm_expert is not None:
                        # 优先使用保存的 cross-attn 列表（denoise 阶段 append 的条目）
                        cross_list = getattr(vlm_expert, "cross_attn_means", None)
                        if cross_list and len(cross_list) > 0:
                            attn = cross_list[-1]  # 应为 cpu tensor (B, Q, K)
                        else:
                            # 兜底：使用 last_attn_mean（可能来自其他 attention 调用）
                            attn = getattr(vlm_expert, "last_attn_mean", None)
                        prefix_layout = getattr(vlm_expert, "last_prefix_layout", None)
                        # 将 token-level attention (B, Q, K) -> key-level importance (K,)
                        try:
                            attn_np = attn.cpu().numpy()  # (B, Q, K)
                            key_attn = attn_np[0].mean(axis=0)  # (K,)

                            obs_raw = obs.get_observation()
                            # 尝试从 observation 中读取原始图像尺寸（fallback 会使用 layout 中的 orig_hw 或 224x224）
                            target_hw = None
                            target_hw = (480, 640)
                            # for img_key in (self.policy_image_features or []):
                            #     if isinstance(obs_raw, dict) and img_key in obs_raw:
                            #         img_data = obs_raw[img_key]
                            #         try:
                            #             # 支持 numpy / torch tensor / PIL-like ndarray
                            #             if isinstance(img_data, torch.Tensor):
                            #                 h, w = int(img_data.shape[-2]), int(img_data.shape[-1])
                            #             else:
                            #                 h, w = int(img_data.shape[-2]), int(img_data.shape[-1])
                            #             target_hw = (h, w)
                            #             break
                            #         except Exception:
                            #             target_hw = None

                            attention_maps_per_image = {}
                            if prefix_layout is not None:
                                for layout in prefix_layout.get("image_layouts", []):
                                    s, e = layout["start_idx"], layout["end_idx"]
                                    h, w = layout["h"], layout["w"]
                                    key_slice = key_attn[s:e]
                                    if key_slice.size != h * w:
                                        # 如果长度不匹配，尝试裁剪或填充
                                        pad_or_trim = h * w
                                        if key_slice.size > pad_or_trim:
                                            key_slice = key_slice[:pad_or_trim]
                                        else:
                                            key_slice = np.pad(key_slice, (0, pad_or_trim - key_slice.size),
                                                               mode="constant")
                                    # reshape -> (1,1,h,w) 方便 interpolate
                                    arr = torch.from_numpy(key_slice.reshape(1, 1, h, w).astype("float32"))
                                    # 如果没有目标像素尺寸，优先使用 layout.orig_hw，否则 224x224
                                    tgt_h, tgt_w = None, None
                                    if target_hw is not None:
                                        tgt_h, tgt_w = target_hw
                                    elif layout.get("orig_hw") is not None:
                                        tgt_h, tgt_w = layout["orig_hw"]
                                    else:
                                        tgt_h, tgt_w = 224, 224

                                    try:
                                        up = F.interpolate(arr, size=(tgt_h, tgt_w), mode="bilinear",
                                                           align_corners=False)
                                        heatmap = up.squeeze().cpu().numpy()
                                        heat_rgb = heatmap_to_rgb(heatmap, cmap="magma")
                                        # blended = blend_heatmap_on_image(obs_img, heat_rgb, alpha=0.55)
                                    except Exception:
                                        heatmap = arr.squeeze().cpu().numpy()

                                    attention_maps_per_image[f"image_{layout['image_index']}"] = heat_rgb
                                # # For language instruction
                                # lang_tokens = ["BOS", "Approach", "the", "red", "cube", ",", "pick", "it", "up", ",",
                                #                "move", "it", "to", "the", "purple", "box", ",", "release", "it", ",",
                                #                "return", "to", "the", "initial", "position", ".", "EOS"]
                                # # --- Token strip (render text into an image) ---
                                # H = 40
                                # W = len(lang_tokens) * 40  # one cell per token
                                # lang_img = np.ones((H, W, 3), dtype=np.uint8) * 255
                                #
                                # for i, tok in enumerate(lang_tokens):
                                #     x = i * 40 + 5
                                #     cv2.putText(
                                #         lang_img,
                                #         tok,
                                #         (x, 25),
                                #         cv2.FONT_HERSHEY_SIMPLEX,
                                #         0.4,
                                #         (0, 0, 0),
                                #         1,
                                #         cv2.LINE_AA,
                                #     )
                                # # Attention strip
                                # lang_attn = key_attn[128: 155]
                                # lang_attn = lang_attn / lang_attn.max()
                                # lang_attn_heatmap = lang_attn.reshape(1, -1)
                                # lang_attn_heatmap = np.repeat(lang_attn_heatmap, H, axis=0)
                                # lang_attn_heatmap = np.repeat(lang_attn_heatmap, H, axis=1)
                                # lang_attn_heatmap_img = (lang_attn_heatmap * 255).astype(np.uint8)
                                # lang_attn_heatmap_img = cv2.applyColorMap(lang_attn_heatmap_img, cv2.COLORMAP_JET)
                            # rr.log("camera/top", rr.Image(obs.get_observation()["top"]))
                            # rr.log("camera/wrist", rr.Image(obs.get_observation()["wrist"]))
                            # rr.log("camera/attn_top", rr.Image(attention_maps_per_image["image_0"]))
                            # rr.log("camera/attn_wrist", rr.Image(attention_maps_per_image["image_1"]))
                            overlay_top = cv2.addWeighted(obs.get_observation()["top"], 0.5,
                                                          attention_maps_per_image["image_0"], 0.5,
                                                          0)
                            overlay_wrist = cv2.addWeighted(obs.get_observation()["wrist"], 0.5,
                                                          attention_maps_per_image["image_1"], 0.5,
                                                          0)

                            rr.log("camera/top_overlay", rr.Image(overlay_top))
                            rr.log("camera/wrist_overlay", rr.Image(overlay_wrist))
                            # rr.log("lang/overlay", rr.Image(np.vstack([lang_attn_heatmap_img, lang_img])))
                            # rr.log("lang/attn", rr.Tensor(lang_attn.reshape(1, -1)))
                            # rr.log("lang/tokens", rr.Image(lang_img))
                        except Exception as e:
                            # fallback: 原有行为
                            self.logger.error(f"Failed to map attention to images: {e}")
                            # log_rerun_data(observation=obs.get_observation(), compress_images=True)
                            rr.log("camera/top", rr.Image(obs.get_observation()["top"]))
                            rr.log("camera/wrist", rr.Image(obs.get_observation()["wrist"]))
                else:
                    log_rerun_data(observation=obs.get_observation(), compress_images=True)
            except Exception as e:
                self.logger.info(f"Failed to log attention: {e}")

            start_time = time.perf_counter()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time * 1000:.2f}ms, total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            time.sleep(
                max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            )  # sleep controls inference latency

            return actions

        except grpc.RpcError as e:
            self.logger.info(f"gRPC error while streaming actions to client {client_id}: {e}")
            return services_pb2.Empty()
        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()
        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")
            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.info(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features, atol=torch.tensor(self.config.obs_similarity_atol, dtype=torch.float32)):
            self.logger.info(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        with self._predicted_timesteps_lock:    # prevent twice #0 obs
            if obs.get_timestep() in self._predicted_timesteps:
                return False

        self.logger.debug(
            f"[ENQUEUE CHECK] obs_ts={obs.get_timestep()} must_go={obs.must_go} "
            f"last_processed_is_none={self.last_processed_obs is None} "
            f"predicted_timesteps={self._predicted_timesteps}"
        )

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"[Enqueuing observation] timestep:{obs.timestep} | Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(self, t_0: float, action_chunk: list[torch.Tensor], i_0: int) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(timestamp=t_0 + i * self.config.environment_dt, timestep=i_0 + i, action=action)
            for i, action in enumerate(action_chunk)
        ]

    def _rtc_enabled(self) -> bool:
        """RTC 是否应该被使用，完全由 checkpoint 的 config.json 里的
        rtc_config.enabled 决定 —— 这样切换 RTC on/off 做对照实验时，
        只需要改 config.json，policy_server.py 的代码逻辑本身不用动。"""
        rtc_config = getattr(self.policy.config, "rtc_config", None)
        return rtc_config is not None and rtc_config.enabled

    def _compute_rtc_kwargs(self, observation_t: TimedObservation) -> dict:
        """根据"上一次预测的 chunk"和这次观测的 timestep，算出 RTC 需要的
        三个参数。如果是第一次推理（没有历史 chunk），三个都返回 None，
        RTC 内部逻辑本身就支持 prev_chunk_left_over=None（相当于不做
        guidance），所以这里不需要特殊处理第一次的情况。
        """
        if not self._rtc_enabled() or self.last_raw_action_chunk is None:
            return {"inference_delay": None, "prev_chunk_left_over": None, "execution_horizon": None}

        current_timestep = observation_t.get_timestep()
        # consumed = 距离上次推理，机器人已经执行了多少步。
        # 同时这个数字就是 RTC 要的 inference_delay（单位：步）。
        consumed = current_timestep - self.last_raw_action_chunk_start_timestep

        if consumed <= 0:
            # 理论上不应该出现（新观测的 timestep 不该比上次预测时还早），
            # 出现说明上游 obs 乱序了，保守起见当作没有历史可用。
            self.logger.warning(
                f"[RTC] consumed={consumed} <= 0 (current_timestep={current_timestep}, "
                f"last_chunk_start={self.last_raw_action_chunk_start_timestep}), 跳过 RTC guidance"
            )
            return {"inference_delay": None, "prev_chunk_left_over": None, "execution_horizon": None}

        if consumed >= self.last_raw_action_chunk.shape[0]:
            # 上一个 chunk 已经被完全消费完了，没有 leftover 可用
            self.logger.debug(f"[RTC] 上一个 chunk 已消费完 (consumed={consumed}), 本次不做 RTC guidance")
            return {"inference_delay": None, "prev_chunk_left_over": None, "execution_horizon": None}

        prev_chunk_left_over = self.last_raw_action_chunk[consumed:, :]

        self.logger.info(
            f"[RTC] inference_delay={consumed} steps | "
            f"prev_chunk_left_over shape={tuple(prev_chunk_left_over.shape)}"
        )

        return {
            "inference_delay": consumed,
            "prev_chunk_left_over": prev_chunk_left_over,
            "execution_horizon": self.policy.config.rtc_config.execution_horizon,
        }

    def _get_action_chunk(
        self, observation: dict[str, torch.Tensor], observation_t: TimedObservation
    ) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        rtc_kwargs = self._compute_rtc_kwargs(observation_t)
        chunk = self.policy.predict_action_chunk(observation, **rtc_kwargs)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        chunk = chunk[:, : self.actions_per_chunk, :]

        # 把这次预测的"原始"（归一化空间、postprocess 之前）chunk 存下来，
        # 留给下一次推理算 RTC guidance 用。squeeze 掉 batch 维，因为
        # server 一次只服务一个 client，不需要保留 batch 维。
        if self._rtc_enabled():
            self.last_raw_action_chunk = chunk.detach().clone().squeeze(0)
            self.last_raw_action_chunk_start_timestep = observation_t.get_timestep()

        return chunk

    def _predict_action_chunk(self, observation_t: TimedObservation) -> list[TimedAction]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation, observation_t)
        inference_time = time.perf_counter() - start_inference
        self.logger.debug(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()

        """5. Convert to TimedAction list"""
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(), list(action_tensor), observation_t.get_timestep()
        )
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk

    def stop(self):
        """Stop the server"""
        self._reset_server()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Start rerun.io
    init_rerun("camera_demo")

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    server.wait_for_termination()

    policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()