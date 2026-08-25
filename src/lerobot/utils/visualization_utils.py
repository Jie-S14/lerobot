# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numbers
import os

import numpy as np
import rerun as rr
import matplotlib.pyplot as plt

from lerobot.processor import RobotAction, RobotObservation

from .constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR

# track whether we initialized/started rerun in this process
_rerun_started = False
_rerun_spawned = False

def init_rerun(
    session_name: str = "lerobot_control_loop", ip: str | None = None, port: int | None = None
) -> None:
    """
    Initializes the Rerun SDK for visualizing the control loop.

    Args:
        session_name: Name of the Rerun session.
        ip: Optional IP for connecting to a Rerun server.
        port: Optional port for connecting to a Rerun server.
    """
    global _rerun_started, _rerun_spawned
    batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
    os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
    rr.init(session_name)
    memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
    if ip and port:
        rr.connect_grpc(url=f"rerun+http://{ip}:{port}/proxy")
        _rerun_started = True
        _rerun_spawned = False
    else:
        rr.spawn(memory_limit=memory_limit, detach_process=False)
        _rerun_started = True
        _rerun_spawned = True

def stop_rerun(timeout_s: float = 2.0) -> None:
    """
    Try to gracefully stop / flush the rerun SDK started by init_rerun.
    This function is defensive: it attempts common teardown APIs (flush, shutdown, disconnect, close)
    and ignores exceptions so teardown never raises during program shutdown.
    """
    global _rerun_started, _rerun_spawned
    if not _rerun_started:
        return

    try:
        rr.disconnect()
        # rr.rerun_shutdown()
        # try to flush any pending data first
        # if hasattr(rr, "flush"):
        #     try:
        #         rr.flush()
        #     except Exception:
        #         # some rr versions may not implement flush or may raise internally
        #         pass

        # # prefer public shutdown if available
        # if hasattr(rr, "shutdown"):
        #     try:
        #         rr.shutdown()
        #     except Exception:
        #         pass
        # # fallback disconnect / close variants found in different rr versions
        # elif hasattr(rr, "disconnect_grpc"):
        #     try:
        #         rr.disconnect_grpc()
        #     except Exception:
        #         pass
        # elif hasattr(rr, "disconnect"):
        #     try:
        #         rr.disconnect()
        #     except Exception:
        #         pass
        # elif hasattr(rr, "close"):
        #     try:
        #         rr.close()
        #     except Exception:
        #         pass
    except Exception as e:
        # be maximally defensive: swallow all errors during teardown
        pass
    finally:
        _rerun_started = False
        _rerun_spawned = False


def _is_scalar(x):
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


def log_rerun_data(
    observation: RobotObservation | None = None,
    action: RobotAction | None = None,
    compress_images: bool = False,
) -> None:
    """
    Logs observation and action data to Rerun for real-time visualization.

    This function iterates through the provided observation and action dictionaries and sends their contents
    to the Rerun viewer. It handles different data types appropriately:
    - Scalars values (floats, ints) are logged as `rr.Scalars`.
    - 3D NumPy arrays that resemble images (e.g., with 1, 3, or 4 channels first) are transposed
      from CHW to HWC format, (optionally) compressed to JPEG and logged as `rr.Image` or `rr.EncodedImage`.
    - 1D NumPy arrays are logged as a series of individual scalars, with each element indexed.
    - Other multi-dimensional arrays are flattened and logged as individual scalars.

    Keys are automatically namespaced with "observation." or "action." if not already present.

    Args:
        observation: An optional dictionary containing observation data to log.
        action: An optional dictionary containing action data to log.
        compress_images: Whether to compress images before logging to save bandwidth & memory in exchange for cpu and quality.
    """
    if observation:
        for k, v in observation.items():
            if v is None:
                continue
            key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                arr = v
                # Convert CHW -> HWC when needed
                if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.ndim == 1:
                    for i, vi in enumerate(arr):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                    rr.log(key, entity=img_entity, static=True)

    if action:
        for k, v in action.items():
            if v is None:
                continue
            key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

            if _is_scalar(v):
                rr.log(key, rr.Scalars(float(v)))
            elif isinstance(v, np.ndarray):
                if v.ndim == 1:
                    for i, vi in enumerate(v):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))
                else:
                    # Fall back to flattening higher-dimensional arrays
                    flat = v.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{key}_{i}", rr.Scalars(float(vi)))


def heatmap_to_rgb(heat: np.ndarray, cmap: str = "jet", vmin: float | None = None, vmax: float | None = None):
    """
    heat: HxW float (not necessarily 0..1)
    returns: HxWx3 uint8 RGB
    """
    if vmin is None:
        vmin = float(np.percentile(heat, 5))
    if vmax is None:
        vmax = float(np.percentile(heat, 95))
    # clip & normalize
    norm = np.clip((heat - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    cmap_f = plt.get_cmap(cmap)
    colored = cmap_f(norm)[:, :, :3]  # RGBA -> RGB
    return (colored * 255).astype(np.uint8)

def blend_heatmap_on_image(image: np.ndarray, heat_rgb: np.ndarray, alpha: float = 0.5):
    """
    image: HxWx3 uint8
    heat_rgb: HxWx3 uint8 (will be resized to image if sizes differ)
    """
    import cv2
    if image.shape[:2] != heat_rgb.shape[:2]:
        heat_rgb = cv2.resize(heat_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = (image.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha).astype(np.uint8)
    return blended