#!/usr/bin/env python
from __future__ import annotations

import logging
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot.robots.isaac_piper.config_isaac_piper import IsaacPiperConfig
from lerobot.robots.isaac_piper.isaac_piper import IsaacPiper
from lerobot.robots.isaac_piper.isaac_piper_utils import Isaac_Piper_Joints
from lerobot.cameras.isaac.camera_isaac import IsaacCameraConfig
from lerobot.processor import RobotObservation
from lerobot.utils.robot_utils import precise_sleep

JOINT_NAMES = [j.name for j in Isaac_Piper_Joints]  # 严格保持这个顺序: joint1..6, gripper_joint1, gripper_joint2
ACTION_DIM = len(JOINT_NAMES)  # 8
_DEFAULT_JOINT_LOW = [-2.6179, 0.0, -2.967, -1.745, -1.22, -2.09439, 0.0, -0.35]
_DEFAULT_JOINT_HIGH = [2.6179, 3.14, 0.0, 1.745, 1.22, 2.09439, 0.35, 0.0]
_CUBE_SIZE = (0.05076477313041683, 0.050764773130416885, 0.05076477313041683)
_BOX_SIZE = (0.1683948377237406, 0.15239914167349733, 0.029272011849160284)

class IsaacPiperEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        task: str = "pick_place",
        camera_name: tuple[str, ...] = ("top", "wrist"),
        obs_type: str = "pixels_agent_pos",
        observation_height: int = 480,
        observation_width: int = 640,
        episode_length: int = 300,
        fps: int = 15,
        sim_steps_per_action: int = 1,
        headless: bool = False,
        stage_path: str = "/home/shenjie/usd/train/2cam_top_wst_goal_1pos_real_official.usd",
        robot_prim_path: str = "/World/piper_official/base_link",
        object_prim_path: str = "/World/red_block",
        goal_prim_path: str = "/World/small_KLT",
        obj_pose_path: str = "",
        goal_pose_path: str = "",
        obj_zaxis_offset: float = 0.78524,
        goal_zaxis_offset: float = 0.77553,
        top_camera_prim_path: str = "/World/TopCamera",
        wrist_camera_prim_path: str = "/World/piper_official/camera_link/WristCamera",
        joint_low: list | None = None,
        joint_high: list | None = None,
        # success_xy_threshold: float = 0.05,  # 物体中心到goal中心的xy距离阈值(m),需按small_KLT实际内径标定
        # success_z_low: float = 0.012,  # 物体落入筐内后, z相对goal_z的最小偏移(m)
        success_diff_z_high: float = 0.02,  # 物体落入筐内后, z相对goal_z的最大偏移(m),需按筐深+方块尺寸标定
        gripper_open_threshold: float = 0.1,  # 判定"夹爪已松开"的开口宽度阈值,量纲同gripper joint (0~0.7)
        success_hold_steps: int = 5,  # 条件需连续保持的步数,做去抖动
        cube_size: tuple[float, float, float] = _CUBE_SIZE,
        box_size: tuple[float, float, float] = _BOX_SIZE,
        xy_rotation_safe_margin: bool = True,  # True: 用cube的xy对角线半长做安全边界(抗任意yaw旋转);False: 用cube半边长(更宽松但假设cube摆正)
        # z_tolerance: float = 0.02,  # z判定只做粗过滤,不需要卡太死
        episode_index: int = 0,
        n_envs: int = 1,
    ):
        super().__init__()
        self.task = task
        self.obs_type = obs_type
        self.fps = fps
        self.observation_height = observation_height
        self.observation_width = observation_width
        self.camera_name = list(camera_name)
        self._max_episode_steps = episode_length
        self.sim_steps_per_action = sim_steps_per_action
        self._reset_stride = n_envs
        self.init_state_id = episode_index
        # self._success_xy_threshold = success_xy_threshold
        # self._success_z_low = success_z_low
        self._success_diff_z_high = _BOX_SIZE[2]
        self._gripper_open_threshold = gripper_open_threshold
        self._success_hold_steps = success_hold_steps
        self._success_hold_counter = 0

        cube_half = np.asarray(cube_size, dtype=np.float64) / 2.0
        box_half = np.asarray(box_size, dtype=np.float64) / 2.0

        if xy_rotation_safe_margin:
            # cube可能绕z任意旋转,取xy对角线半长做保守边界,保证任意朝向下cube都完全落在筐内才算成功
            cube_xy_margin = float(np.linalg.norm(cube_half[:2]))
        else:
            cube_xy_margin = float(np.max(cube_half[:2]))

        xy_threshold_x = box_half[0] - cube_xy_margin
        xy_threshold_y = box_half[1] - cube_xy_margin
        if xy_threshold_x <= 0 or xy_threshold_y <= 0:
            raise ValueError(
                f"box太小装不下cube(考虑旋转裕量后): threshold_x={xy_threshold_x}, threshold_y={xy_threshold_y}. "
                "检查cube_size/box_size是否传反,或改用xy_rotation_safe_margin=False。"
            )
        self._xy_threshold_x = xy_threshold_x
        self._xy_threshold_y = xy_threshold_y
        # self._still_z_diff = cube_half[2] - box_half[2]   # does not work
        # self._z_tolerance = z_tolerance

        # 相机 prim 路径的映射,camera_name 里每个 key 必须在这里有对应关系
        cam_prim_map = {"top": top_camera_prim_path, "wrist": wrist_camera_prim_path}
        cameras = {
            cam: IsaacCameraConfig(
                prim_path=cam_prim_map[cam],
                width=observation_width,
                height=observation_height,
                fps=fps,
            )
            for cam in self.camera_name
        }

        # 关键修复:simulation_dt 不会因为传了 fps 就自动重算,必须手动同步
        cfg = IsaacPiperConfig(
            stage_path=stage_path,
            robot_prim_path=robot_prim_path,
            object_prim_path=object_prim_path,
            goal_prim_path=goal_prim_path,
            fps=fps,
            simulation_dt=1.0 / fps,       # 显式同步,不能省略
            headless=headless,             # 依赖 config_isaac_piper.py 补上这个字段
            cameras=cameras,
            obj_pose_path=obj_pose_path,
            goal_pose_path=goal_pose_path,
            obj_zaxis_offset=obj_zaxis_offset,
            goal_zaxis_offset=goal_zaxis_offset,
        )
        self.robot = IsaacPiper(config=cfg)
        self.robot.connect()
        self._n_recorded_eps = len(self.robot._obj_poses)

        images = {
            cam: spaces.Box(low=0, high=255,
                             shape=(observation_height, observation_width, 3), dtype=np.uint8)
            for cam in self.camera_name
        }
        if obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict({
                "pixels": spaces.Dict(images),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(ACTION_DIM,), dtype=np.float64),
            })
        elif obs_type == "pixels":
            self.observation_space = spaces.Dict({"pixels": spaces.Dict(images)})
        else:
            raise NotImplementedError(f"obs_type={obs_type} not supported for IsaacPiperEnv")

        low = np.array(joint_low if joint_low is not None else _DEFAULT_JOINT_LOW, dtype=np.float32)
        high = np.array(joint_high if joint_high is not None else _DEFAULT_JOINT_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(ACTION_DIM,), dtype=np.float32)

    # _format_raw_obs / reset / step / _check_success / render / close 与之前版本相同,不变

    def _format_raw_obs(self, raw_obs: dict) -> RobotObservation:
        # robot.get_observation() 已经把 joint 值和相机帧拍平在同一个 dict 里返回,直接取用即可
        images = {cam: raw_obs[cam] for cam in self.camera_name}
        agent_pos = np.array([raw_obs[name] for name in JOINT_NAMES], dtype=np.float64)

        if self.obs_type == "pixels":
            return {"pixels": images}
        return {"pixels": images, "agent_pos": agent_pos}

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        # 注意:robot.reset_env(ep, seed) 内部并不使用 seed 做随机化——
        # cube/goal 的初始位置完全由 ep(CSV里记录的episode_id)决定。
        # 这里循环使用录制好的100组初始状态,机制上等价于libero的init_state_id % len(...)。
        ep = self.init_state_id % self._n_recorded_eps
        self.task = self.robot.tasks[ep]
        self.robot.reset_env(ep=ep, seed=seed or 0)
        self.init_state_id += self._reset_stride

        raw_obs = self.robot.get_observation()
        observation = self._format_raw_obs(raw_obs)
        self._step_count = 0
        self._success_hold_counter = 0  # 新增:每次reset清零
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray):
        start_loop_t = time.perf_counter()
        if action.ndim != 1:
            raise ValueError(f"Expected 1-D action (shape (action_dim,)), got shape {action.shape}")

        action_dict = dict(zip(JOINT_NAMES, action.tolist(), strict=True))
        self.robot.send_action(action_dict)

        # send_action 只下发目标位置,不会推进物理/渲染循环——env 必须自己步进仿真
        for _ in range(self.sim_steps_per_action):
            self.robot.world.step(render=True)

        # 非 headless（GUI 调试）时按 fps 节拍等待，headless（真实 eval）时不等待
        if not self.robot.config.headless:
            target_dt = self.sim_steps_per_action / self.fps  # 一个 env.step() 应占用的墙钟时间
            elapsed = time.perf_counter() - start_loop_t
            logging.debug(f"Elapsed time: {elapsed:.2f}s")
            time.sleep(max(0.0, target_dt - elapsed))

        raw_obs = self.robot.get_observation()
        observation = self._format_raw_obs(raw_obs)

        is_success = self._check_success()
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_steps
        terminated = bool(is_success)
        done = terminated or truncated

        reward = float(is_success)  # 占位:稀疏0/1奖励,后续如需shaped reward再补

        info = {"task": self.task, "done": done, "is_success": is_success, "n_step": self._step_count}
        if done:
            info["final_info"] = {"task": self.task, "done": bool(done), "is_success": bool(is_success), }
            # self.reset()

        return observation, reward, terminated, truncated, info

    def _check_success(self) -> bool:
        """
        成功判定逻辑:
          1. XY: 把 object 相对 goal(box) 中心的偏移, 旋转到 box 的局部坐标系下(因为box每个episode朝向不同),
             再用 box 的半长/半宽(减去cube的旋转安全裕量)做矩形范围判断——这是主要判据。
          2. Z: 只做粗过滤(排除物体还悬空/异常穿模), 不作为主要判据——因为这个box很矮,
             筐底高度和桌面高度几乎相同,Z轴对"是否真正放入筐内"区分度很弱。
          3. 夹爪需处于张开状态,排除"正抓着物体从筐上方掠过"的假阳性。
          4. 以上条件需连续保持 success_hold_steps 步(去抖动),排除单帧穿模/抖动导致的瞬时误判。
        """
        obj_pos, _ = self.robot._object.get_world_pose()
        goal_pos, goal_quat = self.robot._goal.get_world_pose()
        obj_pos = np.asarray(obj_pos, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        goal_quat = np.asarray(goal_quat, dtype=np.float64)  # (w, x, y, z), 与 _move_object 保持一致
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        roll, pitch, yaw = quat_to_euler_angles(goal_quat, degrees=False)   # radian, rotate around z-axis
        yaw += np.radians(-90)
        logging.debug(f"obj_pos={obj_pos}, goal_pos={goal_pos}, goal_ori={yaw}")

        diff = obj_pos[:2] - goal_pos[:2]

        # diff向量相反yaw角度旋转，在原坐标系的投影
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        local_x = diff[0] * cos_y + diff[1] * sin_y
        local_y = -diff[0] * sin_y + diff[1] * cos_y
        in_xy_range = (abs(local_x) < self._xy_threshold_x) and (abs(local_y) < self._xy_threshold_y)
        logging.info(f"local_x={local_x}, local_y={local_y}")
        logging.info(f"_xy_threshold_x={self._xy_threshold_x}, _xy_threshold_y={self._xy_threshold_y}")
        logging.info(f"in_xy_range={in_xy_range}")

        z_diff = obj_pos[2] - goal_pos[2]
        in_z_range = abs(z_diff) < self._success_diff_z_high
        logging.info(f"z_diff={z_diff}, success_diff_z_high={self._success_diff_z_high} in_z_range={in_z_range}")

        joint_positions = self.robot.joint_positions
        gripper_joint1 = abs(joint_positions[JOINT_NAMES.index("gripper_joint1")])
        gripper_joint2 = abs(joint_positions[JOINT_NAMES.index("gripper_joint2")])
        gripper_closed = (gripper_joint1 + gripper_joint2) < self._gripper_open_threshold
        logging.info(f"gripper_joint1={gripper_joint1}, gripper_joint2={gripper_joint2}, gripper_closed={gripper_closed}")

        condition_met = in_xy_range and in_z_range and gripper_closed

        if condition_met:
            self._success_hold_counter += 1
        else:
            self._success_hold_counter = 0

        return self._success_hold_counter >= self._success_hold_steps

    def render(self):
        raw_obs = self.robot.get_observation()
        return raw_obs[self.camera_name[0]]

    def close(self):
        self.robot.disconnect()


def create_isaac_piper_envs(
    task: str,
    n_envs: int,
    env_cls,
    gym_kwargs: dict[str, Any] | None = None,
) -> dict[str, dict[int, Any]]:
    if n_envs != 1:
        raise NotImplementedError(
            "IsaacPiperEnv 目前只支持 n_envs=1:每次 IsaacPiper.connect() 都会起一个独立的 "
            "SimulationApp,Isaac Sim 4.2 不支持同进程多实例。"
        )
    gym_kwargs = dict(gym_kwargs or {})

    def _make_env():
        return IsaacPiperEnv(n_envs=n_envs, **gym_kwargs)

    vec_env = env_cls([_make_env])
    return {task: {0: vec_env}}