from __future__ import annotations
 
import logging
import time
from functools import cached_property
from typing import Any
 
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.agxpiper.agxpiper_utils import AGXPiper_Joints
from lerobot.robots.robot import Robot
 
from .config_agxpiper import AGXPiperConfig
 
logger = logging.getLogger(__name__)
 
# piper_sdk 官方 demo 里关节反馈/控制的原始单位是「0.001 度」的整数
# （即 raw = degrees * 1000）。度 <-> 弧度换算用标准常数，这部分是通用数学关系，
# 不依赖固件版本，可以放心用；但 raw <-> degrees 的比例关系需要用你的 demo 实测确认。
_DEG_PER_RAW_UNIT = 0.001
_RAD_PER_DEG = 3.141592653589793 / 180.0
_METER_PER_GRIPPER_RAW_UNIT =  1000.0 * 1000.0  # 0.001mm -> 1m
 
 
class AGXPiper(Robot):
    """AgileX Piper 6 自由度机械臂 + 夹爪，双路 USB Webcam。"""
 
    config_class = AGXPiperConfig
    name = "piper"
 
    def __init__(self, config: AGXPiperConfig):
        super().__init__(config)
        self.config = config
        self._bus = None  # piper_sdk 的 C_PiperInterface(_V2) 实例，connect() 时创建
        self._is_connected = False
        self._last_action = None
        # 官方相机工厂：把 {"front_top": OpenCVCameraConfig(...), "wrist": OpenCVCameraConfig(...)}
        # 变成 {"front_top": OpenCVCamera(...), "wrist": OpenCVCamera(...)}
        self.cameras = make_cameras_from_configs(config.cameras)
 
    @cached_property
    def _piper_joint_names(self) -> list[str]:
        return [joint.name for joint in AGXPiper_Joints]
    
    # ------------------------------------------------------------------
    # 特征契约（Robot 抽象基类要求实现）
    # ------------------------------------------------------------------
    @cached_property
    def observation_features(self) -> dict[str, Any]:
        # HWC is the standard image array format
        cam_features = {
            key: (value.height, value.width, 3) for key, value in self.cameras.items()
        }

        return {**self.action_features, **cam_features}
 
    @cached_property
    def action_features(self) -> dict[str, Any]:
        return {f"{name}": float for name in self._piper_joint_names}
 
    @property
    def is_connected(self) -> bool:
        return self._is_connected and all(cam.is_connected for cam in self.cameras.values())
 
    @property
    def is_calibrated(self) -> bool:
        # Piper 用绝对值编码器，出厂已标定，正常情况下不需要 lerobot 那套
        # 「记录关节零位/量程」的标定流程（这点和 SO100 这类需要手动标定的舵机臂不同）。
        # 如果你的固件/型号并非如此，请把这里改成实际的标定检查逻辑。
        return True
 
    # ------------------------------------------------------------------
    # 连接生命周期
    # ------------------------------------------------------------------
    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            raise RuntimeError(f"{self} already connected")

        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, PiperFW
        robot = ArmModel.PIPER
        firmware_version = PiperFW.DEFAULT
        robot_cfg = create_agx_arm_config(
            robot=robot,
            firmeware_version=firmware_version,
            interface="socketcan",
            channel="can0",
            # receive_own_messages=True,
            # local_loopback=True,
        )
        self.robot = AgxArmFactory.create_arm(robot_cfg)
        self.robot.connect()
        self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)

        while not self.robot.enable():
            print("Enabling AGXPiper...")
            time.sleep(0.01)

        self.robot.set_motion_mode(self.robot.OPTIONS.MOTION_MODE.CPV)
        deadline = time.monotonic() + self.config.connect_timeout_s
        while time.monotonic() < deadline:
            status = self.robot.get_arm_status()
            if status is not None and getattr(status.msg, "motion_status", None) == 0:
                print("motion done")
                break
            time.sleep(0.02)
 
        self._is_connected = True
 
        if calibrate and not self.is_calibrated:
            self.calibrate()
 
        self.configure()
 
        for cam in self.cameras.values():
            cam.connect()
 
        logger.info(f"{self} connected.")
 
    def calibrate(self) -> None:
        # 见 is_calibrated 的说明：绝对编码器 + 出厂标定，这里不需要额外动作。
        # 保留方法是为了满足 Robot 抽象基类的接口契约。
        logger.info(f"{self}: Piper 使用绝对值编码器，出厂已标定，跳过标定流程。")
 
    def configure(self) -> None:
        # 如果固件支持且未在构造 C_PiperInterface_V2 时设置成功，可在这里补充调用
        # SetSDKJointLimitParam / SetSDKGripperRangeParam 之类的软件限位设置。
        # 具体函数名请以你安装的 piper_sdk 版本的 demo（piper_set_sdk_param.py）为准。
        pass
 
    def disconnect(self) -> None:
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected")
 
        if self.config.disable_torque_on_disconnect:
            self.robot.disable()
        else:
            self._move_to_zero()
 
        for cam in self.cameras.values():
            cam.disconnect()
 
        self._bus = None
        self._is_connected = False
        logger.info(f"{self} disconnected.")

    def _move_to_zero(self):
        position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for idx, pos in enumerate(position):
            self.robot.move_cpv_pos(idx+1, pos)
        # end effector
        self.end_effector.move_gripper_m(0.000)

 
    # ------------------------------------------------------------------
    # 观测 / 动作
    # ------------------------------------------------------------------
    def get_obs_joints(self):
        obs: dict[str, Any] = {}
        joint_msg = self.robot.get_joint_angles().msg
        gripper_msg = self.end_effector.get_gripper_status().msg.value
        for idx, name in enumerate(self._piper_joint_names[:-2]):
            obs[f"{name}"] = joint_msg[idx]

        # grippers' distance, only need half, joint8 is negative
        obs[f"{self._piper_joint_names[-2]}"] = gripper_msg / 2
        obs[f"{self._piper_joint_names[-1]}"] = -1.0 * gripper_msg / 2

        return list(obs.values())

    def get_observation(self) -> dict[str, Any]:
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")
 
        obs: dict[str, Any] = {}
 
        joint_msg = self.robot.get_joint_angles().msg
        gripper_msg = self.end_effector.get_gripper_status().msg.value

        logger.debug(f"Obs from real piper: {joint_msg}, {gripper_msg}")
 
        for idx, name in enumerate(self._piper_joint_names[:-2]):
            obs[f"{name}"] = joint_msg[idx]

        # grippers' distance, only need half, joint8 is negative
        obs[f"{self._piper_joint_names[-2]}"] = gripper_msg / 2
        obs[f"{self._piper_joint_names[-1]}"] = -1.0 * gripper_msg / 2

        logger.debug(f"Obs in policy's convention: {obs}")

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.async_read()
 
        return obs
 
    def send_action(self, action: dict[str, Any]):
        if not self._is_connected:
            raise RuntimeError(f"{self} is not connected. Call connect() first.")
 
        goal = dict(action)

        if self.config.max_relative_target is not None:
            current = self.get_observation()
            for key in self._piper_joint_names:
                delta = goal[key] - current[key]
                clipped_delta = max(
                    -self.config.max_relative_target, min(self.config.max_relative_target, delta)
                )
                goal[key] = current[key] + clipped_delta

        joints = [goal[f"{name}"] for name in self._piper_joint_names[:-2]]
        grippers_dist = goal[self._piper_joint_names[-2]] * 2

        # filter joints
        if self._last_action is None:
            filtered_joint_degrees = joints
        else:
            filtered_joint_degrees = [
                self.config.alpha * p + (1-self.config.alpha) * l
                for p, l in zip(joints, self._last_action)
            ]
        self._last_action = filtered_joint_degrees
        for idx, pos in enumerate(filtered_joint_degrees):
            self.robot.move_cpv_pos(idx+1, pos)
        self.end_effector.move_gripper_m(grippers_dist)

        return [*filtered_joint_degrees, goal[self._piper_joint_names[-2]], goal[self._piper_joint_names[-1]]]

    def reset_env(self, ep: int, seed: int):
        self._move_to_zero()
        time.sleep(0.5)
        logger.info("AGXPiper after resetting.")

    def __str__(self) -> str:
        return f"AGXPiper({self.config.id})"
