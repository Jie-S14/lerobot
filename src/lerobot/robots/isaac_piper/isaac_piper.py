import csv
from functools import cached_property
import json
import logging
from pathlib import Path
import random
import threading
import time
from typing import Any, Optional

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.isaac_piper.isaac_piper_utils import Isaac_Piper_Joints
from lerobot.robots.robot import Robot
from .config_isaac_piper import IsaacPiperConfig
from lerobot.cameras.isaac.camera_isaac import IsaacCamera  # type: ignore
from lerobot.utils.random_utils import get_random_position, get_random_orientation
from lerobot.utils.utils import get_euclidean_distance

logger = logging.getLogger(__name__)

class IsaacPiper(Robot):
    config_class = IsaacPiperConfig
    name = "isaac_piper"

    def __init__(self, config: IsaacPiperConfig, world: Optional[Any] = None):
        """
        :param config: IsaacPiperConfig instance
        :param world: Optional existing omni.isaac.core.World (or similar). If provided, cameras and robot will
                      attempt to reuse it instead of creating their own Worlds/stages.
        """
        super().__init__(config)
        self.config = config
        self._connected = False
        # placeholders for Isaac handles
        self._app = None
        self._robot = None
        self._object = None
        self._goal = None
        self._object_leaf = None
        self._tasks = []
        self._obj_poses = {}
        self._distractor_poses = {}
        self._goal_poses = {}

        # prepare camera wrappers but do not connect them yet
        self._cameras = make_cameras_from_configs(config.cameras)
        
        with open(config.obj_pose_path, "r", encoding="utf-8") as f:
            # self._obj_poses = json.load(f)["object"]
            reader = csv.DictReader(f)
            for row in reader:
                match row["role"]:
                    case "cube" if row["is_target"] == "True":
                        self._obj_poses[str(row["episode_id"])] = {"position": [float(row["x"]), float(row["y"]), self.config.obj_zaxis_offset],
                                                                    "orientation": float(row["orientation_deg"]),
                                                                    "color": row["color"]}
                    case "cube" if row["is_target"] == "False":
                        self._distractor_poses[str(row["episode_id"])] = {"position": [float(row["x"]), float(row["y"]), self.config.obj_zaxis_offset],
                                                                    "orientation": float(row["orientation_deg"]),
                                                                    "color": row["color"]}
                    case "box":
                        self._goal_poses[str(row["episode_id"])] = {"position": [float(row["x"]), float(row["y"]), self.config.goal_zaxis_offset],
                                                              "orientation": float(row["orientation_deg"])}
                        self._tasks.append(row["instruction"])

        # with open(config.goal_pose_path, "r", encoding="utf-8") as f:
        #     self._goal_poses = json.load(f)["goal"]


    @cached_property
    def observation_features(self) -> dict:
        # HWC is the standard image array format
        cam_features = {
            key: (value.height, value.width, 3) for key, value in self._cameras.items()
        }

        return {**self.action_features, **cam_features}

    @cached_property
    def _piper_joint_names(self) -> list[str]:
        return [joint.name for joint in Isaac_Piper_Joints]

    @cached_property
    def action_features(self) -> dict:
        return {f"{name}": float for name in self._piper_joint_names}
    
    @cached_property
    def tasks(self) -> list[str]:
        return self._tasks

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True, world: Optional[Any] = None) -> None:
        """
        Connect robot to Isaac. Optionally pass a `world` to share with cameras / other robots.
        If a shared `world` is given here it will be attached to camera wrappers created in __init__.
        """
        # Lazy import to avoid hard dependency at import time
        try:
            # must-have extensions for Isaac Sim 4.2
            config = {
                "headless": self.config.headless,
                "exts": [
                    "omni.isaac.ros2_bridge",  # must-have
                    "omni.isaac.core_nodes"  # resolve IsaacReadSimulationTime warning
                ]
            }
            # First have to instantiate sim_app
            # import isaacsim
            from omni.isaac.kit import SimulationApp
            self._app = SimulationApp(config)

            from omni.isaac.core.utils.extensions import enable_extension, disable_extension
            enable_extension("omni.isaac.ros2_bridge")

            from omni.isaac.core.simulation_context import SimulationContext
            from omni.isaac.core.utils.stage import open_stage
            from omni.isaac.core import World
            from omni.isaac.core.robots import Robot
            from omni.isaac.core.prims import XFormPrim
            from omni.isaac.core.prims import RigidPrim
            from omni.isaac.core.articulations import Articulation

            simulation_context = SimulationContext()
            open_stage(usd_path=self.config.stage_path)
            self._app.update()
            
            self.world = World(physics_dt=self.config.simulation_dt,
                             rendering_dt=self.config.simulation_dt)
            self._robot = Articulation(prim_path=self.config.robot_prim_path,
                            name=self.config.robot_prim_path.split("/")[-1])
            self._object = XFormPrim(prim_path=self.config.object_prim_path,
                                    name=self.config.object_prim_path.split("/")[-1])
            self._goal = XFormPrim(prim_path=self.config.goal_prim_path,
                                    name=self.config.goal_prim_path.split("/")[-1])

            self._object_leaf = RigidPrim(prim_path=self.config.object_prim_path + "/Cube")

            self.world.scene.add(self._robot)
            self.world.scene.add(self._object)
            self.world.scene.add(self._goal)
            self.world.scene.add(self._object_leaf)

            if len(self._distractor_poses) > 0:
                self._distractor = XFormPrim(prim_path=self.config.distractor_prim_path,
                                                    name=self.config.distractor_prim_path.split("/")[-1])
                self.world.scene.add(self._distractor)

            for _, cam in self._cameras.items():
                cam.attach_to_world(self.world)   # comment if use opencv cam
                cam.connect()
            self.world.reset()      # Have to be after all　 initialization, otherwise no data

            logger.info("Connected to Isaac robot")
            
        except Exception as e:
            logger.error(f"Failed to connect to Isaac robot: {e}")
            raise RuntimeError("Omniverse Isaac imports failed. Make sure Isaac Sim 4.2 Python environment is active.")

        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        # For simulated robot this can be always True
        return True

    def calibrate(self) -> None:
        # No-op for simulation or implement if needed
        pass

    @property
    def cameras(self) -> dict[str, IsaacCamera]:
        return self._cameras

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError("IsaacSimRobot not connected")

        # joints
        joint_positions = self._robot.get_joint_positions().tolist()   # convert ndarray with dtype=float32 to list[float]
        obs = dict(zip(self._piper_joint_names, joint_positions))
        # cameras
        for name in self._cameras.keys():
            try:
                frame = self._cameras[name].read()
                if frame is None or frame.shape[0] == 0:
                    logger.warning(f"frame {frame} is empty. Probably because no world.reset() after camera initialization.")
                obs[name] = frame

            except Exception as e:
                logger.warning(f"Failed to read camera {name}: {e}")
                obs[name] = None
        return obs

    def _move_object(self, object,  pos, ori) -> None:
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        quat = euler_angles_to_quat(np.array(ori), degrees=True)
        logger.info(f"move_obj: {pos}, {ori}, {quat}")
        object.set_world_pose(
                position=np.array(pos),
                orientation=quat)  # (w, x,y,z)

    def reset_env(self, ep: int, seed: int) -> None:
        """
        Use it when recording/evaluating dataset
        Args:
            seed: master seed
            ep: offset seed
        Returns:

        """
        self.world.reset()  # reset() must before move_obj() otherwise reset() reloads USD
        self._set_cube_material(self.config.object_prim_path, self._obj_poses[f"{ep}"]["color"])
        
        self._move_object(self._object,
                          self._obj_poses[f"{ep}"]["position"],
                          [0, 0, self._obj_poses[f"{ep}"]["orientation"]])

        self._move_object(self._goal,
                          self._goal_poses[f"{ep}"]["position"],
                          [0, 0, self._goal_poses[f"{ep}"]["orientation"]])

        if len(self._distractor_poses) > 0:
            self._set_cube_material(self.config.distractor_prim_path, self._distractor_poses[f"{ep}"]["color"])
            self._move_object(self._distractor,
                              self._distractor_poses[f"{ep}"]["position"],
                              [0, 0, self._goal_poses[f"{ep}"]["orientation"]])

        for cam in self._cameras.values():
            cam.warmup()

    def _set_cube_material(self, object_prim_path, material_name):
        # update the cube material
        from omni.usd import get_context
        from pxr import UsdShade

        stage = get_context().get_stage()
        cube_prim = stage.GetPrimAtPath(f"{object_prim_path}/Cube")

        material_prim = stage.GetPrimAtPath(f"{object_prim_path}/Materials/{material_name.capitalize()}")

        assert cube_prim.IsValid(), f"Invalid cube prim: {cube_prim.GetPath()}"
        assert material_prim.IsValid(), f"Invalid material: {material_prim.GetPath()}"

        UsdShade.MaterialBindingAPI(cube_prim).Bind(UsdShade.Material(material_prim))

    def send_action(self, action: RobotAction) -> RobotAction:
        # For inference
        from omni.isaac.core.utils.types import ArticulationAction

        joint_targets = np.array(list(action.values()))
        arti_action = ArticulationAction(joint_positions=joint_targets)

        self._robot.apply_action(arti_action)

        # logger.info(f"send_action: self._robot.get_joint_positions(): {self._robot.get_joint_positions()}")
        return action
    
    def _get_real_dof_names(self):
        logger.info(f"The real Isaac Sim robot DOF names: {self._robot.dof_names}")

    @property
    def joint_positions(self):
        return self._robot.get_joint_positions().tolist()   # convert ndarray with dtype=float32 to list[float]

    def configure(self) -> None:
        """
        Apply any one-time or runtime configuration to the robot.
        This may include setting motor parameters, control modes, or initial state.
        """
        pass


    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass
        self.world.stop()
        # self._app.close() # comment it to solve eval process robot client thread disconnect hang issue 
        # TODO: shutdown SimulationApp if created (self._app) and cleanup stage if owned
        self._connected = False

    def check_success(self) -> bool:
        """
        成功判定逻辑:
          1. XY: 把 object 相对 goal(box) 中心的偏移, 旋转到 box 的局部坐标系下(因为box每个episode朝向不同),
             再用 box 的半长/半宽(减去cube的旋转安全裕量)做矩形范围判断——这是主要判据。
          2. Z: 只做粗过滤(排除物体还悬空/异常穿模), 不作为主要判据——因为这个box很矮,
             筐底高度和桌面高度几乎相同,Z轴对"是否真正放入筐内"区分度很弱。
          3. 夹爪需处于张开状态,排除"正抓着物体从筐上方掠过"的假阳性。
          4. 以上条件需连续保持 success_hold_steps 步(去抖动),排除单帧穿模/抖动导致的瞬时误判。
        """
        JOINT_NAMES = [j.name for j in Isaac_Piper_Joints]  # 严格保持这个顺序: joint1..6, gripper_joint1, gripper_joint2
        _CUBE_SIZE = (0.05076477313041683, 0.050764773130416885, 0.05076477313041683)
        _BOX_SIZE = (0.1683948377237406, 0.15239914167349733, 0.029272011849160284)

        gripper_open_threshold = 0.01,  # 判定"夹爪已松开"的开口宽度阈值,量纲同gripper joint (0~0.07)
        cube_half = np.asarray(_CUBE_SIZE, dtype=np.float64) / 2.0
        box_half = np.asarray(_BOX_SIZE, dtype=np.float64) / 2.0
        success_diff_z_high = _BOX_SIZE[2]
        xy_rotation_safe_margin = True,  # True: 用cube的xy对角线半长做安全边界(抗任意yaw旋转);False: 用cube半边长(更宽松但假设cube摆正)

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

        # obj_pos, _ = self.robot._object.get_world_pose()
        obj_pos, _ = self._object_leaf.get_world_pose()
        goal_pos, goal_quat = self._goal.get_world_pose()
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
        in_xy_range = (abs(local_x) < xy_threshold_x) and (abs(local_y) < xy_threshold_y)
        logging.info(f"local_x={local_x}, local_y={local_y}")
        logging.info(f"_xy_threshold_x={xy_threshold_x}, _xy_threshold_y={xy_threshold_y}")
        logging.info(f"in_xy_range={in_xy_range}")

        z_diff = obj_pos[2] - goal_pos[2]
        in_z_range = abs(z_diff) < success_diff_z_high
        logging.info(f"z_diff={z_diff}, success_diff_z_high={success_diff_z_high} in_z_range={in_z_range}")

        joint_positions = self.joint_positions
        gripper_joint1 = abs(joint_positions[JOINT_NAMES.index("gripper_joint1")])
        gripper_joint2 = abs(joint_positions[JOINT_NAMES.index("gripper_joint2")])
        gripper_closed = gripper_joint1 < gripper_open_threshold and gripper_joint2 < gripper_open_threshold
        logging.info(f"gripper_joint1={gripper_joint1}, gripper_joint2={gripper_joint2}, gripper_closed={gripper_closed}")

        condition_met = in_xy_range and in_z_range and gripper_closed

        if condition_met:
            return True
        else:
            return False