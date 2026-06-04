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
        # store provided/shared world
        self.world: Optional[Any] = world
        # prepare camera wrappers but do not connect them yet
        self.cameras = make_cameras_from_configs(config.cameras)
        self.joint_names = config.joint_names or []
        # If you want to publish commands instead of directly setting in Isaac,
        # implement a publisher (e.g., ROS2) here.
        self._ros2_publisher = None
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop_event: Optional[threading.Event] = None
        # to statistic object and goal positions for dataset analysis, key: episode index, value: (object pos, goal pos)
        self._obj_pos = { "object": {} }
        self._goal_pos = { "goal": {} }

        EP_CONF_PATH = Path(__file__).parent / self.config.ep_conf_path
        with open(EP_CONF_PATH, "r", encoding="utf-8") as f:
            self._ep_conf = json.load(f)

    @property
    def observation_features(self) -> dict:
        # example: images and joint states
        cam_features = {
            key: (value.height, value.width, 3) for key, value in self.cameras.items()
        }

        return {**self.joints_features, **cam_features}

    @property
    def action_features(self) -> dict:
        return self.joints_features

    @property
    def joints_features(self) -> dict[str, type]:
        joints_features = {
            name: float for name in self.config.joint_names
        }
        return joints_features

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
            # 配置需要加载的扩展
            config = {
                "headless": False,
                "exts": [
                    "omni.isaac.ros2_bridge",  # 必须显式包含这个
                    "omni.isaac.core_nodes"  # 解决 IsaacReadSimulationTime 警告
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

            self.world.scene.add(self._robot)
            self.world.scene.add(self._object)
            self.world.scene.add(self._goal)

            for key, cam in self.cameras.items():
                cam.attach_to_world(self.world)
                cam.connect()
            self.world.reset()      # Have to be after all initialization, otherwise no data

            logger.info("Connected to Isaac robot")
            
        except Exception:
            # allow ROS-only mode
            # if self.config and getattr(self.config, "use_ros2_action_interface", False):
            #     pass
            # else:
            raise RuntimeError("Omniverse Isaac imports failed. Make sure Isaac Sim 4.2 Python environment is active.")

        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        # For simulated robot this can be always True
        return True

    def calibrate(self) -> None:
        # No-op for simulation or implement if needed
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise RuntimeError("IsaacSimRobot not connected")

        # joints
        joint_positions = self._robot.get_joint_positions()
        obs = dict(zip(self.joint_names, joint_positions))
        # cameras
        for name in self.cameras.keys():
            try:
                frame = self.cameras[name].read()
                if frame is None or frame.shape[0] == 0:
                    logger.warning(f"frame {frame} is empty. Probably because no world.reset() after camera initialization.")
                obs[name] = frame
                # Optionally include timestamps for alignment downstream
                # obs[f"images.{name}_ts"] = ts
            except Exception as e:
                logger.warning(f"Failed to read camera {name}: {e}")
                obs[name] = None
                # obs[f"images.{name}_ts"] = None
        return obs


    def _get_random_obj_pos_ori(self, obj_idx):
        """
        Returns:
            object positions (x, y, z) and orientation
        """
        # check conf files to see data structure
        obj = self._pos_conf["object"][obj_idx]
        obj_pos = obj["pos"]
        obj_pos_noise = obj["pos_noise"]
        obj_ori_noise = obj["ori_noise"]
        x_ran = obj_pos[0] + random.uniform(-obj_pos_noise[0], obj_pos_noise[0])
        y_ran = obj_pos[1] + random.uniform(-obj_pos_noise[1], obj_pos_noise[1])
        ori_ran = random.uniform(-obj_ori_noise, obj_ori_noise)
        z = 0.78806
        pos = [x_ran, y_ran, z]
        ori = [0.0, 0.0, ori_ran]

        logger = logging.getLogger(__name__)
        logger.info(f"random object position: {pos}, object orientation: {ori}")
        return pos, ori

    def _get_random_goal_pos_ori(self, goals):
        goal_idx = random.choice(goals)
        goal = self._pos_conf["goal"][goal_idx]
        goal_pos = goal["pos"]
        goal_pos_noise = goal["pos_noise"]
        goal_ori_noise = goal["ori_noise"]

        x_ran = goal_pos[0] + random.uniform(-goal_pos_noise[0], goal_pos_noise[0])
        y_ran = goal_pos[1] + random.uniform(-goal_pos_noise[1], goal_pos_noise[1])
        ori_ran = random.uniform(-goal_ori_noise, goal_ori_noise)
        z = 0.79019
        pos = [x_ran, y_ran, z]
        ori = [0.0, 0.0, ori_ran]

        logger = logging.getLogger(__name__)
        logger.info(f"random goal position: {pos}, goal orientation: {ori}")
        return pos, ori


    def _get_rule_object(self, index: int):
        """
        根据 index 找到对应 object
        """
        for rule in self._ep_conf["rules"]:
            if rule["start"] <= index < rule["end"]:
                return rule

        return None

    def _move_object(self, object,  pos, ori) -> None:
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        quat = euler_angles_to_quat(np.array(ori), degrees=True)
        logger.info(f"move_obj: {pos}, {ori}, {quat}")
        object.set_world_pose(
                position=np.array(pos),
                orientation=quat)  # (w, x,y,z)

    def _get_random_arm_pose(self):
        low_limits = np.array([-0.872665, 0.0, -1.5708, -0.785398, -0.785398, -0.785398, 0.0, -0.038])
        high_limits = np.array([0.872665, 1.8326, 0.0, 0.785398, 0.785398, 0.785398, 0.038, 0.0])
        random_joints = np.random.uniform(low=low_limits, high=high_limits)
        ret = dict(zip(self.joint_names, random_joints))
        return ret


    def reset_env(self, ep: int, seed: int) -> None:
        """
        Use it when recording dataset
        Args:
            seed: master seed
            ep: offset seed
        Returns:

        """
        self.world.reset()  # reset() must before move_obj() otherwise reset() reloads USD
        # random seed = master seed + no.episode, make dataset extendable
        random.seed(ep+seed)
        distance_flag = False
        while not distance_flag:
            obj_pos_x, obj_pos_y = get_random_position(self._ep_conf["limits"]["object"]["angle"][0],
                                                     self._ep_conf["limits"]["object"]["angle"][1],
                                                     self._ep_conf["limits"]["object"]["radius"][0],
                                                     self._ep_conf["limits"]["object"]["radius"][1],
                                                     self._ep_conf["limits"]["origin"][0],
                                                     self._ep_conf["limits"]["origin"][1])
            goal_pos_x, goal_pos_y = get_random_position(self._ep_conf["limits"]["goal"]["angle"][0],
                                                         self._ep_conf["limits"]["goal"]["angle"][1],
                                                         self._ep_conf["limits"]["goal"]["radius"][0],
                                                         self._ep_conf["limits"]["goal"]["radius"][1],
                                                         self._ep_conf["limits"]["origin"][0],
                                                         self._ep_conf["limits"]["origin"][1])
            distance = get_euclidean_distance(obj_pos_x, obj_pos_y,
                                          goal_pos_x, goal_pos_y)
            if distance < self._ep_conf["limits"]["min_distance"]:
                distance_flag = False
                logger.info(f"reset_env(): Distance between object and goal: {distance}, less than the minimum distance. Re-generate.")
                continue
            else:
                distance_flag = True
                break

        obj_ori = get_random_orientation(-self._ep_conf["limits"]["object"]["orientation"],
                                         self._ep_conf["limits"]["object"]["orientation"])
        goal_ori = get_random_orientation(-self._ep_conf["limits"]["goal"]["orientation"],
                                          self._ep_conf["limits"]["goal"]["orientation"])

        logger.info(f"reset_env(): object pos: ({obj_pos_x}, {obj_pos_y}), ori: {obj_ori}")
        self._move_object(self._object,
                          [obj_pos_x, obj_pos_y, self._ep_conf["limits"]["object"]["z_axis"]],
                          [0, 0, obj_ori])
        # self._obj_pos["object"][f"{ep}"] = { "position": [obj_pos_x, obj_pos_y], "orientation": obj_ori }

        logger.info(f"reset_env(): goal pos: ({goal_pos_x}, {goal_pos_y}), ori: {goal_ori}")
        self._move_object(self._goal,
                          [goal_pos_x, goal_pos_y, self._ep_conf["limits"]["goal"]["z_axis"]],
                          [0, 0, goal_ori])
        # self._goal_pos["goal"][f"{ep}"] = { "position": [goal_pos_x, goal_pos_y], "orientation": goal_ori }

        for cam in self.cameras.values():
            cam.warmup()

    def send_action(self, action: RobotAction) -> RobotAction:
        # For inference
        from omni.isaac.core.utils.types import ArticulationAction

        # joint_targets = np.array([0.1, -0.2, 0.3, -0.1, 0.3, 0.0, 0.04, 0.04])
        joint_targets = np.array(list(action.values()))
        # joint_targets[-2:] *= 100   # old dataset post process
        arti_action = ArticulationAction(joint_positions=joint_targets)
        # joint_indices = self._robot.dof_names

        self._robot.apply_action(arti_action)

        # logger.info(f"send_action: self._robot.get_joint_positions(): {self._robot.get_joint_positions()}")
        return action

    @property
    def joint_positions(self):
        return self._robot.get_joint_positions()

    def configure(self) -> None:
        # Any runtime config (control gains, camera settings etc.)
        pass


    def disconnect(self) -> None:
        for cam in self.cameras.values():
            try:
                if cam.is_connected:
                    cam.disconnect()
            except Exception:
                pass
        ts = time.time_ns()
        # json.dump(self._obj_pos, open(f"/home/shenjie/Documents/object_positions_{ts}.json", "w"), indent=4)
        # json.dump(self._goal_pos, open(f"/home/shenjie/Documents/goal_positions_{ts}.json", "w"), indent=4)
        # self.world.stop()
        # self._app.close()
        # TODO: shutdown SimulationApp if created (self._app) and cleanup stage if owned
        self._connected = False