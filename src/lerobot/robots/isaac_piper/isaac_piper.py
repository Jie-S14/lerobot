import logging
import threading
import time
from typing import Any, Optional, Tuple

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from .config_isaac_piper import IsaacPiperConfig

# NOTE: Omniverse / Isaac imports are optional here; keep them lazy to avoid import errors
# import omni.isaac.core etc. inside connect() when available.

# Import IsaacCamera class so we can create cameras bound to the same world if provided.
from lerobot.cameras.isaac.camera_isaac import IsaacCamera  # type: ignore

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

    @property
    def observation_features(self) -> dict:
        # example: images and joint states
        cam_features = {
            f"{key}": {"dtype": "video", "shape": [value.height, value.width, 3], "name": ["height", "width", "channels"]} for key, value in self.cameras.items()
        }
        state_feat = {"dtype": "float32", "shape": (len(self.joint_names),), "names": self.joint_names}
        return {"state": state_feat, **cam_features}

    @property
    def action_features(self) -> dict:
        return {"dtype": "float32", "shape": (len(self.joint_names),), "names": self.joint_names}

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

            simulation_context = SimulationContext()
            open_stage(usd_path=self.config.stage_path)
            self._app.update()
            
            self.world = World(physics_dt=self.config.simulation_dt,
                             rendering_dt=self.config.simulation_dt)
            self._robot = Robot(prim_path=self.config.robot_prim_path, 
                            name=self.config.robot_prim_path.split("/")[-1])
            self._object = XFormPrim(prim_path=self.config.object_prim_path,
                                    name=self.config.object_prim_path.split("/")[-1])

            self.world.scene.add(self._robot)
            self.world.scene.add(self._object)

            for key, cam in self.cameras.items():
                cam.attach_to_world(self.world)
                cam.connect()
            self.world.reset()

            logger.info("Connected to Isaac robot")
            
        except Exception:
            # allow ROS-only mode
            if self.config and getattr(self.config, "use_ros2_action_interface", False):
                pass
            else:
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
        obs: RobotObservation = {}
        self.world.step(render=True)
        # joints
        joint_positions = self._robot.get_joint_positions()
        obs = dict(zip(self.config.joint_names, joint_positions))
        # cameras
        for name in self.cameras.keys():
            try:
                frame = self.cameras[name].read()
                obs[name] = frame
                # Optionally include timestamps for alignment downstream
                # obs[f"images.{name}_ts"] = ts
            except Exception as e:
                logger.warning(f"Failed to read camera {name}: {e}")
                obs[name] = None
                # obs[f"images.{name}_ts"] = None
        return obs

    def move_obj(self, pos, quat) -> None:
        # pos = [0.5, 0.5, 0.78466]
        # quat = [1.0, 0.0, 0.0, 0.0]
        self._object.set_world_pose(
                position=np.array(pos),
                orientation=quat)  # (w, x,y,z)
        tup, tuo = self._object.get_world_pose()
        logger.info(f"self._object.get_world_pose(): {tup}, {tuo}")

    def reset_env(self, position: list|None, quat: list|None) -> None:
        self.world.reset()  # reset() must before move_obj() otherwise reset() reloads USD
        if position is not None and quat is not None:
            self.move_obj(position, quat)
        for cam in self.cameras.values():
            cam.warmup()

    def send_action(self, action: RobotAction) -> RobotAction:
        # Simulator follows moveit2
        return action

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
        # self.world.stop()
        # self._app.close()
        # TODO: shutdown SimulationApp if created (self._app) and cleanup stage if owned
        self._connected = False