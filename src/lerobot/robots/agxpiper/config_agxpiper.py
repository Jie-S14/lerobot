from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("agxpiper")
@dataclass
class AGXPiperConfig(RobotConfig):
    can_port: str = "can0"
    sdk_interface_version: str = "v2"  # "v1"
    judge_flag: bool = True
    # Only for V2
    start_sdk_joint_limit: bool = True
    start_sdk_gripper_limit: bool = True

    # filter action alpha
    alpha: float = 1.0
    
    # gripper_max_stroke_m: float = 0.08
    connect_timeout_s: float = 5.0
    max_relative_target: float | None = None
    disable_torque_on_disconnect: bool = False
    cameras: dict[str, CameraConfig] = field(default_factory=dict)