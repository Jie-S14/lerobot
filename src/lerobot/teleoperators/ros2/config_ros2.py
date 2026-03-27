from dataclasses import dataclass
from typing import List, Optional
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("ros2")
@dataclass
class Ros2TeleoperatorConfig(TeleoperatorConfig):
    topic: str = "/isaac_joint_commands"
    # "JointState" or "Float32MultiArray"
    msg_type: str = "JointState"
    node_name: str = "lerobot_ros2_teleop"
    qos: int = 10
    # If provided, use these joint names to build a name->value dict
    joint_names: Optional[List[str]] = None