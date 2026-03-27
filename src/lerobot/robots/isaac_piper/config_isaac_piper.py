from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from typing import Dict, List


@RobotConfig.register_subclass("isaac_piper")
@dataclass
class IsaacPiperConfig(RobotConfig):
    # URL/path to the USD stage or empty if connecting to running Isaac
    stage_path: str = "/home/shenjie/ws/Piper_ros_moveit/src/piper/piper_moveit_config/config/piper/piper_cube.usd"
    # Path to robot prim in USD (e.g. "/World/robot")
    robot_prim_path: str = "/World/piper"
    # Path to object to interact with (e.g. a block to push)
    object_prim_path: str = "/World/red_block"
    # Isaac Sim fps
    fps: int = 30
    # camera prims mapping name -> prim_path
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # joint names in the order your policy/dataset expects
    joint_names: List[str] = field(default_factory=list)
    # If using ROS2-based action interface set this to True
    use_ros2_action_interface: bool = True
    # ROS2 topic to listen for joint commands (if use_ros2_action_interface)
    ros2_command_topic: str = "/isaac_joint_commands"

    # Simulation management options (Robot can manage stepping if it created/owns the world)
    manage_simulation: bool = True      # if True, IsaacPiperRobot starts internal sim loop
    simulation_dt: float = 1.0 / fps      # seconds per physics/render step when managing sim
    simulation_render: bool = True       # pass render=True to world.step(...) if supported
    auto_open_stage: bool = True        # open stage from stage_path on connect() if True