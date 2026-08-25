from dataclasses import dataclass, field
from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig
from typing import Dict, List


@RobotConfig.register_subclass("isaac_piper")
@dataclass
class IsaacPiperConfig(RobotConfig):
    # URL/path to the USD stage or empty if connecting to running Isaac
    stage_path: str = "/home/shenjie/ws/Piper_ros_moveit/src/piper/piper_moveit_config/config/piper/piper_white.usd"
    # Path to robot prim in USD (e.g. "/World/robot")
    robot_prim_path: str = "/World/piper/base_link"
    # Path to object to interact with (e.g. a block to push)
    object_prim_path: str = "/World/red_block"
    goal_prim_path: str = "/World/small_KLT"
    # Isaac Sim fps
    fps: int = 30
    # camera prims mapping name -> prim_path
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # joint names are in the config Isaac_Piper_Joints
    # ROS2 topic to listen for joint commands
    ros2_command_topic: str = "/isaac_joint_commands"

    # configs for recording/evaluation with random positions and orientations
    obj_pose_path: str = "/home/shenjie/ws/data_viz/data/obj_posori_record_2cam_top_wst_goal_blu_1pos_15hz_100ep_42_100.json" # use Path(__file__).parent to read relative path
    goal_pose_path: str = "/home/shenjie/ws/data_viz/data/goal_posori_record_2cam_top_wst_goal_blu_1pos_15hz_100ep_42_100.json"
    obj_zaxis_offset: float = 0.78806  # offset to apply to object z-axis position
    goal_zaxis_offset: float = 0.77553  # offset to apply to goal z-axis position

    # Simulation management options (Robot can manage stepping if it created/owns the world)
    manage_simulation: bool = True      # if True, IsaacPiperRobot starts internal sim loop
    simulation_dt: float = 1.0 / fps      # seconds per physics/render step when managing sim
    simulation_render: bool = True       # pass render=True to world.step(...) if supported
    auto_open_stage: bool = True        # open stage from stage_path on connect() if True