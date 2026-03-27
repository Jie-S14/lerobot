from dataclasses import dataclass
from typing import Literal
from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("isaac")
@dataclass
class IsaacCameraConfig(CameraConfig):
    """
    Camera config for Isaac Sim cameras.

    - prim_path: required, USD prim path of the camera in the stage (used at runtime).
    - width/height/fps: required for dataset_features construction and validation.
    - is_depth: whether this camera produces depth (affects dtype / encoding).
    - pixel_format: runtime pixel format (e.g. 'rgb8', 'rgba8', 'r32f').
    - ros_topic: optional if using ROS bridge instead of direct Isaac API.
    """

    prim_path: str
    # name: str
    is_warmup: bool = False
    warmup_steps: int = 10
    is_depth: bool = False
    pixel_format: Literal["rgb8", "rgba8", "r32f"] = "rgb8"
    ros_topic: str | None = None

    def __post_init__(self) -> None:
        # base CameraConfig may validate common fields
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        if not self.prim_path:
            raise ValueError("`prim_path` is required for IsaacCameraConfig (used to locate the camera prim).")
        for attr in ("width", "height", "fps"):
            val = getattr(self, attr, None)
            if val is None or (isinstance(val, int) and val <= 0):
                raise ValueError(f"`{attr}` must be a positive integer for IsaacCameraConfig.")
        if self.pixel_format not in ("rgb8", "rgba8", "r32f"):
            raise ValueError(f"Unsupported pixel_format '{self.pixel_format}' for IsaacCameraConfig.")