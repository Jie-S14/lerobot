import logging
from threading import Event, Lock, Thread
from typing import Any, Optional, Tuple

import numpy as np

from ..camera import Camera
from .configuration_isaac import IsaacCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class IsaacCamera(Camera):
    """
    Camera wrapper for Omniverse Isaac Sim cameras.

    - Accepts a mandatory `world` object (e.g. omni.isaac.core.World) so multiple cameras/robots can share it.
    - Maintains a background read thread that updates latest_frame and latest_timestamp.
    - read()/async_read() return an HWC uint8 numpy array (RGB) or single-channel float32 for depth.
    - async_read_with_timestamp/read_with_timestamp return (frame, timestamp).
    - `_capture_frame_from_isaac()` must be implemented to actually pull a frame from your local Isaac 4.2 API.
    """

    def __init__(self, config: IsaacCameraConfig, world: Any = None):
        super().__init__(config)
        self.config: IsaacCameraConfig = config

        # Shared world (mandatory) to attach the class camera from Isaac Sim to the workstage.
        self.world: Any = world

        self._connected = False
        self._camera_handle: Optional[Any] = None  # store Isaac camera handle/reader

    def __str__(self) -> str:
        return f"IsaacCamera({self.config.prim_path})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._camera_handle is not None

    def attach_to_world(self, world: Any) -> None:
        """Attach an existing omni.isaac.core.World (or similar) so we can share handles."""
        self.world = world

    def find_cameras(self) -> list[dict[str, Any]]:
        raise NotImplementedError("Camera detection is not implemented for Isaac Sim cameras.")

    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        
        if self.world is None:
            raise RuntimeError(f"{self} requires a world to connect (used to locate camera prim). Provide a world or implement connect() to handle headless operation.")

        # Lazy import of Omniverse/Isaac APIs.
        try:
            from omni.isaac.sensor import Camera
            self._camera_handle = Camera(prim_path=self.config.prim_path, 
                        name=self.config.prim_path.split("/")[-1],
                        resolution=(self.config.width, self.config.height),
                        frequency=self.config.fps)
            self.world.scene.add(self._camera_handle)

        except Exception as e:
            raise RuntimeError(f"Failed to create Isaac camera handle for {self.config.prim_path}: {e}")

        self._connected = True

        self.warmup()
        logger.info(f"{self} connected (camera handle creation deferred if not found).")

    def warmup(self) -> None:
        if self.config.is_warmup:
            logger.info(f"{self} warming up.")
            for _ in range(self.config.warmup_steps):
                self.world.step()
        logger.info(f"{self} warmup complete.")

    def read(self):
        frame = self._camera_handle.get_rgb()

        if frame is None:
            raise RuntimeError("Camera returned no frame")

        return frame

    def async_read(self, timeout_ms: float = 100):
        # To read frames from Isaac Sim, we need to capture frames at the same thread as the simulation loop - world.step().
        # This is because Isaac Sim's API is not thread-safe and may require synchronization with the simulation loop.
        return self.read()

    def get_last_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Return cached latest (frame, timestamp) without blocking.
        """
        with self.frame_lock:
            return self.latest_frame, self.latest_timestamp

    def disconnect(self) -> None:
        if not self.is_connected and (self.thread is None or not self.thread.is_alive()):
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread and self.thread.is_alive():
            if self.stop_event:
                self.stop_event.set()
            self.thread.join(timeout=2.0)
            self.thread = None
            self.stop_event = None

        # TODO: clean up Isaac camera handle if created
        self._camera_handle = None
        self._connected = False
        logger.info(f"{self} disconnected.")