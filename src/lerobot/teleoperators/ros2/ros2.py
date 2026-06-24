from functools import cached_property
import logging
import threading
from typing import Any, Dict, Optional
from lerobot.robots.isaac_piper.isaac_piper_utils import Isaac_Piper_Joints
from lerobot.teleoperators.ros2.config_ros2 import Ros2TeleoperatorConfig
from lerobot.utils.decorators import check_if_not_connected
from lerobot.teleoperators.teleoperator import Teleoperator

try:
    import rclpy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float32MultiArray
    _HAS_RCLPY = True
except Exception:
    # rclpy may not be available at import time (e.g. on CI / non-ROS machines)
    rclpy = None  # type: ignore
    JointState = None  # type: ignore
    Float32MultiArray = None  # type: ignore
    _HAS_RCLPY = False

logger = logging.getLogger(__name__)

class Ros2Teleoperator(Teleoperator):
    """A Teleoperator that subscribes to a ROS2 topic (e.g. /isaac_joint_command)
    and exposes the latest received command via get_action()."""

    def __init__(self, config: Ros2TeleoperatorConfig):
        self.config = config
        self.lock = threading.RLock()
        self.latest: Optional[Dict[str, Any]] = None
        self.node = None
        self.spin_thread: Optional[threading.Thread] = None
        self.sub = None
        self._is_running = False

    def connect(self) -> None:
        if not _HAS_RCLPY:
            raise RuntimeError("rclpy not available. Install ROS2 python packages to use Ros2Teleoperator.")
        if self.node is not None:
            return
        rclpy.init(args=None)
        self.node = rclpy.create_node(self.config.node_name)
        self._is_running = True

        # choose message class
        msg_cls = None
        if self.config.msg_type.lower().endswith("jointstate") and JointState is not None:
            msg_cls = JointState
        elif ("float32" in self.config.msg_type.lower() and "array" in self.config.msg_type.lower()) and Float32MultiArray is not None:
            msg_cls = Float32MultiArray
        else:
            # fallback try JointState then Float32MultiArray
            msg_cls = JointState if JointState is not None else Float32MultiArray

        if msg_cls is None:
            raise RuntimeError("Unable to resolve ROS2 message class for ros2 teleop. Make sure sensor_msgs/std_msgs are installed.")

        self.sub = self.node.create_subscription(msg_cls, self.config.topic, self._callback, self.config.qos)

        # spin in background thread to keep callbacks alive
        self.spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self.spin_thread.start()
        logger.info("Ros2Teleoperator connected and subscribed to %s", self.config.topic)

    def _spin_loop(self):
        while rclpy.ok() and self._is_running:
            if self.node is not None:
                rclpy.spin_once(self.node, timeout_sec=0.0667)  # ~15Hz

    def _callback(self, msg) -> None:
        """Convert incoming ROS msg into a simple action dict and store latest."""
        action: Dict[str, Any] = {}
        try:
            # sensor_msgs/JointState: names + positions
            if JointState is not None and isinstance(msg, JointState):
                positions = list(msg.position) if msg.position is not None else []
                names = list(msg.name) if msg.name is not None else None
                if names and len(names) == len(positions):
                    action = {n: float(p) for n, p in zip(names, positions)}
                else:
                    action = {"positions": positions}
            # std_msgs/Float32MultiArray: raw array under data
            elif Float32MultiArray is not None and isinstance(msg, Float32MultiArray):
                data = list(msg.data) if msg.data is not None else []
                if self.config.joint_names and len(self.config.joint_names) == len(data):
                    action = {n: float(v) for n, v in zip(self.config.joint_names, data)}
                else:
                    action = {"positions": data}
            else:
                # Generic fallback: try common attributes
                if hasattr(msg, "position"):
                    pos = list(getattr(msg, "position") or [])
                    action = {"positions": pos}
                elif hasattr(msg, "data"):
                    data = list(getattr(msg, "data") or [])
                    action = {"positions": data}
                else:
                    # as last resort, store repr
                    action = {"raw": repr(msg)}
        except Exception as e:
            logger.exception("Error parsing ROS2 teleop message: %s", e)
            action = {"error": str(e)}

        with self.lock:
            self.latest = action

    def get_action(self) -> Dict[str, Any]:
        with self.lock:
            if self.latest is None:
                # Return empty dict so downstream processors can handle no-op
                return {}
            # return a shallow copy
            return dict(self.latest)

    def disconnect(self) -> None:
        if not _HAS_RCLPY:
            return
        if self.node is None:
            return
        self._is_running = False
        self.spin_thread.join()
        # destroy subscription and node
        try:
            if self.sub is not None:
                self.node.destroy_subscription(self.sub)
                self.sub = None

            # set temporary node reference and nullify original to prevent access from spin thread
            temp_node = self.node
            self.node = None

            # destroy node outside of spin thread to avoid deadlock
            temp_node.destroy_node()

        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            if self.spin_thread is not None:
                # 如果想彻底安全，这里可以 self.spin_thread.join(timeout=0.2)
                self.spin_thread.join(timeout=0.2)
                self.spin_thread = None
            # shutdown rclpy — careful if other nodes exist in the same process

        logger.info("Ros2Teleoperator disconnected")

    @property
    def is_connected(self) -> bool:
        return self.node is not None
    
    @property
    def action_features(self) -> dict:
        return {f"{name}": float for name in self._piper_joint_names}
    
    @property
    def feedback_features(self) -> dict:
        return {}
    
    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None: 
        raise NotImplementedError("Piper does not support feedback")

    @cached_property
    def _piper_joint_names(self) -> list[str]:
        return [joint.name for joint in Isaac_Piper_Joints]