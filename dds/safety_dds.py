import json
import time
import threading
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

class SafetyDDS(DDSObject):
    """
    Safety signal handler.
    Latches unsafe state for a short duration to ensure simulation steps capture it.
    """
    def __init__(self, node_name: str = "safety_dds"):
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        self._initialized = True
        self.node_name = node_name
        self.last_unsafe_time = 0.0
        self.last_fuzzy_time = 0.0
        self.unsafe_timeout = 0.5  # Seconds to latch the unsafe state
        self.is_unsafe_flag = False
        self.is_fuzzy_flag = False
        self.lock = threading.Lock()
        
        # Initialize subscriber immediately
        self.setup_subscriber()
        print(f"[{self.node_name}] Safety DDS node initialized")

    def setup_publisher(self) -> bool:
        return True

    def dds_publisher(self) -> None:
        pass

    def setup_subscriber(self) -> bool:
        try:
            self.subscriber = ChannelSubscriber("rt/safety_signal", String_)
            self.subscriber.Init(self.dds_subscriber, 10)
            print(f"[{self.node_name}] Safety subscriber initialized on rt/safety_signal")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Failed to setup subscriber: {e}")
            return False

    def dds_subscriber(self, msg: String_):
        try:
            data = json.loads(msg.data)
            if data.get("is_unsafe", False):
                with self.lock:
                    self.last_unsafe_time = time.time()
                    self.is_unsafe_flag = True
            if data.get("is_fuzzy", False):
                with self.lock:
                    self.last_fuzzy_time = time.time()
                    self.is_fuzzy_flag = True
                        
        except json.JSONDecodeError:
            pass # print(f"[{self.node_name}] Failed to decode JSON: {msg.data}")
        except Exception as e:
            print(f"[{self.node_name}] Error in subscriber: {e}")

    def is_unsafe(self) -> bool:
        """
        Returns True if an unsafe signal was received recently (within timeout).
        """
        with self.lock:
            if time.time() - self.last_unsafe_time < self.unsafe_timeout:
                return True
            self.is_unsafe_flag = False
            return False

    def is_fuzzy(self) -> bool:
        """
        Returns True if a fuzzy signal was received recently (within timeout).
        """
        with self.lock:
            if time.time() - self.last_fuzzy_time < self.unsafe_timeout:
                return True
            self.is_fuzzy_flag = False
            return False