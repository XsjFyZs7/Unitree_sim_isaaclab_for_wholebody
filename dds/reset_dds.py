# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Reset DDS communication class
"""

from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_


class ResetDDS(DDSObject):
    """Reset DDS node"""

    def __init__(self, node_name: str = "reset_dds"):
        """Initialize the reset DDS node"""
        if hasattr(self, '_initialized'):
            return
        super().__init__()

        self._initialized = True
        self.node_name = node_name
        self.publisher = None
        self.subscriber = None
        # setup the shared memory
        self.setup_shared_memory(
            output_shm_name="isaac_reset_episode_cmd", 
            output_size=128, 
            outputshm_flag=True,
            inputshm_flag=False,
        )
        print(f"[{self.node_name}] Reset DDS node initialized")

    def setup_publisher(self) -> bool:
        """Setup the reset command publisher"""
        pass

    def setup_subscriber(self, callback=None) -> bool:
        """Setup the reset command subscriber"""
        try:
            self.subscriber = ChannelSubscriber("rt/reset_episode", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, callback), 1)
            print(f"[{self.node_name}] Reset subscriber initialized")
            return True
        except Exception as e:
            print(f"[{self.node_name}] Failed to initialize reset subscriber: {e}")
            return False

    def publish(self, reset_flag: bool):
        """Publish a reset signal by writing to shared memory to be non-blocking."""
        try:
            cmd_data = {"reset": reset_flag}
            if self.output_shm:
                self.output_shm.write_data(cmd_data)
        except Exception as e:
            print(f"[{self.node_name}] Failed to write reset command to shared memory: {e}")

    def dds_subscriber(self, msg: String_, callback):
        """Process the subscribe data and write to shm"""
        try:
            reset_flag = msg.data == "true"
            cmd_data = {"reset": reset_flag}
            if self.output_shm:
                self.output_shm.write_data(cmd_data)
            
            # also call the original callback if it exists for compatibility
            if callback:
                callback(msg)
        except Exception as e:
            print(f"[{self.node_name}] Failed to process subscribe data: {e}")

    def get_reset_command(self) -> Optional[Dict[str, Any]]:
        """Get the reset command from shared memory.
        
        Returns:
            Dict: The reset command. Returns None if no command is present.
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None

    def dds_publisher(self):
        """This method is deprecated. Use publish() to write to shared memory."""
        pass
