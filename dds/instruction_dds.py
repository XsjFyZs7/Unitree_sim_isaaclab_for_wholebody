# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0
"""
Instruction DDS communication class
Publish the instruction text to VLM
"""

from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

class InstructionDDS(DDSObject):
    """Instruction DDS node (singleton pattern)"""
    
    def __init__(self, node_name: str = "instruction_dds"):
        """Initialize the instruction DDS node"""
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        
        self._initialized = True
        self.node_name = node_name
        # Setup shared memory (optional, serves as a local cache)
        self.setup_shared_memory(
            output_shm_name="isaac_instruction_cmd", 
            input_shm_name="isaac_instruction_state",
            output_size=1024, 
            input_size=1024,
            outputshm_flag=True,
            inputshm_flag=True,
        )
        print(f"[{self.node_name}] Instruction DDS node initialized")

    def setup_publisher(self) -> bool:
        """Setup the instruction publisher"""
        try:
            self.publisher = ChannelPublisher("rt/instruction", String_)
            self.publisher.Init()
            print(f"[{self.node_name}] Instruction publisher initialized (rt/instruction)")
            return True
        except Exception as e:
            print(f"instruction_dds [{self.node_name}] Failed to initialize publisher: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        # This node acts as a publisher from the Sim side
        return True
    
    def publish_instruction(self, text: str):
        """Publish instruction text"""
        if not hasattr(self, 'publisher') or self.publisher is None:
            # Try to setup publisher if not ready (e.g. if called before dds_manager started)
            if not self.setup_publisher():
                print(f"[{self.node_name}] Publisher not initialized!")
                return
            
        try:
            msg = String_(text)
            self.publisher.Write(msg)
            # Also write to shared memory for debugging/local access
            if self.input_shm:
                 self.input_shm.write_data({"instruction": text})
            print(f"[{self.node_name}] Published instruction: {text}")
        except Exception as e:
            print(f"[{self.node_name}] Error publishing instruction: {e}")

    def dds_publisher(self) -> Any:
        pass

    def dds_subscriber(self, msg: Any, datatype: str = None) -> Dict[str, Any]:
        return {}