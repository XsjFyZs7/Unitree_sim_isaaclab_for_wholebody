# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Sim state DDS communication class
Specialized in publishing and receiving sim state data
"""

import threading
import torch
from typing import Any, Dict, Optional
from dds.dds_base import DDSObject
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__String_

import json

class SimStateDDS(DDSObject):
    """Sim state DDS node (singleton pattern)"""
    
    def __init__(self, env, task_name,node_name:str="sim_state_dds"):
        """Initialize the sim state DDS node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized') and self._initialized:
            return
        super().__init__()
        self.node_name = node_name
        self.env = env
        self.task_name = task_name
        self._initialized = True
        self.sim_state = std_msgs_msg_dds__String_()

        # setup the shared memory
        self.setup_shared_memory(
            input_shm_name="isaac_sim_state",  # read sim state data for publishing
            input_size=4096,
            outputshm_flag=False
        )

        print(f"[{self.node_name}] Sim state DDS node initialized")
        self.nav_task_latest = {}
        try:
            self.nav_task_sub = ChannelSubscriber("rt/nav_task", String_)
            self.nav_task_sub.Init(self._nav_task_cb, 10)
        except Exception as e:
            pass

    
    def setup_publisher(self) -> bool:
        """Setup the publisher of the sim state"""
        try:
            self.publisher = ChannelPublisher("rt/sim_state", String_)
            self.publisher.Init()
            
            print(f"[{self.node_name}] Sim state publisher initialized")
            return True
        except Exception as e:
            print(f"sim_state_dds [{self.node_name}] Sim state publisher initialization failed: {e}")
            return False
    
    def setup_subscriber(self) -> bool:
        """Setup the subscriber of the sim state"""
        try:
            self.subscriber = ChannelSubscriber("rt/sim_state_cmd", String_)
            self.subscriber.Init(lambda msg: self.dds_subscriber(msg, ""), 1)
            
            print(f"[{self.node_name}] Sim state subscriber initialized")
            return True
        except Exception as e:
            print(f"sim_state_dds [{self.node_name}] Sim state subscriber initialization failed: {e}")
            return False
    
    
    def dds_publisher(self) -> Any:
        """Process the publish data"""
        try:
            data = self.input_shm.read_data()
            if data is None:
                return
            # get sim state from environment
            if self.nav_task_latest:
                try:
                    data["nav_pair_id"] = self.nav_task_latest.get("pair_id")
                    data["nav_start_point"] = self.nav_task_latest.get("start_point")
                    data["nav_end_point"] = self.nav_task_latest.get("end_point")
                    data["nav_status"] = self.nav_task_latest.get("status")
                except Exception:
                    pass
            sim_state = json.dumps(data)
            self.sim_state.data = sim_state
            self.publisher.Write(self.sim_state)
        except Exception as e:
            print(f"sim_state_dds [{self.node_name}] Error processing publish data: {e}")
            return None
    
    def dds_subscriber(self, msg: String_,datatype:str=None) -> Dict[str, Any]:
        """Process the subscribe data"""
        try:
            # Parse received sim state command
            data = json.loads(msg.data)
            
            # Process the command (implement according to your needs)
            # For example, you might want to apply the received state to the environment
            return data
        except Exception as e:
            print(f"sim_state_dds [{self.node_name}] Error processing subscribe data: {e}")
            return None

    def _nav_task_cb(self, msg: String_):
        try:
            self.nav_task_latest = json.loads(msg.data)
        except Exception:
            pass

    def tensors_to_list(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.tensors_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.tensors_to_list(i) for i in obj]
        return obj

    def sim_state_to_json(self,data):
        data_serializable = self.tensors_to_list(data)
        json_str = json.dumps(data_serializable)
        return json_str

    def write_sim_state_data(self, sim_state_data=None):
        """Write sim state data to shared memory to trigger publishing
        
        Args:
            sim_state_data: Optional sim state data. If None, will get current state from environment
        """
        try:
            if sim_state_data is None:
                # Get current sim state from environment
                sim_state_data = {"trigger": "publish_sim_state"}
            
            # write to the input shared memory for publishing
            if self.input_shm:
                self.input_shm.write_data(sim_state_data)
                
        except Exception as e:
            print(f"sim_state_dds [{self.node_name}] Error writing sim state data: {e}")

    def get_sim_state_command(self) -> Optional[Dict[str, Any]]:
        """Get the sim state control command
        
        Returns:
            Dict: the sim state command, return None if there is no new command
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None