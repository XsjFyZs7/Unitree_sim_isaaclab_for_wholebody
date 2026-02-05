# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
import torch
import os
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg
from . import mdp
# use Isaac Lab native event system

from tasks.common_config import  G1RobotPresets, CameraPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_pickplace_cylindercfg_wholebody import TableCylinderSceneCfgWH
from tasks.common_scene.base_scene_pickplace_cylindercfg import TableCylinderSceneCfg
##
# Scene definition
##

@configclass
class ObjectTableSceneCfg(TableCylinderSceneCfgWH):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(init_pos=(-4.5, -2.8, 0.8),
    # robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(init_pos=(-3.5, 1.5, 0.8),
        # init_rot=(0.7071, 0, 0, -0.7071))
        init_rot=(1.0, 0, 0, 0))

    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=10, track_air_time=True, debug_vis=True)
    # 6. add camera configuration 
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()
    
@configclass
class ObjectTableSceneCfg_v2(TableCylinderSceneCfg):
    """object table scene configuration class
    inherits from G1SingleObjectSceneCfg, gets the complete G1 robot scene configuration
    can add task-specific scene elements or override default configurations here
    """
    
    # Humanoid robot w/ arms higher
    # 5. humanoid robot configuration 
    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(init_pos=(6.5, 12.0, 0.8),
    # robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex1_wholebody(init_pos=(-3.5, -6.5, 0.8),
        init_rot=(0.7071, 0, 0, -0.7071))
        # init_rot=(1.0, 0, 0, 0))

    contact_forces = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/.*", history_length=10, track_air_time=True, debug_vis=True)
    # 6. add camera configuration 
    front_camera = CameraPresets.g1_front_camera()
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera()
    robot_camera = CameraPresets.g1_world_camera()
##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)



@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class
        defines all state observation values for policy decision
        inherit from ObsGroup base class 
        """

        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states)
        robot_gipper_state = ObsTerm(func=mdp.get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=mdp.get_camera_image)
        robot_imu_state = ObsTerm(func=mdp.get_robot_boy_joint_states)

        def __post_init__(self):
            """post initialization function
            set the basic attributes of the observation group
            """
            self.enable_corruption = False  # disable observation value corruption
            self.concatenate_terms = False  # disable observation item connection

    # observation groups
    # create policy observation group instance
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    
    # check if reach the goal position
    # success = DoneTerm(
    #     func=mdp.check_robot_reached_goal,
    #     params={
    #         "goal_position_attr": "goal_position",
    #         "distance_threshold_attr": "distance_threshold",
    #     }
    # )
    
    out_time = DoneTerm(
        func=mdp.check_timeout_termination,
        params={
            "time_threshold": 40
        }
    )
    
    collision_by_force = DoneTerm(
        func=mdp.contact_force_termination, 
        params={
            "force_threshold": 25.0
        }
    )
    
    fall_by_rowpitch = DoneTerm(
        func=mdp.check_fall_risk_termination,
        params={
            "roll_threshold": 0.785,
            "pitch_threshold": 0.785
        }
    )
    
    # joint_limit = DoneTerm(
    #     func=mdp.check_joint_limit_termination,
    #     params={
    #         "position_threshold": 0.1,
    #         "torque_threshold_ratio": 0.9,  # 使用比例值替代固定值
    #         "duration_threshold": 10,
    #         "check_frequency": 5,
    #     }
    # )

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward,weight=1.0)

@configclass
class EventCfg:
    pass

    reset_robot = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # position range parameter
            "pose_range": {
                "x": [-0.05, 0.05],
                "y": [-0.05, 0.05],
            },
            # speed range parameter (empty dictionary means using default value)
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot")
        },
    )
    
    # File-based Disturbance Scheduler (Benchmark Mode)
    # Assigns a specific test case (time, velocity) from the JSON file to each episode sequentially
    # disturbance_scheduler = EventTermCfg(
    #     func=mdp.reset_disturbance_scheduler,
    #     mode="reset",
    #     params={
    #         "benchmark_path": "/home/wyh/WYH/isaac_nav_bridge/external_force/safety_benchmark_cases.json",
    #     },
    # )

    # # File-based Disturbance Trigger
    # # Runs every step to check if the scheduled time has arrived
    # disturbance_trigger = EventTermCfg(
    #     func=mdp.apply_disturbance_scheduler,
    #     mode="interval", 
    #     interval_range_s=(0.02, 0.02), # Check every step
    #     params={
    #         "benchmark_path": "/home/wyh/WYH/isaac_nav_bridge/external_force/safety_benchmark_cases.json",
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
    
    # reset_object = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,  # use uniform distribution reset function
    #     mode="reset",   # set event mode to reset
    #     params={
    #         # position range parameter
    #         "pose_range": {
    #             "x": [-0.05, 0.05],  # x axis position range: -0.05 to 0.0 meter
    #             "y": [-0.05, 0.05],   # y axis position range: 0.0 to 0.05 meter
    #         },
    #         # speed range parameter (empty dictionary means using default value)
    #         "velocity_range": {},
    #         # specify the object to reset
    #         "asset_cfg": SceneEntityCfg("object"),
    #     },
    # )


@configclass
class NavG129Dex1WholebodyEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     replicate_physics=True # enable physics replication
                                                     )
    # scene: ObjectTableSceneCfg = ObjectTableSceneCfg_v2(num_envs=1, # environment number: 1
    #                                                     env_spacing=2.5, # environment spacing: 2.5 meter
    #                                                     replicate_physics=True # enable physics replication
    #                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    # MDP settings
        
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    commands = None # command manager
    rewards: RewardsCfg = RewardsCfg()  # reward manager
    curriculum = None # curriculum manager
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.scene.contact_forces.update_period = self.sim.dt
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

                # 物理材料属性设置 / Physics material properties
        self.sim.physics_material.static_friction = 1.0  # 静摩擦系数 / Static friction
        self.sim.physics_material.dynamic_friction = 1.0  # 动摩擦系数 / Dynamic friction
        self.sim.physics_material.friction_combine_mode = "max"  # 摩擦力合并模式 / Friction combine mode
        self.sim.physics_material.restitution_combine_mode = "max"  # 恢复系数合并模式 / Restitution combine mode
        # create event manager
        self.event_manager = SimpleEventManager()

        # register "reset object" event
        # self.event_manager.register("reset_object_self", SimpleEvent(
        #     func=lambda env: base_mdp.reset_root_state_uniform(
        #         env,
        #         torch.arange(env.num_envs, device=env.device),
        #         pose_range={"x": [-0.05, 0.05], "y": [0.0, 0.05]},
        #         velocity_range={},
        #         asset_cfg=SceneEntityCfg("object"),
        #     )
        # ))
        
        # self.event_manager.register("reset_all_self", SimpleEvent(
        #     func=lambda env: base_mdp.reset_scene_to_default(
        #         env,
        #         torch.arange(env.num_envs, device=env.device))
        # ))