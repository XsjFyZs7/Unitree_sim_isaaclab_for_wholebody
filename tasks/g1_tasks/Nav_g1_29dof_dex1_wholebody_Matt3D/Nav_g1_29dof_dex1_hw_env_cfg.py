# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
import tempfile
import torch
import os
from dataclasses import MISSING

from pink.tasks import FrameTask

import isaaclab.envs.mdp as base_mdp
from isaaclab.utils.noise import UniformNoiseCfg
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

from tasks.common_config import  G1RobotEnvPresets, CameraEnvPresets  # isort: skip
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager

# import public scene configuration
from tasks.common_scene.base_scene_wholebody_headless import TableCylinderSceneCfgWH

def _get_env_tuple(key, default):
    """Helper to parse environment variable to tuple."""
    val = os.environ.get(key)
    print(f"[Config DEBUG] Loading {key}: env_val='{val}', default={default}")
    
    if val:
        try:
            result = tuple(float(x) for x in val.split(','))
            print(f"[Config DEBUG] Successfully parsed {key} -> {result}")
            return result
        except ValueError as e:
            print(f"[Config DEBUG] Failed to parse {key}: {e}. Reverting to default.")
            pass
    
    print(f"[Config DEBUG] Using default for {key} -> {default}")
    return default

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
    # Dynamic init state from environment variables
    robot: ArticulationCfg = G1RobotEnvPresets.g1_29dof_dex1_wholebody(
        init_pos=_get_env_tuple("ISAAC_ROBOT_INIT_POS", (9.1, 3.8, 0.8)),
        init_rot=_get_env_tuple("ISAAC_ROBOT_INIT_ROT", (0.7, 0, 0, 0.))
    )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=10, track_air_time=True, debug_vis=False)
    # 6. add camera configuration 
    front_camera = CameraEnvPresets.g1_front_camera()
    left_wrist_camera = CameraEnvPresets.left_gripper_wrist_camera()
    right_wrist_camera = CameraEnvPresets.right_gripper_wrist_camera()
    robot_camera = CameraEnvPresets.g1_world_camera()

    
##
# MDP settings
##
@configclass
class ActionsCfg:
    """defines the action configuration related to robot control, using direct joint angle control
    """
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)

@configclass
class CommandsCfg:
    """Command terms for the robot."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),
    )
@configclass
class ObservationsCfg:
    """
    defines all available observation information
    """
    @configclass
    class PolicyCfg(ObsGroup):
        """policy group observation configuration class"""
        # NaVILA-aligned observations
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        
        # Keep original specialized observations if needed, or remove if fully replacing
        robot_joint_state = ObsTerm(func=mdp.get_robot_boy_joint_states) # Replaced by joint_pos/vel
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

@configclass
class RewardsCfg:
    reward = RewTerm(func=mdp.compute_reward,weight=1.0)

@configclass
class EventCfg:
    pass

    # reset_robot = EventTermCfg(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         # position range parameter
    #         "pose_range": {
    #             "x": [-0.05, 0.05],
    #             "y": [-0.05, 0.05],
    #         },
    #         # speed range parameter (empty dictionary means using default value)
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("robot")
    #     },
    # )
    
    # File-based Disturbance Scheduler (Benchmark Mode)
    # Assigns a specific test case (time, velocity) from the JSON file to each episode sequentially
    disturbance_scheduler = EventTermCfg(
        func=mdp.reset_disturbance_scheduler,
        mode="reset",
        params={
            "benchmark_path": "/home/wyh/WYH/isaac_nav_bridge/external_force/safety_benchmark_cases.json",
        },
    )

    # File-based Disturbance Trigger
    # Runs every step to check if the scheduled time has arrived
    disturbance_trigger = EventTermCfg(
        func=mdp.apply_disturbance_scheduler,
        mode="interval", 
        interval_range_s=(0.02, 0.02), # Check every step
        params={
            "benchmark_path": "/home/wyh/WYH/isaac_nav_bridge/external_force/safety_benchmark_cases.json",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
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
class NavG129Dex1WholebodyMatt3DEnvCfg(ManagerBasedRLEnvCfg):
    """
    inherits from ManagerBasedRLEnvCfg, defines all configuration parameters for the entire environment
    """

    # 1. scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, # environment number: 1
                                                     env_spacing=2.5, # environment spacing: 2.5 meter
                                                     replicate_physics=True # enable physics replication
                                                     )
    # basic settings
    observations: ObservationsCfg = ObservationsCfg()   # observation configuration
    actions: ActionsCfg = ActionsCfg()                  # action configuration
    commands: CommandsCfg = CommandsCfg()               # command configuration
    # MDP settings
        
    terminations: TerminationsCfg = TerminationsCfg()    # termination configuration
    events = EventCfg()                                  # event configuration
    # commands = None # command manager
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