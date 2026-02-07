
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os
import argparse
import contextlib
import pinocchio
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root
if project_root not in sys.path:
    sys.path.append(project_root)

# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--instruction", type=str, default="", help="Instruction text for VLM")
parser.add_argument("--action_source", type=str, default="dds_wholebody", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/Code/xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=100, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="assets/model/policy.onnx", help="model path")
parser.add_argument("--reward_interval", type=int, default=10, help="step interval for reward calculation")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")

parser.add_argument("--physics_dt", type=float, default=None, help="physics time step, e.g., 0.005")
parser.add_argument("--render_interval", type=int, default=None, help="render interval steps (>=1)")
parser.add_argument("--camera_write_interval", type=int, default=None, help="camera write interval steps (>=1)")


parser.add_argument(
    "--no_render",
    action="store_true",
    default=False,
    help="disable rendering updates entirely (overrides render interval)",
)
parser.add_argument("--solver_iterations", type=int, default=None, help="physx solver iteration count (e.g., 4)")
parser.add_argument("--gravity_z", type=float, default=None, help="override gravity z (e.g., -9.8)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False, help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--camera_jpeg", action="store_true", default=True, help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")

parser.add_argument("--physx_substeps", type=int, default=None, help="physx substeps per step")
parser.add_argument("--camera_include", type=str, default="front_camera,left_wrist_camera,right_wrist_camera", help="comma-separated camera names to enable")
parser.add_argument("--camera_exclude", type=str, default="world_camera", help="comma-separated camera names to disable")

parser.add_argument("--env_reward_interval", type=int, default=5, help="environment reward compute interval (steps)")
parser.add_argument("--seed", type=int, default=42, help="environment seed")
parser.add_argument("--max_episodes", type=int, default=1, help="Maximum number of episodes to run")
parser.add_argument("--isaac_scene_usd", type=str, default=None, help="ISAAC_SCENE_USD environment variable value")
parser.add_argument("--isaac_robot_init_pos", type=str, default=None, help="ISAAC_ROBOT_INIT_POS environment variable value")
parser.add_argument("--isaac_robot_init_rot", type=str, default=None, help="ISAAC_ROBOT_INIT_ROT environment variable value")
parser.add_argument("--episode_id", type=str, default="", help="Scene ID for logging")


# Navigation Task Arguments
parser.add_argument("--goal_pos", type=str, default=None, help="Goal position (x,y,z) for navigation metrics")
parser.add_argument("--success_radius", type=float, default=3.0, help="Success radius for navigation metrics")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")

# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Set environment variables for scene and robot configuration
if args_cli.isaac_scene_usd:
    os.environ["ISAAC_SCENE_USD"] = args_cli.isaac_scene_usd
if args_cli.isaac_robot_init_pos:
    os.environ["ISAAC_ROBOT_INIT_POS"] = args_cli.isaac_robot_init_pos
if args_cli.isaac_robot_init_rot:
    os.environ["ISAAC_ROBOT_INIT_ROT"] = args_cli.isaac_robot_init_rot

if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)



# [Fix] Register local extensions path
# This must be done BEFORE AppLauncher is initialized so Kit knows where to look for extensions
local_ext_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaaclab_exts")
if os.path.exists(local_ext_path):
    sys.argv.append(f"--ext-folder={local_ext_path}")
    print(f"[INFO] Added local extension path: {local_ext_path}")
else:
    print(f"[WARN] Local extension path not found: {local_ext_path}")

app_launcher = AppLauncher(args_cli, enable_extensions=["omni.kit.asset_converter", "omni.isaac.core"])
simulation_app = app_launcher.app

# [Fix] Delayed imports to prevent 'omni.kit_app' pre-loading warning
# These modules might import omni/pxr libraries indirectly, so we import them after SimulationApp starts

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from image_server.image_server import ImageServer
from dds.dds_create import create_dds_objects, create_dds_objects_replay

# [Fix] Monkey patch for Isaac Sim 5.0.0 ResolvedPath issue (ResolvedPath vs str type mismatch)
try:
    # Patch 1: omni.usd._impl.utils.is_usd_crate_file (Handles general file checks)
    import omni.usd._impl.utils
    _original_is_usd_crate_file = omni.usd._impl.utils.is_usd_crate_file
    def _patched_is_usd_crate_file(path):
        if not isinstance(path, str):
            path = str(path)
        return _original_is_usd_crate_file(path)
    omni.usd._impl.utils.is_usd_crate_file = _patched_is_usd_crate_file

    # Patch 2: Usd.CrateInfo.Open (Fixes ArgumentError in is_usd_crate_file_version_supported and others)
    # This is the specific error seen in logs: Usd.CrateInfo.Open(ResolvedPath) -> ArgumentError
    from pxr import Usd
    _original_crate_info_open = Usd.CrateInfo.Open
    def _patched_crate_info_open(path):
        if not isinstance(path, str):
            path = str(path)
        return _original_crate_info_open(path)
    Usd.CrateInfo.Open = _patched_crate_info_open
    
    print(f"[INFO] Monkey patched omni.usd utils and Usd.CrateInfo.Open for Isaac Sim 5.0.0. ISAAC_SCENE_USD={os.environ.get('ISAAC_SCENE_USD')}")
except Exception as e:
    print(f"[WARN] Failed to apply monkey patch: {e}")


from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
import numpy as np
from tools.measures import MeasureManager, PathLength, DistanceToGoal, Success, SPL

def setup_signal_handlers(controller,dds_manager=None):
    """set signal handlers"""
    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



def main():
    """main function"""
    # import cProfile
    # import pstats
    # import io
    # profiler = cProfile.Profile()
    # profiler.enable()
    import os
    import atexit
    try:
        os.setpgrp()
        current_pgid = os.getpgrp()
        print(f"Setting process group: {current_pgid}")
        
        def cleanup_process_group():
            try:
                print(f"Cleaning up process group: {current_pgid}")
                import signal
                os.killpg(current_pgid, signal.SIGTERM)
            except Exception as e:
                print(f"Failed to clean up process group: {e}")
        
        atexit.register(cleanup_process_group)
        
    except Exception as e:
        print(f"Failed to set process group: {e}")
    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return
    
    # create environment
    print("\ncreate environment...")
    try:
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        env.reset_count = 0 # Initialize reset counter shared with action provider
        try:
            sensors_dict = getattr(env.scene, "sensors", {})
            if sensors_dict:
                print("Sensors in the environment:")
                for name, sensor in sensors_dict.items():
                    print(name, sensor)
                print("="*60)
        except Exception as e:
            print(f"[sim] failed to list sensors: {e}")
        print(f"\ncreate environment success ...")
        try:
            env._reward_interval = max(1, int(args_cli.env_reward_interval))
            env._reward_counter = 0
            env._reward_last = None
            print(f"[env] reward compute interval set to {env._reward_interval} steps")
        except Exception as e:
            print(f"[env] failed to set reward interval: {e}")
        if args_cli.physics_dt is not None:
            try:
                env.sim.set_substep_time(args_cli.physics_dt)
                print(f"[sim] physics dt set to {args_cli.physics_dt}")
            except Exception:
                try:
                    env.sim.dt = args_cli.physics_dt
                    print(f"[sim] physics dt assigned to env.sim.dt={args_cli.physics_dt}")
                except Exception as e:
                    print(f"[sim] failed to set physics dt: {e}")
        headless_mode = bool(getattr(args_cli, "headless", False))
        render_interval = None
        if args_cli.render_interval is not None:
            try:
                render_interval = max(1, int(args_cli.render_interval))
            except Exception as e:
                print(f"[sim] invalid render_interval value {args_cli.render_interval}: {e}")
        try:
            if args_cli.no_render:
                env.sim.render_interval = 1_000_000
                env.sim.render_mode = "offscreen"
                print("[sim] rendering disabled via --no_render")
            elif headless_mode:
                env.sim.render_mode = "offscreen"
                env.sim.render_interval = render_interval or 1
                print(f"[sim] headless offscreen rendering every {env.sim.render_interval} steps")
            elif render_interval is not None:
                env.sim.render_interval = render_interval
                print(f"[sim] render_interval set to {env.sim.render_interval}")
        except Exception as e:
            print(f"[sim] failed to configure rendering: {e}")
        if args_cli.camera_write_interval is not None:
            try:
                import tasks.common_observations.camera_state as cam_state
                cam_state._camera_cache['write_interval_steps'] = max(1, int(args_cli.camera_write_interval))
                print(f"[camera] write interval steps set to {cam_state._camera_cache['write_interval_steps']}")
            except Exception as e:
                print(f"[camera] failed to set write interval: {e}")

        try:
            if args_cli.solver_iterations is not None:
                env.sim.physx.solver_iteration_count = int(args_cli.solver_iterations)
                print(f"[sim] solver_iteration_count={env.sim.physx.solver_iteration_count}")
            if args_cli.physx_substeps is not None:
                try:
                    env.sim.physx.substeps = int(args_cli.physx_substeps)
                except Exception:
                    try:
                        env.sim.set_substeps(int(args_cli.physx_substeps))
                    except Exception:
                        pass
                print(f"[sim] physx_substeps set to {args_cli.physx_substeps}")
            if args_cli.gravity_z is not None:
                g = float(args_cli.gravity_z)
                env.sim.physx.gravity = (0.0, 0.0, g)
                print(f"[sim] gravity set to {env.sim.physx.gravity}")
        except Exception as e:
            print(f"[sim] failed to set physx params: {e}")
        if args_cli.skip_cvtcolor:
            os.environ["CAMERA_SKIP_CVTCOLOR"] = "1"
        try:
            import tasks.common_observations.camera_state as cam_state
            enable_jpeg = bool(args_cli.camera_jpeg) or (os.getenv("CAMERA_JPEG") == "1")
            jpeg_quality = int(args_cli.camera_jpeg_quality if args_cli.camera_jpeg else os.getenv("CAMERA_JPEG_QUALITY", args_cli.camera_jpeg_quality))
            cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=args_cli.skip_cvtcolor)
            include = [n.strip() for n in (args_cli.camera_include or "").split(',') if n.strip()]
            exclude = [n.strip() for n in (args_cli.camera_exclude or "").split(',') if n.strip()]
            try:
                cam_state.set_camera_allowlist(include)
            except Exception:
                pass
            try:
                sensors_dict = getattr(env.scene, "sensors", {})
                for name, sensor in sensors_dict.items():
                    lname = name.lower()
                    if "camera" not in lname:
                        continue
                    if exclude and name in exclude:
                        for attr_name, value in [("enabled", False), ("is_enabled", False)]:
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, value)
                                except Exception:
                                    pass
                        for meth in ("set_active", "disable", "pause"):
                            if hasattr(sensor, meth):
                                try:
                                    getattr(sensor, meth)(False)
                                except Exception:
                                    pass
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
                    elif include and name not in include:
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
            except Exception as e:
                print(f"[camera] failed to tune sensors: {e}")
        except Exception as e:
            print(f"[camera] failed to apply writer options: {e}")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        return
    
    # get robot stiffness and damping parameters from runtime environment
    print("\n" + "="*60)
    print("🔍 Getting robot stiffness and damping parameters from runtime environment")
    print("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ Successfully got robot parameters!")
        else:
            print("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        print(f"⚠️ Error getting robot parameters: {e}")
    
    print("="*60)
    
    if not getattr(args_cli, "headless", False) and not args_cli.no_render:
        print("\n")
        print("***  Please left-click on the Sim window to activate rendering. ***")
        print("\n")
    else:
        print("\n")
        print("***  Running without GUI; rendering handled offscreen. ***")
        print("\n")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            # position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,
            focus_distance=1.2
        )
    env.sim.reset()
    env.reset()
    
    # [Fix] Headless/No-render mode collision issue: Warmup physics
    # In headless mode, complex meshes (like Matterport) sometimes fail to cook collision meshes 
    # if simulation starts immediately. Running a few steps warms up the physics engine.
    if getattr(args_cli, "headless", False) or args_cli.no_render:
        warmup_steps = 200 # NaVILA uses 200 steps for H1/G1 robots
        print(f"[INFO] Headless/No-render mode detected. Performing physics warmup ({warmup_steps} steps) to ensure collision meshes are loaded...")
        
        # Access robot to hold it in place
        robot = env.scene["robot"]
        # Use the default root state (defined in config) as the safe holding position
        safe_root_state = robot.data.default_root_state.clone()
        
        # Check if initial height is safe (approximate check)
        avg_z = safe_root_state[:, 2].mean().item()
        # print(f"[DEBUG] Robot Default Root State Z: {avg_z:.4f}")
        
        if avg_z < 0.5:
            print(f"[WARN] Robot initial Z height is approx {avg_z:.2f}m. Ensure ISAAC_ROBOT_INIT_POS is set high enough (e.g. >1.0m for H1) to prevent clipping.")

        for i in range(warmup_steps):
            # Force reset robot root state every step to prevent falling while collision meshes are cooking
            robot.write_root_state_to_sim(safe_root_state)
            env.sim.step()
            
        env.reset()
        print("[INFO] Physics warmup completed.")
        
        # Check position AFTER warmup and reset
        robot.update(dt=env.sim.get_physics_dt())
        curr_pos_after = robot.data.root_pos_w[0].cpu().numpy()
        print(f"[INFO] Robot Position AFTER Warmup & Reset: {curr_pos_after} (Z should be stable)")
    
    
    # create simplified control configuration
    try:    
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller
    
    # [Navigation Metrics] Initialization
    measure_manager = MeasureManager()
    
    # Parse goal_pos
    goal_pos = None
    if args_cli.goal_pos:
        try:
            goal_pos = [float(x) for x in args_cli.goal_pos.split(',')]
        except ValueError:
            print(f"[Warning] Invalid goal_pos format: {args_cli.goal_pos}")

    # Construct minimal episode dict for measures
    episode_info = {
        "goals": [{"radius": args_cli.success_radius}],
        "goal_pos": goal_pos,
        "start_pos": [0, 0, 0], # Placeholder, updated internally if needed or via CLI
        "gt_locations": [] # Can be populated if waypoints are available
    }

    # Register Measures
    if goal_pos:
        print(f"[Metrics] Initializing Navigation Metrics (Goal: {goal_pos}, Radius: {args_cli.success_radius})")
        measure_manager.register_measure(PathLength(env, episode_info, measure_manager))
        measure_manager.register_measure(DistanceToGoal(env, episode_info))
        measure_manager.register_measure(Success(env, episode_info, measure_manager))
        measure_manager.register_measure(SPL(env, episode_info, measure_manager))
        
        # Reset measures
        measure_manager.reset_measures()
    else:
        print("[Metrics] No goal_pos provided. Navigation metrics disabled.")

    # Stuck Detection Variables
    prev_pos = None
    same_pos_count = 0
    STUCK_THRESHOLD = 0.01 # 1cm
    STUCK_STEPS = 500 # 5.0s at 100Hz (Increased from 50 to avoid false positives)

    if not args_cli.replay_data:
        print("========= create image server =========")
        try:
            server = ImageServer(fps=30, Unit_Test=False)
        except Exception as e:
            print(f"Failed to create image server: {e}")
            return
        print("========= create image server success =========")
        print("========= create dds =========")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
            safety_dds = dds_manager.get_object("safety")
            
            # Setup instruction publisher
            instr_pub = ChannelPublisher("rt/instruction", String_)
            instr_pub.Init()
            instr_msg = String_(args_cli.instruction)

            # Publish instruction using InstructionDDS
            if args_cli.instruction:
                try:
                    instruction_dds = dds_manager.get_object("instruction")
                    if instruction_dds:
                        time.sleep(0.5) 
                        instruction_dds.publish_instruction(args_cli.instruction)
                except Exception as e:
                    print(f"[DDS] Failed to publish instruction: {e}")
            
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
    else:
        print("========= create dds =========")
        try:
            create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider
    
    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        print(f"args_cli.task: {args_cli.task}")
        if not args_cli.replay_data and ("Wholebody" in args_cli.task or args_cli.enable_wholebody_dds):
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True
            control_config.use_rl_action_mode = True
        action_provider = create_action_provider(env,args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            return
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    print("========= create controller success =========")
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data:
        setup_signal_handlers(controller,dds_manager)
    else:
        setup_signal_handlers(controller)
    print("Note: The DDS in Sim transmits messages on channel 7. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(7).")
    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        
        # [Fix] RTF Locking initialization
        # Try to get simulation dt from environment config
        try:
            sim_dt = env.unwrapped.cfg.sim.dt
            decimation = env.unwrapped.cfg.decimation
            step_dt = sim_dt * decimation
            print(f"[RTF Lock] Detected sim_dt={sim_dt}, decimation={decimation}, step_dt={step_dt}")
        except:
            # Fallback to step_hz
            step_dt = 1.0 / args_cli.step_hz
            print(f"[RTF Lock] Could not detect sim params, using step_hz={args_cli.step_hz} -> step_dt={step_dt}")

        rtf_start_time = time.time()

        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        
        
        reward_interval = max(1, args_cli.reward_interval)

        # use torch.inference_mode() and exception suppression
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                # Publish instruction periodically (or every step)
                if not args_cli.replay_data:
                    try:
                        instr_pub.Write(instr_msg)
                    except Exception:
                        pass
                
                # Check for max episodes exit condition
                # current_reset_count = getattr(env, "reset_count", 0)
                # if current_reset_count >= args_cli.max_episodes:
                #     print(f"\n[Main] Reached max episodes ({args_cli.max_episodes}), exiting simulation loop.")
                #     break

                current_time = time.time()
                loop_count += 1
                
                # [DEBUG] Fall Monitor
                if loop_count < 50:
                    robot_z = env.scene["robot"].data.root_pos_w[0, 2].item()
                    print(f"[DEBUG] Step {loop_count}: Robot Z = {robot_z:.4f}")
                    if robot_z < -1.0:
                         print("[FATAL] Robot has fallen below -1.0m! Collision mesh likely missing.")

                if not args_cli.replay_data:
                    try:
                        env_state = env.scene.get_state()
                        env_state_json =  sim_state_to_json(env_state)
                        # Get robot articulation to access joint torques
                        joint_torques = sim_state_to_json(env.scene["robot"].data.applied_torque)

                        # Call termination functions to get boolean flags
                        # Use standard Isaac Lab MDP functions to align with EnvCfg
                        from tasks.common_termination.base_termination_nav_wholebody import contact_force_termination, check_fall_risk_termination
                        from tasks.common_termination.base_termination_nav_wholebody import check_joint_limit_termination, check_timeout_termination
                        is_contact_force_exceeded = contact_force_termination(env)
                        is_fall_risk = check_fall_risk_termination(env)
                        
                        nav_task_latest = getattr(sim_state_dds, "nav_task_latest", {})
                        
                        # [Navigation Metrics] Update
                        # 1. Update Measures
                        if goal_pos:
                            measure_manager.update_measures()
                        
                        # 2. Stuck Detection
                        current_pos = env.scene["robot"].data.root_pos_w[0].detach().cpu().numpy()
                        if prev_pos is not None:
                            dist = np.linalg.norm(current_pos - prev_pos)
                            if dist < STUCK_THRESHOLD:
                                same_pos_count += 1
                            else:
                                same_pos_count = 0
                        prev_pos = current_pos
                        
                        is_stuck = same_pos_count >= STUCK_STEPS
                        
                        # 3. Stop Command Check (Logic: VLM outputs "Stop" instruction or velocity is zeroed by provider)
                        # Checking instruction string from DDS or ActionProvider state if available
                        # Assuming ActionProvider might have a way to signal stop, or we check instruction text
                        is_stop_called = False
                        if "Stop" in args_cli.instruction: # Simple check if initial instruction was stop (unlikely)
                             is_stop_called = True
                        
                        # Check if nav_task_latest has instruction update indicating stop
                        if isinstance(nav_task_latest, dict) and "instruction" in nav_task_latest:
                             if "stop" in nav_task_latest["instruction"].lower():
                                 is_stop_called = True

                        # Set stop flag for Success measure
                        setattr(env, "is_stop_called", is_stop_called)

                        # is_joint_limit = check_joint_limit_termination(env)
                        # is_timeout = check_timeout_termination(env)
                        
                        # Get Measurements
                        measurements = measure_manager.get_measurements() if goal_pos else {}

                        sim_state = {
                            "init_state": env_state_json,
                            "task_name": args_cli.task,
                            "episode_id": args_cli.episode_id,
                            "joint_torque": joint_torques,
                            "is_unsafe": safety_dds.is_unsafe(),
                            "force_exceeded": is_contact_force_exceeded.item(), # .item() converts tensor to Python bool
                            "fall_risk": is_fall_risk.item(),
                            "reset_count": getattr(env, "reset_count", 0),
                            "metrics": measurements,
                            "is_success": measurements.get("success", 0.0) > 0.5 if "success" in measurements else False
                        }
                        
                        # Force stop if stuck or max steps (handled by termination check usually, but enforcing here if needed)
                        if is_stuck:
                            print(f"[Info] Robot stuck for {STUCK_STEPS} steps. Forcing reset/done logic if applicable.")
                    except Exception as e:
                        print(f"Failed to get env state: {e}")
                        raise e
                    try:
                    # sim_state = json.dumps(sim_state)
                        sim_state_dds.write_sim_state_data(sim_state)
                    except Exception as e:
                        print(f"Failed to write sim state: {e}")
                        raise e
                    # print(f"reset_pose_dds: {reset_pose_dds}")
                    try:
                        reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    except Exception as e:
                        print(f"Failed to get reset pose command: {e}")
                        raise e
                    # # print(f"reset_pose_cmd: {reset_pose_cmd}")
                    # Compute current reward values manually if needed for debugging
                    
                    if reset_pose_cmd is not None:
                        try:
                            reset_category = reset_pose_cmd.get("reset_category")
                            # print(f"reset_category: {reset_category}")
                            if (args_cli.enable_wholebody_dds and (reset_category == '1' or reset_category == '2')) or (not args_cli.enable_wholebody_dds and reset_category == '1'):
                                print("reset object")
                                env_cfg.event_manager.trigger("reset_object_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                                if hasattr(env, "reset_count"): env.reset_count += 1
                            elif reset_category == '2' and not args_cli.enable_wholebody_dds:
                                print("reset all")
                                env_cfg.event_manager.trigger("reset_all_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                                if hasattr(env, "reset_count"): env.reset_count += 1
                        except Exception as e:
                            print(f"Failed to write reset pose command: {e}")
                            raise e
                else:
                    if action_provider.get_start_loop() and data_idx<len(data_json_list):
                        print(f"data_idx: {data_idx}")
                        try:
                            sim_state,task_name = action_provider.load_data(data_json_list[data_idx])
                            if task_name!=args_cli.task:
                                raise ValueError(f" The {task_name} in the dataset is different from the {args_cli.task} being executed .")
                        except Exception as e:
                            print(f"Failed to load data: {e}")
                            raise e
                        try:
                            env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                            env.sim.reset()
                            time.sleep(1)
                            action_provider.start_replay()
                            data_idx+=1
                        except Exception as e:
                            print(f"Failed to start replay: {e}")
                            raise e
                # print(f"env_state: {env_state}")
                # calculate instantaneous loop time
                loop_dt = current_time - last_loop_time
                last_loop_time = current_time
                recent_loop_times.append(loop_dt)
                
                # keep recent 100 loop times
                if len(recent_loop_times) > 100:
                    recent_loop_times.pop(0)
                
                # Check for max episodes exit condition AFTER data collection to ensure last frame is recorded
                if getattr(env, "reset_count", 0) >= args_cli.max_episodes:
                    print(f"\n[Main] Reached max episodes ({args_cli.max_episodes}), exiting simulation loop.")
                    break

                # [Fix] RTF Locking Execution
                # Ensure we don't run faster than real-time
                # Target time for this step (relative to start)
                expected_time = (loop_count - 1) * step_dt
                actual_time = time.time() - rtf_start_time
                
                if actual_time < expected_time:
                    sleep_time = expected_time - actual_time
                    if sleep_time > 0.001: # Avoid sleeping for negligible amounts
                        time.sleep(sleep_time)

                # execute control step (in main thread, support rendering)
                controller.step()

                # print statistics and loop frequency periodically
                if current_time - last_stats_time >= args_cli.stats_interval:
                    # calculate while loop execution frequency
                    elapsed_time = current_time - loop_start_time
                    loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # calculate moving average frequency (based on recent loop times)
                    if recent_loop_times:
                        avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                        moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                        min_loop_time = min(recent_loop_times)
                        max_loop_time = max(recent_loop_times)
                        max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                        min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                    else:
                        moving_avg_frequency = 0
                        min_freq = max_freq = 0
                    
                    print(f"\n=== While loop execution frequency statistics ===")
                    print(f"loop execution count: {loop_count}")
                    print(f"running time: {elapsed_time:.2f} seconds")
                    print(f"overall average frequency: {loop_frequency:.2f} Hz")
                    print(f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)")
                    print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                    print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                    if recent_loop_times:
                        print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                    print(f"=============================")
                    
                    # print_stats(controller)
                    last_stats_time = current_time
       
                # check environment state
                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break
                # rate_limiter.sleep(env)
    except KeyboardInterrupt:
        print("\nuser interrupted program")
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
    
    finally:
        # clean up resources
        print("\nclean up resources...")
        controller.cleanup()
        
        env.close()
        print("cleanup completed")
    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("time")
    # ps.print_stats(30)  

    # print(s.getvalue())

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Performing final cleanup...")
        
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")
            
        print("Program exit completed")
        
        # Force exit immediately
        import os
        import signal
        try:
            # Check if running on Linux/Unix to use killpg
            if os.name != 'nt':
                print("Force killing process group to release resources...")
                os.killpg(os.getpgrp(), signal.SIGKILL)
            else:
                print("Force exiting on Windows...")
                os._exit(0)
        except Exception as e:
            print(f"Failed to kill process group: {e}")
            os._exit(0)

# python sim_single_eposide_main.py --device cpu  --enable_cameras  --task Isaac-Nav-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds --enable_wholebody_dds