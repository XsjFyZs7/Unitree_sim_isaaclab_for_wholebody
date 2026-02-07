# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0      
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import os
import sys

# Ensure project root is in sys.path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up from tasks/common_scene to unitree_sim_isaaclab
project_root_path = os.path.dirname(os.path.dirname(current_dir))
if project_root_path not in sys.path:
    sys.path.append(project_root_path)

# Ensure PROJECT_ROOT env var is set
if "PROJECT_ROOT" not in os.environ:
    os.environ["PROJECT_ROOT"] = project_root_path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from tasks.common_config import CameraBaseCfg  # isort: skip
import os

# [新增] 导入 Matterport 配置
# 作为标准 Python 包导入 (确保 matterport_sim 在 Python 路径中)
try:
    from matterport_sim.config.importer_cfg import MatterportImporterCfg
    print("[Config DEBUG] Import MatterportImporterCfg successfully.")
except Exception as e:
    import traceback
    print(f"\n[ERROR] Failed to import MatterportImporterCfg from matterport_sim.")
    print(f"[ERROR] Error details: {e}")
    print(f"[ERROR] Detailed traceback:")
    traceback.print_exc()
    print("-" * 60)
    print("[WARNING] Fallback to default/None for MatterportImporterCfg.\n")
    MatterportImporterCfg = None

project_root = os.environ.get("PROJECT_ROOT")

@configclass
class TableCylinderSceneCfgWH(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
    
    # [修改] 使用 terrain 替代原本的 room_walls
    # NaVILA 使用 MatterportImporter 作为地形加载器
    if MatterportImporterCfg:
        terrain = MatterportImporterCfg(
            prim_path="/World/matterport",
            terrain_type="matterport",
            # 这里会自动处理 .obj 到 .usd 的转换，如果已经是 .usd 则直接加载
            obj_filepath=os.environ.get("ISAAC_SCENE_USD", f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd"),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            debug_vis=False,
            groundplane=False, # 设为 False，避免与场景自带地面冲突 (除非场景没地面)
        )
        # 移除原本的 room_walls，避免重复加载
        room_walls = None
    else:
        # Fallback if extension missing (保留旧逻辑作为备份)
        room_walls = AssetBaseCfg(
            prim_path="/World/Scene",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=[0.0, 0.0, 0],
                rot=[1.0, 0.0, 0.0, 0.0]
            ),
            spawn=UsdFileCfg(
                usd_path=os.environ.get("ISAAC_SCENE_USD", f"{project_root}/assets/objects/small_warehouse/small_warehouse_digital_twin.usd"),
                scale=(1.0, 1.0, 1.0),
            ),
        )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )
    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
                                                    pos_offset=(-1.9, -5.0, 1.8),
                                                    rot_offset=( -0.40614,0.78544, 0.4277, -0.16986))