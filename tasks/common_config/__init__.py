"""
公共配置模块
提供可复用的机器人和相机配置
"""

from .robot_configs import RobotBaseCfg, H12RobotPresets, RobotJointTemplates,G1RobotPresets, RobotEnvCfg, G1RobotEnvPresets
from .camera_configs import CameraBaseCfg, CameraPresets, CameraEnvCfg, CameraEnvPresets

__all__ = [
    "RobotBaseCfg",
    "G1RobotPresets",
    "H12RobotPresets",
    "RobotJointTemplates", 
    "CameraBaseCfg",
    "CameraPresets",
    "RobotEnvCfg",
    "G1RobotEnvPresets",
    "CameraEnvCfg",
    "CameraEnvPresets"
] 