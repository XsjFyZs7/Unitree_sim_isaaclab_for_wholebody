from __future__ import annotations

import json
import torch
from typing import TYPE_CHECKING, Tuple, Optional, Dict, Any

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

import os
import datetime
LOG_FILE_PATH = None

from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states

def check_robot_reached_goal(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_position: Optional[Tuple[float, float, float]] = None,  # 目标点位置，可以为None
    distance_threshold: Optional[float] = None,  # 距离阈值，可以为None
    goal_position_attr: str = "goal_position",  # 环境属性中存储目标位置的属性名
    distance_threshold_attr: str = "distance_threshold",  # 环境属性中存储距离阈值的属性名
) -> torch.Tensor:
    """检查机器人是否到达目标点
    
    Args:
        env: ManagerBasedRLEnv - 强化学习环境实例
        robot_cfg: SceneEntityCfg - 机器人实体配置
        goal_position: Optional[Tuple[float, float, float]] - 目标点位置 (x, y, z)，如果为None则从环境属性中获取
        distance_threshold: Optional[float] - 距离阈值，如果为None则从环境属性中获取
        goal_position_attr: str - 环境属性中存储目标位置的属性名
        distance_threshold_attr: str - 环境属性中存储距离阈值的属性名
        
    Returns:
        torch.Tensor - 布尔张量，表示机器人是否到达目标点
    """
    # 获取机器人实体
    robot = env.scene[robot_cfg.name]
    
    # 获取机器人当前位置
    robot_position = robot.data.root_pos_w
    
    # 从环境属性中获取目标位置和距离阈值（如果未直接提供）
    if goal_position is None and hasattr(env, goal_position_attr):
        goal_position = getattr(env, goal_position_attr)
    elif goal_position is None:
        # 默认目标位置，如果既未提供也未在环境中设置
        goal_position = (0.0, 0.0, 0.0)
    
    if distance_threshold is None and hasattr(env, distance_threshold_attr):
        distance_threshold = getattr(env, distance_threshold_attr)
    elif distance_threshold is None:
        # 默认距离阈值，如果既未提供也未在环境中设置
        distance_threshold = 0.1
    
    # 计算机器人与目标点的距离
    goal_pos = torch.tensor(goal_position, device=env.device).expand(robot_position.shape[0], -1)
    distance = torch.norm(robot_position - goal_pos, dim=1)
    
    # 判断距离是否小于阈值
    reached = distance < distance_threshold
    
    return reached

def contact_force_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str = "contact_forces",
    force_threshold: float = 50.0,  # 接触力阈值，可根据实际情况调整
) -> torch.Tensor:
    """
    基于接触力的终止条件函数，当机器人与场景中物体的水平接触力超过阈值时触发任务终止。
    """
    
    global LOG_FILE_PATH
    
    # 1. 检查并获取传感器
    if contact_sensor_name not in env.scene.keys():
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_sensor = env.scene[contact_sensor_name]
    
    # 2. 获取接触力数据
    contact_forces = contact_sensor.data.net_forces_w
    if contact_forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # 3. 计算水平方向的力
    # 假设 Z 轴是垂直方向，我们只考虑 X 和 Y 方向的力
    horizontal_forces = contact_forces[..., :2]  # Shape: (num_envs, num_bodies, 2)
    horizontal_force_magnitudes = torch.norm(horizontal_forces, dim=-1)  # Shape: (num_envs, num_bodies)

    # 4. 找出每个环境中最大的水平接触力
    max_horizontal_force_per_env = torch.max(horizontal_force_magnitudes, dim=1)[0]  # Shape: (num_envs,)

    # 5. 判断是否终止
    terminate = max_horizontal_force_per_env > force_threshold    
    if torch.any(terminate):
        # 找出需要重置的环境的索引
        print("-------------------")
        print("max_horizontal_force_per_env:", max_horizontal_force_per_env[terminate])
        print("Out of Max Range!!!")

    return terminate

def get_robot_imu_data(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """get the robot IMU data
    
    Args:
        env: ManagerBasedRLEnv - reinforcement learning environment instance
    
    Returns:
        torch.Tensor
        - the first 3 elements are position
        - the next 4 elements are rotation quaternion
        - the next 3 elements are linear velocity
        - the last 3 elements are angular velocity
    """
    root_state = env.scene["robot"].data.root_state_w
    pos = root_state[:, :3]  # position
    quat = root_state[:, 3:7]  # rotation quaternion
    vel = root_state[:, 7:10]  # linear velocity
    ang_vel = root_state[:, 10:13]  # angular velocity
    imu_data = torch.cat([pos, quat, vel, ang_vel], dim=1)
    
    return imu_data

def quaternion_to_rpy(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to roll, pitch, yaw (radians).
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)
    pitch = torch.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(t3, t4)

    return torch.stack([roll, pitch, yaw], dim=1)

def check_fall_risk_termination(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str = "contact_forces",
    roll_threshold: float = 0.785,  # 横滚角阈值（弧度），约28.6度
    pitch_threshold: float = 0.785,  # 俯仰角阈值（弧度），约28.6度
) -> torch.Tensor:
    """
    基于IMU姿态数据和接触检测的跌倒风险判断函数。
    当机器人姿态角超过阈值或非腿部与地面接触时，判定为存在跌倒风险。
    
    Args:
        env: 环境实例
        contact_sensor_name: 接触传感器的名称，默认为"contact_forces"
        force_threshold: 非腿部接触力阈值
        roll_threshold: 横滚角阈值（弧度）
        pitch_threshold: 俯仰角阈值（弧度）
        
    Returns:
        torch.Tensor: 布尔张量，表示是否存在跌倒风险
    """
    # 初始化设备和环境数量
    device = env.device
    num_envs = env.num_envs
    
    imu_data = get_robot_imu_data(env)
    if imu_data.shape[0] == 0:
        # 如果没有IMU数据，返回False
        print("-----------------NO IMU------------------")
        attitude_risk = torch.zeros(num_envs, dtype=torch.bool, device=device)
    else:
        # 提取姿态四元数
        quat = imu_data[:, 3:7]  # 四元数部分
        
        # 转换为欧拉角（roll, pitch, yaw）
        rpy = quaternion_to_rpy(quat)
        roll, pitch = rpy[:, 0], rpy[:, 1]  # 提取横滚角和俯仰角
        
        # 判断姿态角是否超过阈值（取绝对值后比较）
        termination = (torch.abs(roll) > roll_threshold) | (torch.abs(pitch) > pitch_threshold)

        if torch.any(termination):
            # 找出需要重置的环境的索引
            print("-------------------")
            print("ROLL:", roll, "PITCH:", pitch)
            print("Out of Max Fall Range!!!")
        
    return termination

def check_joint_limit_termination(
    env: ManagerBasedRLEnv,
    position_threshold: float = 0.1,  # 关节位置接近限制的阈值（与限制的距离比例）
    torque_threshold_ratio: float = 0.9,  # 关节力矩阈值占最大力矩的比例
    duration_threshold: int = 10,  # 持续时间阈值（帧数）
    check_frequency: int = 5,  # 检查频率（每隔多少帧检查一次）
    export_limits_path: Optional[str] = "/home/wyh/WYH/data_process/config/joint_limit_config.json",
) -> torch.Tensor:
    """
    基于关节电机状态的终止条件函数，检测关节是否长时间接近限制位置。
    当关节位置接近上下限且力矩较大，并且这种状态持续一段时间，判定为电机可能损坏或卡死。

    Args:
        env: 环境实例
        position_threshold: 关节位置接近限制的阈值（与限制范围的比例，0.1表示距离限制的10%范围内）
        torque_threshold_ratio: 关节力矩阈值占最大力矩的比例，例如0.9表示90%的最大力矩
        duration_threshold: 持续时间阈值（帧数），状态持续超过此值触发终止
        check_frequency: 检查频率（每隔多少帧检查一次），避免每帧都检查增加计算负担
        critical_joints: 重点监测的关节列表，如果为None则监测所有关节

    Returns:
        torch.Tensor: 布尔张量，表示是否应该终止任务
    """
    
    # 初始化设备和环境数量
    device = env.device
    num_envs = env.num_envs

    # 使用静态变量存储状态持续时间计数器和上一次检查的时间步
    if not hasattr(check_joint_limit_termination, "_counter"):
        check_joint_limit_termination._counter = torch.zeros(num_envs, dtype=torch.int, device=device)
        check_joint_limit_termination._last_step = -1

    # 获取当前时间步
    current_step = env.common_step_counter
    
    # 在每个 episode 开始时重置计数器
    if current_step == 0:
        check_joint_limit_termination._counter.zero_()

    # 如果不是检查频率的倍数，直接返回False
    if current_step % check_frequency != 0 or current_step == check_joint_limit_termination._last_step:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 更新上一次检查的时间步
    check_joint_limit_termination._last_step = current_step

    # 获取关节状态数据
    joint_data = get_robot_boy_joint_states(env, enable_dds=False)
    if joint_data.shape[0] == 0:
        # 如果没有关节数据，返回False
        return torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 获取关节数量
    n_joints = joint_data.shape[1] // 3

    # 提取关节位置、速度和力矩
    joint_pos = joint_data[:, :n_joints]  # 关节位置
    joint_torque = joint_data[:, 2*n_joints:3*n_joints]  # 关节力矩

    # 获取关节名称、位置限制和力矩限制
    robot = env.scene["robot"]
    joint_names = robot.data.joint_names
    joint_pos_limits = robot.data.joint_pos_limits
    joint_effort_limits = robot.data.joint_effort_limits
    joint_velocity_limits = getattr(robot.data, "joint_velocity_limits", None)
    if not hasattr(check_joint_limit_termination, "_joint_limits"):
        names = list(joint_names)
        pos = joint_pos_limits[0].cpu().numpy().tolist()
        eff = joint_effort_limits[0].cpu().numpy().tolist() if joint_effort_limits is not None else None
        vel = joint_velocity_limits[0].cpu().numpy().tolist() if joint_velocity_limits is not None else None
        d = {}
        for i, n in enumerate(names):
            qmin, qmax = pos[i]
            vmax = vel[i] if vel is not None else None
            tmax = eff[i] if eff is not None else None
            d[n] = {"qmin": qmin, "qmax": qmax, "vmax": vmax, "tmax": tmax}
        check_joint_limit_termination._joint_limits = d
        if export_limits_path:
            try:
                os.makedirs(os.path.dirname(export_limits_path), exist_ok=True)
                with open(export_limits_path, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
    
    # 如果指定了重点监测的关节，则只监测这些关节
    critical_indices = list(range(len(joint_names)))

    # 初始化关节状态异常标志
    joint_at_limit = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # 检查每个关节是否接近限制位置且力矩较大
    for idx in critical_indices:
        if idx >= joint_pos.shape[1]:
            continue

        joint_name = joint_names[idx]
        joint_pos_val = joint_pos[:, idx].cpu().numpy()
        joint_torque_val = joint_torque[:, idx].cpu().numpy()

        lower_limit = joint_pos_limits[:, idx, 0].cpu().numpy()
        upper_limit = joint_pos_limits[:, idx, 1].cpu().numpy()
        joint_torque_threshold = joint_effort_limits[0, idx] * torque_threshold_ratio
        
        # 获取关节位置限制
        lower_limit = joint_pos_limits[:, idx, 0]
        upper_limit = joint_pos_limits[:, idx, 1]
        limit_range = upper_limit - lower_limit

        # 判断关节是否接近限制位置（距离小于阈值）
        near_lower_limit = (joint_pos[:, idx] - lower_limit) < limit_range * position_threshold
        near_upper_limit = (upper_limit - joint_pos[:, idx]) < limit_range * position_threshold

        # 动态计算当前关节的力矩阈值
        joint_torque_threshold = joint_effort_limits[0, idx] * torque_threshold_ratio

        # 判断力矩是否较大（绝对值大于动态阈值）
        high_torque = torch.abs(joint_torque[:, idx]) > joint_torque_threshold

        # 判断关节是否处于异常状态（接近限制位置且力矩较大）
        joint_abnormal = (near_lower_limit | near_upper_limit) & high_torque

        # 更新关节状态异常标志
        joint_at_limit = joint_at_limit | joint_abnormal
        
        # 如果关节状态异常，则打印信息


    # 更新状态持续时间计数器
    check_joint_limit_termination._counter = torch.where(
        joint_at_limit,
        check_joint_limit_termination._counter + check_frequency,  # 如果异常，增加计数器
        torch.zeros_like(check_joint_limit_termination._counter)   # 如果正常，重置计数器
    )

    # 判断是否超过持续时间阈值
    terminate = check_joint_limit_termination._counter >= duration_threshold

    # 如果有任何环境触发终止，则打印调试信息
    # if torch.any(terminate):
    #     terminating_envs = torch.where(terminate)[0]
        

    return terminate

def check_timeout_termination(
    env: ManagerBasedRLEnv,
    time_threshold: float = 40.0,  # Timeout in seconds
) -> torch.Tensor:
    """
    Terminates the environment if it runs for longer than the specified time threshold.
    
    This function relies on `env.episode_length_buf` which tracks the number of steps
    for each environment in the current episode.
    
    Args:
        env: The reinforcement learning environment instance.
        time_threshold: The maximum allowed time for an episode in seconds.
        
    Returns:
        A boolean tensor indicating which environments should be terminated due to timeout.
    """
    # Calculate elapsed time for each environment
    elapsed_time = env.episode_length_buf * env.physics_dt
    # Check for timeout
    timeout_termination = elapsed_time > time_threshold
    
    if torch.any(timeout_termination):
        print("-------------------")
        print(f"Timeout reached for envs. Threshold: {time_threshold}s")
        print("-------------------")
        
    return timeout_termination