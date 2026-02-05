import torch
import yaml
import json
import os
import numpy as np
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# --- 内部状态管理 ---
_ENV_DISTURBANCE_STATES = {}

class DisturbanceState:
    def __init__(self, num_envs, device):
        self.trigger_times = torch.zeros(num_envs, device=device)
        self.target_velocities = torch.zeros((num_envs, 3), device=device)
        self.has_triggered = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        # Benchmark 回放状态
        self.benchmark_cases = []
        self.global_case_counter = 0 # 记录当前分配到了第几个case
        self.cases_loaded = False
        self.active_case_ids = [-1] * num_envs # 记录每个环境当前正在跑哪个case id

def _get_state(env: "ManagerBasedRLEnv") -> DisturbanceState:
    env_id = id(env)
    if env_id not in _ENV_DISTURBANCE_STATES:
        _ENV_DISTURBANCE_STATES[env_id] = DisturbanceState(env.num_envs, env.device)
    return _ENV_DISTURBANCE_STATES[env_id]

def _load_benchmark_cases(file_path: str, state: DisturbanceState):
    """加载 Benchmark JSON 文件"""
    if state.cases_loaded:
        return

    if not os.path.exists(file_path):
        print(f"[Warning] Benchmark file not found: {file_path}")
        return

    try:
        with open(file_path, 'r') as f:
            state.benchmark_cases = json.load(f)
        state.cases_loaded = True
        print(f"[Benchmark] Loaded {len(state.benchmark_cases)} test cases from {file_path}")
    except Exception as e:
        print(f"[Error] Failed to load benchmark cases: {e}")

# --- 事件函数 1: Reset 时调度 (Benchmark 模式) ---
def reset_disturbance_scheduler(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    benchmark_path: str 
):
    """
    在环境重置时调用。
    从 benchmark 文件中按顺序分配测试用例。
    """
    state = _get_state(env)
    
    # 首次加载
    _load_benchmark_cases(benchmark_path, state)
    
    if not state.cases_loaded or len(state.benchmark_cases) == 0:
        return

    # 1. 重置触发标志
    state.has_triggered[env_ids] = False
    
    # 2. 强制清零机器人速度（根节点和关节），防止残留动量导致 Reset 后不稳定
    try:
        robot: Articulation = env.scene["robot"]
        # Reset root velocity (linear + angular) to zero
        root_vel = torch.zeros((len(env_ids), 6), device=env.device)
        robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        # Reset joint velocity to zero
        joint_vel = torch.zeros((len(env_ids), robot.num_joints), device=env.device)
        robot.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)
    except Exception as e:
        print(f"[Warning] Failed to reset robot velocity in disturbance scheduler: {e}")

    # 3. 为每个被重置的环境分配下一个 Case
    # 注意：env_ids 可能包含多个索引
    for env_idx in env_ids:
        # 获取当前应分配的 case index (循环使用)
        case_idx = state.global_case_counter % len(state.benchmark_cases)
        case_data = state.benchmark_cases[case_idx]
        
        # 记录分配
        state.global_case_counter += 1
        state.active_case_ids[env_idx] = case_data['id']
        
        # 设置参数
        state.trigger_times[env_idx] = case_data['trigger_time']
        
        vel = torch.tensor(case_data['velocity'], dtype=torch.float, device=env.device)
        state.target_velocities[env_idx] = vel
        
        print(f"[Env {env_idx.item()}] Assigned Case {case_data['id']}: Time={case_data['trigger_time']}s, Vel={case_data['velocity']}")

# --- 事件函数 2: Step 时执行 (保持不变) ---
def apply_disturbance_scheduler(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    benchmark_path: str # 参数名需与 reset 保持一致或被忽略，这里仅用于占位
):
    """
    在每一步调用。逻辑与之前完全一致：检查时间并触发。
    """
    # Unconditional debug print to verify function call
    if env.common_step_counter % 100 == 0:
         print(f"[Disturbance Debug] apply_disturbance_scheduler called. Step: {env.common_step_counter}, Num env_ids: {len(env_ids) if env_ids is not None else 'None'}")

    state = _get_state(env)
    
    dt = env.step_dt
    current_times = env.episode_length_buf[env_ids] * dt
    
    trigger_times = state.trigger_times[env_ids]
    already_triggered = state.has_triggered[env_ids]
    
    # 判定触发
    should_trigger = (current_times >= trigger_times) & (~already_triggered) & (trigger_times > 0.01)
    
    trigger_indices = env_ids[should_trigger]
    
    if len(trigger_indices) > 0:
        state.has_triggered[trigger_indices] = True
        
        asset: Articulation = env.scene[asset_cfg.name]
        vel_deltas = state.target_velocities[trigger_indices]
        root_vel = asset.data.root_vel_w[trigger_indices].clone()
        root_vel[:, :3] += vel_deltas
        
        asset.write_root_velocity_to_sim(root_vel, env_ids=trigger_indices)
        
        # 打印日志以便后续分析 (格式: [Benchmark] CaseID Result)
        for idx in trigger_indices:
           case_id = state.active_case_ids[idx]
           print(f"[Benchmark Trigger] Env {idx.item()} triggered Case {case_id} at time {current_times[idx]:.3f}s with vel {vel_deltas[idx].tolist()}")

    # Debug print for first env if waiting
    # Print roughly every 100 steps (approx 0.5s with dt=0.005)
    if 0 in env_ids:
        idx = 0
        if not state.has_triggered[idx] and state.trigger_times[idx] > 0.01:
            if env.episode_length_buf[idx] % 100 == 0:
                print(f"[Disturbance Debug] Env {idx} waiting. Current: {current_times[idx]:.3f}s, Target: {state.trigger_times[idx]:.3f}s")
