import os
import sys
import gzip
import json
import argparse
import subprocess
import time
from pathlib import Path

def read_episodes(dataset_path):
    """
    读取 gzip 压缩的 JSON 数据集文件。
    与 NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/omni/isaac/vlnce/utils/eval_utils.py 中的 read_episodes 保持一致。
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集文件未找到: {dataset_path}")

    print(f"正在加载数据集: {dataset_path} ...")
    with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    
    # 根据 NaVILA-Bench 的实现，直接返回 data["episodes"]
    print(f"成功加载 {len(data['episodes'])} 个 Episode。")
    return data["episodes"]

def format_list_arg(values):
    """将数字列表格式化为逗号分隔的字符串 (例如: '1.0,2.0,3.0')"""
    if values is None:
        return ""
    return ",".join(map(str, values))

def main():
    parser = argparse.ArgumentParser(description="Unitree Simulation 自动化调度脚本")
    
    # 核心参数
    parser.add_argument("--dataset", type=str, default="/home/wyh/WYH/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets/vln_ce_isaac_v1.json.gz", help="数据集路径 (vln_ce_isaac_v1.json.gz)")
    parser.add_argument("--sim_script", type=str, default="sim_single_eposide_main.py", help="单次仿真执行脚本路径")
    parser.add_argument("--assets_dir", type=str, default="/home/wyh/WYH/NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets", help="NaVILA-Bench assets 根目录")
    
    # 传递给 sim_single_eposide_main.py 的固定参数
    parser.add_argument("--task", type=str, default="Isaac-Nav-G129-Dex1-Wholebody-Matt3D", help="任务名称")
    parser.add_argument("--robot_type", type=str, default="g129", help="机器人类型")
    
    # 调度控制
    parser.add_argument("--start_idx", type=int, default=0, help="开始运行的 Episode 索引")
    parser.add_argument("--end_idx", type=int, default=1, help="结束运行的 Episode 索引 (默认运行到最后)")
    parser.add_argument("--dry_run", action="store_true", help="仅打印命令而不执行")
    # 添加 device 参数
    parser.add_argument("--device", type=str, default="cpu", help="仿真设备 (cpu/cuda)")
    
    args = parser.parse_args()

    # 1. 验证仿真脚本是否存在
    sim_script_path = Path(args.sim_script).resolve()
    if not sim_script_path.exists():
        print(f"[Error] 仿真脚本未找到: {sim_script_path}")
        sys.exit(1)

    # 2. 读取数据集
    try:
        episodes = read_episodes(args.dataset)
    except Exception as e:
        print(f"[Error] 读取数据集失败: {e}")
        # 如果找不到文件，尝试在上一级目录查找 (NaVILA-Bench 常用结构)
        # 依次尝试: 
        # 1. ../NaVILA-Bench/data
        # 2. ../NaVILA-Bench/assets
        # 3. ../NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets (ASSETS_DIR origin)
        search_paths = [
            Path("../NaVILA-Bench/data") / args.dataset,
            Path("../NaVILA-Bench/assets") / args.dataset,
            Path("../NaVILA-Bench/isaaclab_exts/omni.isaac.vlnce/assets") / args.dataset
        ]
        
        found = False
        for alt_path in search_paths:
            if alt_path.exists():
                print(f"尝试从备用路径加载: {alt_path}")
                episodes = read_episodes(alt_path)
                found = True
                break
        
        if not found:
            sys.exit(1)

    # 3. 切片需要运行的 Episodes
    end_idx = args.end_idx if args.end_idx is not None else len(episodes)
    episodes_to_run = episodes[args.start_idx : end_idx]
    
    print(f"\n准备运行 Episodes {args.start_idx} 到 {end_idx} (共 {len(episodes_to_run)} 个)")
    print("="*60)

    # 4. 循环执行
    for i, episode in enumerate(episodes_to_run):
        global_idx = args.start_idx + i
        episode_id = episode.get('episode_id', 'N/A')
        print(f"\n[调度器] 正在处理 Episode {global_idx}/{len(episodes)} (ID: {episode_id})")
        
        # 提取关键参数
        scene_id = episode.get("scene_id")
        start_position = episode.get("start_position")
        start_rotation = episode.get("start_rotation")
        
        # 简单校验
        if not scene_id or start_position is None or start_rotation is None:
            print(f"[跳过] Episode {global_idx}: 缺少必要字段 (scene_id, start_position, start_rotation)")
            continue

        print(f"[DEBUG] 原始 Scene ID: {scene_id}")
        print(f"[DEBUG] 原始 Start Position: {start_position}")
        print(f"[DEBUG] 原始 Start Rotation: {start_rotation}")

        # [Fix] Apply Z-offset for robot spawning to prevent falling/collision
        # Logic adapted from navila_eval.py and G1 default config
        # NaVILA adds +1.0 for H1, +0.4 for Go2. G1 is humanoid, similar to H1, defaulting to +0.8 in configs.
        try:
            adjusted_start_position = list(start_position)
            if "h1" in args.task.lower():
                adjusted_start_position[2] += 1.0
            elif "go2" in args.task.lower():
                adjusted_start_position[2] += 0.4
            elif "g1" in args.task.lower():
                adjusted_start_position[2] += 0.8
            else:
                adjusted_start_position[2] += 0.5
            print(f"[DEBUG] Adjusted Start Position (Z-offset applied): {adjusted_start_position}")
        except Exception as e:
            print(f"[WARN] Failed to apply Z-offset: {e}, using original position.")
            adjusted_start_position = start_position

        # 处理场景路径
        # 逻辑对齐 navila_eval.py: 提取场景ID并构造 .usd 路径

        # episode["scene_id"] 可能是 "mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb"
        if args.assets_dir:
            filename = os.path.basename(scene_id)
            scene_base_id = os.path.splitext(filename)[0]
            scene_usd = os.path.join(args.assets_dir, "matterport_usd", scene_base_id, f"{scene_base_id}.usd")
            scene_usd = os.path.abspath(scene_usd)
        else:
            scene_usd = scene_id
        # 运行前校验与规范化，避免空场景
        scene_usd = os.fspath(os.path.normpath(scene_usd))
        print(f"[Scene] USD path resolved: {scene_usd}")
        
        if os.path.isfile(scene_usd):
            file_size = os.path.getsize(scene_usd)
            print(f"[DEBUG] 确认文件存在: {scene_usd} (大小: {file_size/1024/1024:.2f} MB)")
        else:
            print(f"[DEBUG] 文件不存在! 路径: {scene_usd}")
            print(f"[跳过] Episode {global_idx}: USD 文件不存在: {scene_usd}")
            continue

        # 构造命令行参数
        
        # Check for local extension folder
        # sim_script_path is .../sim_single_eposide_main.py
        # local extension should be in .../omni.isaac.matterport
        local_ext_dir = os.path.dirname(sim_script_path)
        local_matterport = os.path.join(local_ext_dir, "omni.isaac.matterport")
        
        extra_args = []

        # 更新命令构造
        cmd = [
            sys.executable, str(sim_script_path),
            "--task", args.task,
            "--robot_type", args.robot_type,
            "--device", args.device, # 用户指定的设备
            "--max_episodes", "1",  # 强制只运行一个 episode 后退出
            "--isaac_scene_usd", scene_usd,
            "--isaac_robot_init_pos", format_list_arg(adjusted_start_position),
            "--isaac_robot_init_rot", format_list_arg(start_rotation),
            "--enable_dex1_dds", # 根据需要开启
            "--enable_wholebody_dds",
            "--enable_cameras",  # 必须启用相机
            # "--headless"   # 无头模式 (保留，否则在服务器上无法运行)
        ] + extra_args
        
        print(f"[执行命令] {' '.join(cmd)}")
        
        if not args.dry_run:
            start_time = time.time()
            try:
                # 调用子进程执行仿真
                # check=True 会在退出码非0时抛出 CalledProcessError
                subprocess.run(cmd, check=True)
                print(f"[成功] Episode {global_idx} 完成，耗时 {time.time() - start_time:.2f}s")
            except subprocess.CalledProcessError as e:
                print(f"[失败] Episode {global_idx} 异常退出 (Exit Code: {e.returncode})")
                # 可以选择 continue 继续下一个，或者 break 停止
                # continue 
            except KeyboardInterrupt:
                print("\n[中断] 用户停止了脚本。")
                break
            except Exception as e:
                print(f"[错误] 未知错误: {e}")

if __name__ == "__main__":
    main()