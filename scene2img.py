"""
Isaac Sim 场景到 Nav2 导航图生成器

这个脚本加载一个 Isaac Sim 场景，使用 Occupancy Map Generator 生成一个 2D 占用栅格地图，
并将其保存为 Nav2 兼容的 .pgm 和 .yaml 文件，然后立即退出。
"""

import os
import sys
import argparse
import signal
import cv2
import numpy as np
import yaml

# 设置项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

# 命令行参数
parser = argparse.ArgumentParser(description="Isaac Sim 场景到 Nav2 导航图生成器")
parser.add_argument("--task", type=str, default="Isaac-Nav-G129-Dex1-Wholebody", help="要加载的 Isaac Lab 任务/场景名称")
parser.add_argument("--output_dir", type=str, default="./nav_maps_2", help="地图文件保存目录")
parser.add_argument("--map_resolution", type=float, default=0.05, help="地图分辨率 (米/像素)")
parser.add_argument("--map_size_x", type=float, default=20.0, help="地图 X 轴尺寸 (米)")
parser.add_argument("--map_size_y", type=float, default=25.0, help="地图 Y 轴尺寸 (米)")
parser.add_argument("--map_height_min", type=float, default=0.1, help="扫描障碍物的最小高度 (米)")
parser.add_argument("--map_height_max", type=float, default=0.8, help="扫描障碍物的最大高度 (米)")

parser.add_argument("--physics_dt", type=float, default=1.0/100.0, help="物理时间步长")
parser.add_argument("--render_interval", type=int, default=1, help="渲染间隔步数")
parser.add_argument("--solver_iterations", type=int, default=4, help="PhysX求解器迭代次数")
parser.add_argument("--physx_substeps", type=int, default=4, help="每个step的PhysX子步数")
parser.add_argument("--gravity_z", type=float, default=-9.81, help="Z轴重力加速度")
parser.add_argument("--seed", type=int, default=42, help="环境种子")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 创建 AppLauncher
import pinocchio 
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 导入 Isaac Sim 相关模块
from tasks.utils.parse_cfg import parse_env_cfg
import gymnasium as gym
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.gen.omap")

# 在这里导入 Occupancy Map 生成器
try:
    import omni
    from isaacsim.asset.gen.omap.bindings import _omap as omap
    print("-------- !!! Load OMNI and OMAP Successful !!! --------")
except ImportError as e:
    print(f"错误：无法导入 Occupancy Map Generator 模块: {e}")
    print("无法继续执行程序，正在退出...")
    sys.exit(1)  # 使用非零退出码表示错误

# 信号处理
def setup_signal_handlers():
    """设置信号处理器"""
    def signal_handler(signum, frame):
        print(f"\n接收到信号 {signum}，正在退出...")
        try:
            simulation_app.close()
        except Exception as e:
            print(f"关闭模拟应用程序失败: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

# 生成并保存占用地图
def generate_and_save_occupancy_map(args):
    """使用 Occupancy Map Generator 生成并保存地图"""
    print("\n正在生成占用栅格地图...")
    
    import omni.physx
    import omni.usd
    
    physx = omni.physx.acquire_physx_interface()
    stage_id = omni.usd.get_context().get_stage_id()
    
    # 创建输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    map_base_name = "nav_map_2"
    map_image_path = os.path.join(output_dir, f"{map_base_name}.pgm")
    map_yaml_path = os.path.join(output_dir, f"{map_base_name}.yaml")

    # 初始化 Occupancy Map 生成器
    generator = omap.Generator(physx, stage_id)

    # 设置生成器参数 - 使用 update_settings 替代 set_parameters
    # 参数顺序: cell_size, occupied_value, unoccupied_value, unknown_value
    generator.update_settings(
        args.map_resolution,  # cell_size (分辨率)
        100,  # occupied_value (占用值)
        0,    # unoccupied_value (空闲值)
        -1    # unknown_value (未知值)
    )
    
    # 设置地图范围
    # 计算地图边界
    half_size_x = args.map_size_x / 2.0
    half_size_y = args.map_size_y / 2.0
    
    # 设置原点和边界，使其与世界坐标系对齐，不再交换X和Y
    generator.set_transform(
        (0, 0, 0),  # 原点在世界中心
        (-half_size_x, -half_size_y, args.map_height_min),  # 最小边界
        (half_size_x, half_size_y, args.map_height_max)     # 最大边界
    )

    # 生成 2D 地图
    print("正在调用 generate2d()... 这可能需要一些时间。")
    generator.generate2d()
    
    # 获取生成的地图数据
    buffer = generator.get_buffer()
    dims = generator.get_dimensions()
    size_x = dims[0]
    size_y = dims[1]
    print(f"地图生成完毕，尺寸: {size_x}x{size_y} 像素")

    if not buffer or size_x == 0 or size_y == 0:
        print("错误：地图生成失败，返回的缓冲区为空或尺寸为0。")
        return False

    # 将返回的缓冲区 (0=free, 100=occupied, -1=unknown) 转换为 PGM 格式
    # PGM: 0=black (occupied), 255=white (free), 205=gray (unknown)
    map_image = np.full((size_y, size_x), 205, dtype=np.uint8) # 默认为未知区域
    buffer_array = np.array(buffer).reshape((size_y, size_x))

    map_image[buffer_array == 0] = 254    # 空闲区域 (free_thresh 范围内)
    map_image[buffer_array == 100] = 0    # 占用区域 (occupied_thresh 范围内)

    # 根据ROS坐标系要求，将图像旋转180度
    # 这对应于官方文档UI中的“Rotate Image: 180 degrees”选项
    map_image = cv2.flip(map_image, -1)

    # 保存 PGM 图像
    cv2.imwrite(map_image_path, map_image)
    print(f"已保存地图图像: {map_image_path}")

    # 创建并保存 YAML 文件
    # Nav2 的原点是地图左下角在世界坐标系中的位置
    origin_x = -args.map_size_x / 2.0
    origin_y = -args.map_size_y / 2.0
    
    yaml_data = {
        "image": f"{map_base_name}.pgm",
        "resolution": args.map_resolution,
        "origin": [origin_x, origin_y, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25 # Isaac Sim 默认 free_thresh 为 0.25
    }

    with open(map_yaml_path, 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file, default_flow_style=False)
    print(f"已保存地图配置文件: {map_yaml_path}")

    return True

# 主函数
def main():
    """主函数"""
    print("=" * 60)
    print("Isaac Sim 场景到 Nav2 导航图生成器已启动")
    print(f"场景: {args.task}")
    print("=" * 60)
    
    # 设置信号处理器
    setup_signal_handlers()
    
    # 解析环境配置
    try:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    except Exception as e:
        print(f"解析环境配置失败: {e}")
        return
    
    # 创建环境
    print("\n正在创建环境...")
    try:
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        print("环境创建成功")
    except Exception as e:
        print(f"创建环境失败: {e}")
        return
    
    # 重置环境以确保所有物体都已加载
    # print("执行 env.sim.reset()...")
    # env.sim.reset()
    # print("执行 env.reset()...")
    # env.reset()
    
    # 生成并保存占用地图
    success = generate_and_save_occupancy_map(args)
    
    # 关闭环境
    print("\n正在关闭环境...")
    env.close()
    print("环境已关闭")
    
    # 退出模拟
    print("\n正在退出 Isaac Sim...")
    simulation_app.close()
    print("Isaac Sim 已退出")
    
    if success:
        print("\n导航图生成成功，程序正常退出")
    else:
        print("\n导航图生成失败，程序退出")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序执行出错: {e}")
    finally:
        print("正在执行最终清理...")
        try:
            simulation_app.close()
        except Exception as e:
            print(f"关闭模拟应用程序失败: {e}")
        
        # 强制退出
        os._exit(0)