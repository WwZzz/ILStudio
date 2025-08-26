import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
from abc import ABC, abstractmethod
from pynput import keyboard
import importlib
import argparse
import yaml
from deploy.teleoperator.base import str2dtype

def load_teleoperator(teleop_cfg: dict, args):
    full_path = teleop_cfg['target']
    module_path, class_name = full_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    TeleOpCls = getattr(module, class_name)
    print(f"Creating Teleop Device: {full_path}")
    teleoperator = TeleOpCls(shm_name=args.shm_name, shm_shape=args.shm_shape, shm_dtype=args.action_dtype, frequency=args.freq, **{k:v for k,v in teleop_cfg.items() if k!='target'})
    return teleoperator

def main():
    parser = argparse.ArgumentParser(description='Teleoperation parameters')
    
    parser.add_argument('--tele-config', type=str, default='configuration/teleop/keyboard.yaml',
                        help='Shared memory name for action buffer')
    parser.add_argument('--shm-name', type=str, default='teleop_action_buffer',
                        help='Shared memory name for action buffer')
    parser.add_argument('--action-dim', type=int, default=7,
                        help='Dimension of action space')
    parser.add_argument('--action-dtype', type=str, default='float64',
                        choices=['float32', 'float64'], help='Data type for actions')
    parser.add_argument('--freq', type=float, default=10.0,
                        help='Teleoperation frequency in Hz')
    args = parser.parse_args()
    
    # 定义共享内存的规格
    args.shm_shape = (1,)
    args.action_dtype = np.dtype([
        ('timestamp', np.float64),
        ('action', str2dtype(args.action_dtype), args.action_dim), 
    ])
    args.shm_size = args.action_dtype.itemsize
    
    # 创建共享内存块
    try:
        shm = shared_memory.SharedMemory(name=args.shm_name, create=True, size=args.shm_size)
        print(f"主程序：成功创建共享内存 '{args.shm_name}'，大小 {args.shm_size} 字节。")
    except FileExistsError:
        print(f"主程序：共享内存 '{args.shm_name}' 已存在，将连接到它。")
        shm = shared_memory.SharedMemory(name=args.shm_name)

    # 初始化遥操作设备和机器人控制器，并传入配置
    print(f"Loading teleop configuration from {args.tele_config}")
    with open(args.tele_config, 'r') as f:
        teleop_cfg = yaml.safe_load(f)
    teleop_dev = load_teleoperator(teleop_cfg, args)

    try:
        print("\n--- 控制说明 ---")
        print("平移: W/S (前/后), A/D (左/右), Q/E (上/下)")
        print("旋转: U/O (绕X轴), I/K (绕Y轴), J/L (绕Z轴)")
        print("夹爪: 按住 空格键 闭合, 松开 张开")
        print("按 Ctrl+C 退出程序。")
        print("-----------------\n")
        teleop_dev.run()
        # teleop_process.start()

    except KeyboardInterrupt:
        print("\n主程序：检测到 Ctrl+C，正在请求所有进程停止...")
    finally:
        teleop_dev.stop()
        print("主程序：所有进程已停止。")
        shm.close()
        shm.unlink()
        print(f"主程序：共享内存 '{args.shm_name}' 已被清理。")


if __name__ == '__main__':
    main()
