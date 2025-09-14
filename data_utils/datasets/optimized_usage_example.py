"""
使用优化数据集的示例代码

展示如何在单机多卡环境下使用优化的内存映射和共享内存数据集
"""

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import numpy as np
from optimized_base import OptimizedEpisodicDataset


class OptimizedRobomimicDataset(OptimizedEpisodicDataset):
    """
    优化的Robomimic数据集实现
    """
    
    def __init__(self, *args, **kwargs):
        # 添加Robomimic特定的参数
        self.freq = kwargs.pop('freq', 10)  # 默认频率
        super().__init__(*args, **kwargs)
    
    def get_language_instruction(self):
        """返回任务的语言指令"""
        # 根据数据集路径推断任务类型
        dataset_dir = self.get_dataset_dir()
        if 'lift' in dataset_dir.lower():
            return "lift the cube"
        elif 'can' in dataset_dir.lower():
            return "pick up the can"
        elif 'square' in dataset_dir.lower():
            return "insert the peg into the hole"
        else:
            return "complete the manipulation task"
    
    def get_freq(self):
        """获取数据集频率"""
        return self.freq
    
    def _fallback_load_onestep(self, dataset_path: str, start_ts: int):
        """回退的数据加载方法，使用传统HDF5加载"""
        import h5py
        
        episode_id = self.dataset_path_list.index(dataset_path)
        episode_len = self.episode_len[episode_id]
        end_ts = min(start_ts + self.chunk_size, episode_len)
        
        with h5py.File(dataset_path, 'r') as f:
            # 加载动作数据
            action_keys = ['/action', '/actions', '/action_ee', '/action_joint']
            action_data = None
            for key in action_keys:
                if key in f:
                    action_data = f[key][start_ts:end_ts]
                    break
            
            if action_data is None:
                raise ValueError("无法找到动作数据")
            
            # 加载状态数据
            if '/obs/robot0_eef_pos' in f and '/obs/robot0_eef_quat' in f:
                eef_pos = f['/obs/robot0_eef_pos'][start_ts]
                eef_quat = f['/obs/robot0_eef_quat'][start_ts]
                state_data = np.concatenate([eef_pos, eef_quat])
            elif '/state' in f:
                state_data = f['/state'][start_ts]
            else:
                state_data = np.array([])
            
            # 加载图像数据
            image_dict = {}
            for cam_name in self.camera_names:
                img_key = f'/obs/{cam_name}_image'
                if img_key in f:
                    image_dict[cam_name] = f[img_key][start_ts]
                else:
                    # 创建dummy图像
                    image_dict[cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 语言指令
            raw_lang = self.get_language_instruction()
            reasoning = ""
        
        return {
            'action': action_data,
            'image': image_dict,
            'state': state_data,
            'language_instruction': raw_lang,
            'reasoning': reasoning
        }


def create_optimized_dataloader(dataset_paths, camera_names, batch_size=8, 
                              num_workers=4, use_distributed=False,
                              **dataset_kwargs):
    """
    创建优化的数据加载器
    
    Args:
        dataset_paths: 数据集路径列表
        camera_names: 相机名称列表
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        use_distributed: 是否使用分布式采样
        **dataset_kwargs: 数据集其他参数
    
    Returns:
        DataLoader对象
    """
    
    # 创建优化数据集
    dataset = OptimizedRobomimicDataset(
        dataset_path_list=dataset_paths,
        camera_names=camera_names,
        use_shared_memory=True,  # 启用共享内存
        memory_map_cache_size=100,  # 内存映射缓存大小
        preload_critical_data=True,  # 预加载关键数据
        **dataset_kwargs
    )
    
    # 创建采样器
    sampler = None
    if use_distributed:
        sampler = DistributedSampler(dataset)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),  # 如果有sampler就不要shuffle
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # 启用内存固定以加速GPU传输
        persistent_workers=True,  # 保持worker进程存活
        prefetch_factor=2,  # 预取因子
        drop_last=True
    )
    
    return dataloader, dataset


def worker_init_fn(worker_id):
    """
    数据加载器worker初始化函数
    设置每个worker的随机种子和共享内存访问
    """
    import random
    import numpy as np
    
    # 设置随机种子
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def test_memory_efficiency():
    """
    测试内存使用效率
    """
    import psutil
    import time
    
    # 模拟数据集路径（请替换为实际路径）
    dataset_paths = [
        "/path/to/your/dataset1.hdf5",
        "/path/to/your/dataset2.hdf5",
    ]
    camera_names = ["front_camera", "wrist_camera"]
    
    print("开始内存效率测试...")
    
    # 记录初始内存使用
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初始内存使用: {initial_memory:.2f} MB")
    
    # 创建优化数据集
    start_time = time.time()
    dataloader, dataset = create_optimized_dataloader(
        dataset_paths, 
        camera_names,
        batch_size=4,
        num_workers=2
    )
    creation_time = time.time() - start_time
    
    after_creation_memory = process.memory_info().rss / 1024 / 1024
    print(f"数据集创建后内存使用: {after_creation_memory:.2f} MB")
    print(f"数据集创建时间: {creation_time:.2f} 秒")
    
    # 测试数据加载
    print("开始数据加载测试...")
    load_times = []
    
    for i, batch in enumerate(dataloader):
        if i >= 10:  # 只测试10个批次
            break
            
        start_time = time.time()
        # 模拟使用数据
        images = batch['image']
        actions = batch['action']
        states = batch['state']
        load_time = time.time() - start_time
        load_times.append(load_time)
        
        if i == 0:
            print(f"批次形状 - 图像: {images.shape}, 动作: {actions.shape}, 状态: {states.shape}")
    
    # 统计加载性能
    avg_load_time = np.mean(load_times)
    max_memory = process.memory_info().rss / 1024 / 1024
    
    print(f"平均批次加载时间: {avg_load_time:.4f} 秒")
    print(f"最大内存使用: {max_memory:.2f} MB")
    print(f"内存增长: {max_memory - initial_memory:.2f} MB")
    
    # 清理缓存
    dataset.clear_cache()
    print("缓存清理完成")


def multi_gpu_training_example():
    """
    多GPU训练示例
    """
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    
    # 初始化分布式环境（通常由accelerate或torchrun处理）
    # dist.init_process_group("nccl")
    
    # 设置设备
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # 创建数据加载器
    dataset_paths = ["/path/to/dataset1.hdf5", "/path/to/dataset2.hdf5"]
    camera_names = ["front_camera", "wrist_camera"]
    
    dataloader, dataset = create_optimized_dataloader(
        dataset_paths,
        camera_names,
        batch_size=8,
        num_workers=4,
        use_distributed=True  # 启用分布式采样
    )
    
    print(f"进程 {local_rank}: 数据集大小 {len(dataset)}, DataLoader批次数 {len(dataloader)}")
    
    # 模拟训练循环
    for epoch in range(2):
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 5:  # 只运行几个批次作为示例
                break
                
            # 将数据移到GPU
            images = batch['image'].to(device, non_blocking=True)
            actions = batch['action'].to(device, non_blocking=True)
            states = batch['state'].to(device, non_blocking=True)
            
            print(f"Epoch {epoch}, Batch {batch_idx}, GPU {local_rank}: "
                  f"处理了 {images.shape[0]} 个样本")
            
            # 这里添加实际的模型训练代码
            # loss = model(images, states, actions)
            # loss.backward()
            # optimizer.step()


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    print("运行内存效率测试...")
    try:
        test_memory_efficiency()
    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保数据集路径正确且文件存在")
    
    print("\n多GPU训练示例...")
    print("使用 'accelerate launch --multi_gpu --num_processes=2 optimized_usage_example.py' 来运行多GPU示例")
