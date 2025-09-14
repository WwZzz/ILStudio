"""
自动优化选择器

根据数据集大小、系统内存和GPU数量自动选择最优的数据加载配置
"""

import os
import yaml
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess


class DatasetOptimizer:
    """数据集自动优化器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化优化器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), "optimization_config.yaml")
        
        self.config_path = config_path
        self.config = self._load_config()
        self.system_info = self._get_system_info()
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件未找到: {self.config_path}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'optimization': {
                'enabled': True,
                'memory_mapping': {'enabled': True, 'cache_size': 100},
                'shared_memory': {'enabled': True, 'preload_critical_data': True},
                'loading_strategy': {'full_preload': False}
            },
            'dataloader': {
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True,
                'prefetch_factor': 2,
                'drop_last': True
            },
            'memory_management': {
                'memory_limit': 16.0,
                'auto_gc': True,
                'gc_interval': 50
            }
        }
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        info = {
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'cpu_count': psutil.cpu_count(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_memory_gb': []
        }
        
        # 获取GPU内存信息
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                info['gpu_memory_gb'].append(gpu_memory)
        
        return info
    
    def analyze_dataset(self, dataset_paths: List[str]) -> Dict:
        """分析数据集大小和特征"""
        analysis = {
            'total_size_gb': 0.0,
            'num_files': len(dataset_paths),
            'avg_file_size_gb': 0.0,
            'estimated_episodes': 0,
            'has_images': False,
            'image_resolution': None,
            'num_cameras': 0
        }
        
        import h5py
        
        total_size_bytes = 0
        total_episodes = 0
        
        for path in dataset_paths:
            if not os.path.exists(path):
                print(f"警告: 数据集文件不存在 {path}")
                continue
                
            # 获取文件大小
            file_size = os.path.getsize(path)
            total_size_bytes += file_size
            
            # 分析HDF5文件内容
            try:
                with h5py.File(path, 'r') as f:
                    # 获取episode数量
                    if '/episode_len' in f:
                        episodes = 1
                    else:
                        # 尝试通过action数据推断
                        action_keys = ['/action', '/actions', '/action_ee', '/action_joint']
                        for key in action_keys:
                            if key in f:
                                episodes = 1
                                break
                        else:
                            episodes = 0
                    
                    total_episodes += episodes
                    
                    # 检查图像数据
                    for key in f.keys():
                        if 'image' in key.lower() or 'camera' in key.lower():
                            analysis['has_images'] = True
                            # 尝试获取图像分辨率
                            if isinstance(f[key], h5py.Dataset) and len(f[key].shape) >= 3:
                                analysis['image_resolution'] = f[key].shape[-3:-1]  # (H, W)
                                analysis['num_cameras'] += 1
                            break
                            
            except Exception as e:
                print(f"分析文件时出错 {path}: {e}")
        
        analysis['total_size_gb'] = total_size_bytes / (1024**3)
        analysis['avg_file_size_gb'] = analysis['total_size_gb'] / max(len(dataset_paths), 1)
        analysis['estimated_episodes'] = total_episodes
        
        return analysis
    
    def select_preset(self, dataset_analysis: Dict) -> str:
        """根据数据集分析结果选择预设配置"""
        total_size_gb = dataset_analysis['total_size_gb']
        
        if total_size_gb < 1:
            return 'small'
        elif total_size_gb < 10:
            return 'medium' 
        elif total_size_gb < 100:
            return 'large'
        else:
            return 'xlarge'
    
    def optimize_config(self, dataset_paths: List[str], 
                       batch_size: int = 8,
                       target_memory_usage: float = 0.7) -> Dict:
        """
        自动优化配置
        
        Args:
            dataset_paths: 数据集路径列表
            batch_size: 批次大小
            target_memory_usage: 目标内存使用率 (0-1)
            
        Returns:
            优化后的配置字典
        """
        # 分析数据集
        dataset_analysis = self.analyze_dataset(dataset_paths)
        
        # 选择预设配置
        preset_name = self.select_preset(dataset_analysis)
        preset_config = self.config.get('presets', {}).get(preset_name, {})
        
        # 合并配置
        optimized_config = self._deep_merge(self.config, preset_config)
        
        # 根据系统信息调整配置
        optimized_config = self._adjust_for_system(
            optimized_config, dataset_analysis, batch_size, target_memory_usage
        )
        
        return optimized_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _adjust_for_system(self, config: Dict, dataset_analysis: Dict, 
                          batch_size: int, target_memory_usage: float) -> Dict:
        """根据系统信息调整配置"""
        
        # 调整worker数量
        cpu_cores = self.system_info['cpu_count']
        available_memory = self.system_info['available_memory_gb']
        
        # 根据CPU核心数和内存调整worker数量
        if available_memory < 8:
            # 低内存系统
            suggested_workers = max(1, cpu_cores // 4)
        elif available_memory < 16:
            # 中等内存系统
            suggested_workers = max(2, cpu_cores // 2)
        else:
            # 高内存系统
            suggested_workers = min(cpu_cores, 8)  # 限制最大worker数
        
        config['dataloader']['num_workers'] = suggested_workers
        
        # 调整内存限制
        memory_limit = available_memory * target_memory_usage
        config['memory_management']['memory_limit'] = memory_limit
        
        # 如果数据集很小且内存充足，启用完全预加载
        if (dataset_analysis['total_size_gb'] < available_memory * 0.3 and 
            dataset_analysis['total_size_gb'] < 5):
            config['optimization']['loading_strategy']['full_preload'] = True
        else:
            config['optimization']['loading_strategy']['full_preload'] = False
        
        # 根据GPU数量调整批次相关设置
        if self.system_info['gpu_count'] > 1:
            config['distributed']['use_distributed_sampler'] = True
            # 多GPU时可以增加prefetch_factor
            config['dataloader']['prefetch_factor'] = min(4, config['dataloader']['prefetch_factor'] * 2)
        
        # 如果有图像数据，调整缓存策略
        if dataset_analysis['has_images']:
            # 图像数据占用内存较大，减少缓存
            if available_memory < 16:
                config['cache_strategy']['lru_cache_size'] = 64
                config['optimization']['memory_mapping']['cache_size'] = 50
        
        return config
    
    def print_optimization_report(self, config: Dict, dataset_analysis: Dict):
        """打印优化报告"""
        print("=" * 60)
        print("数据集优化报告")
        print("=" * 60)
        
        print("\n系统信息:")
        print(f"  总内存: {self.system_info['total_memory_gb']:.1f} GB")
        print(f"  可用内存: {self.system_info['available_memory_gb']:.1f} GB")
        print(f"  CPU核心数: {self.system_info['cpu_count']}")
        print(f"  GPU数量: {self.system_info['gpu_count']}")
        
        print("\n数据集分析:")
        print(f"  总大小: {dataset_analysis['total_size_gb']:.2f} GB")
        print(f"  文件数量: {dataset_analysis['num_files']}")
        print(f"  平均文件大小: {dataset_analysis['avg_file_size_gb']:.2f} GB")
        print(f"  预估episode数: {dataset_analysis['estimated_episodes']}")
        print(f"  包含图像: {dataset_analysis['has_images']}")
        if dataset_analysis['image_resolution']:
            print(f"  图像分辨率: {dataset_analysis['image_resolution']}")
        
        print("\n优化配置:")
        print(f"  启用内存映射: {config['optimization']['memory_mapping']['enabled']}")
        print(f"  启用共享内存: {config['optimization']['shared_memory']['enabled']}")
        print(f"  完全预加载: {config['optimization']['loading_strategy']['full_preload']}")
        print(f"  Worker数量: {config['dataloader']['num_workers']}")
        print(f"  内存限制: {config['memory_management']['memory_limit']:.1f} GB")
        print(f"  缓存大小: {config['optimization']['memory_mapping']['cache_size']}")
        
        print("\n建议:")
        if dataset_analysis['total_size_gb'] > self.system_info['available_memory_gb'] * 0.5:
            print("  - 数据集较大，建议使用内存映射而非完全预加载")
        if self.system_info['gpu_count'] > 1:
            print("  - 检测到多GPU，建议使用分布式数据加载")
        if dataset_analysis['has_images'] and self.system_info['available_memory_gb'] < 16:
            print("  - 包含图像数据且内存有限，建议启用图像压缩缓存")
        
        print("=" * 60)


def auto_optimize_dataset(dataset_paths: List[str], 
                         camera_names: List[str],
                         batch_size: int = 8,
                         **kwargs) -> Tuple[Dict, Dict]:
    """
    自动优化数据集配置的便捷函数
    
    Args:
        dataset_paths: 数据集路径列表
        camera_names: 相机名称列表  
        batch_size: 批次大小
        **kwargs: 其他参数
        
    Returns:
        (优化配置, 数据集分析结果)
    """
    optimizer = DatasetOptimizer()
    
    # 分析数据集
    dataset_analysis = optimizer.analyze_dataset(dataset_paths)
    
    # 优化配置
    optimized_config = optimizer.optimize_config(
        dataset_paths, batch_size, kwargs.get('target_memory_usage', 0.7)
    )
    
    # 打印报告
    if kwargs.get('verbose', True):
        optimizer.print_optimization_report(optimized_config, dataset_analysis)
    
    return optimized_config, dataset_analysis


if __name__ == "__main__":
    # 示例用法
    dataset_paths = [
        "/path/to/your/dataset1.hdf5",
        "/path/to/your/dataset2.hdf5"
    ]
    camera_names = ["front_camera", "wrist_camera"]
    
    config, analysis = auto_optimize_dataset(
        dataset_paths, 
        camera_names,
        batch_size=8,
        target_memory_usage=0.7,
        verbose=True
    )
