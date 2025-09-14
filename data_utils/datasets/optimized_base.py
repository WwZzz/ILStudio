"""
Optimized base dataset class using memory mapping and shared memory for efficient data loading.

This module contains the OptimizedEpisodicDataset class that leverages:
1. Memory-mapped HDF5 files for efficient data access
2. Shared memory for multi-process data sharing
3. Lazy loading with caching strategies
"""

import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
import json
import mmap
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
from time import time
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from data_utils.rotate import quat2axisangle
from collections import OrderedDict
import copy
from concurrent.futures import ThreadPoolExecutor
import warnings
import weakref
import gc
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any


class SharedMemoryManager:
    """管理共享内存的类，确保在多进程环境下数据共享"""
    
    def __init__(self):
        self.shared_blocks = {}
        self.metadata = {}
        self._lock = threading.RLock()
        
    def create_shared_array(self, name: str, shape: tuple, dtype: np.dtype) -> np.ndarray:
        """创建共享内存数组"""
        with self._lock:
            if name in self.shared_blocks:
                return self.get_shared_array(name)
                
            size = np.prod(shape) * np.dtype(dtype).itemsize
            try:
                shm = shared_memory.SharedMemory(create=True, size=size, name=name)
            except FileExistsError:
                # 如果已存在，直接连接
                shm = shared_memory.SharedMemory(name=name)
                
            self.shared_blocks[name] = shm
            self.metadata[name] = {'shape': shape, 'dtype': dtype}
            
            # 创建numpy数组视图
            array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            return array
    
    def get_shared_array(self, name: str) -> Optional[np.ndarray]:
        """获取已存在的共享内存数组"""
        if name not in self.shared_blocks:
            try:
                shm = shared_memory.SharedMemory(name=name)
                self.shared_blocks[name] = shm
                if name in self.metadata:
                    meta = self.metadata[name]
                    return np.ndarray(meta['shape'], dtype=meta['dtype'], buffer=shm.buf)
            except FileNotFoundError:
                return None
        else:
            shm = self.shared_blocks[name]
            if name in self.metadata:
                meta = self.metadata[name]
                return np.ndarray(meta['shape'], dtype=meta['dtype'], buffer=shm.buf)
        return None
    
    def cleanup(self):
        """清理共享内存"""
        with self._lock:
            for name, shm in self.shared_blocks.items():
                try:
                    shm.close()
                    shm.unlink()
                except:
                    pass
            self.shared_blocks.clear()
            self.metadata.clear()


class MemoryMappedHDF5:
    """内存映射的HDF5文件包装器"""
    
    def __init__(self, filepath: str, cache_size: int = 100):
        self.filepath = filepath
        self.cache_size = cache_size
        self._file_handle = None
        self._mmap_handle = None
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        
    def __enter__(self):
        self._open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._close()
        
    def _open(self):
        """打开文件并创建内存映射"""
        if self._file_handle is None:
            # 以只读方式打开HDF5文件
            self._file_handle = h5py.File(self.filepath, 'r', rdcc_nbytes=1024**3, rdcc_nslots=10007)
            
    def _close(self):
        """关闭文件和内存映射"""
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        if self._mmap_handle:
            self._mmap_handle.close()
            self._mmap_handle = None
            
    @lru_cache(maxsize=128)
    def get_dataset_info(self, key: str) -> Dict[str, Any]:
        """获取数据集信息（形状、类型等）"""
        with self._lock:
            if self._file_handle is None:
                self._open()
            
            if key in self._file_handle:
                dataset = self._file_handle[key]
                return {
                    'shape': dataset.shape,
                    'dtype': dataset.dtype,
                    'size': dataset.size
                }
        return None
    
    def load_data(self, key: str, slice_obj=None) -> np.ndarray:
        """加载数据，使用缓存机制"""
        cache_key = f"{key}_{slice_obj}"
        
        with self._lock:
            # 检查缓存
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)  # 更新LRU顺序
                return self._cache[cache_key]
            
            # 确保文件已打开
            if self._file_handle is None:
                self._open()
                
            if key not in self._file_handle:
                raise KeyError(f"Key {key} not found in HDF5 file")
                
            # 加载数据
            dataset = self._file_handle[key]
            if slice_obj is not None:
                data = dataset[slice_obj]
            else:
                data = dataset[:]
                
            # 添加到缓存
            if len(self._cache) >= self.cache_size:
                self._cache.popitem(last=False)  # 移除最久未使用的项
            self._cache[cache_key] = data.copy()
            
            return data
    
    def clear_cache(self):
        """清理缓存"""
        with self._lock:
            self._cache.clear()


class OptimizedEpisodicDataset(torch.utils.data.Dataset):
    """
    优化的episodic数据集类，使用内存映射和共享内存
    """
    
    def __init__(self, dataset_path_list: list, camera_names: list, action_normalizers: dict = {},
                 state_normalizers: dict = {}, data_args=None, chunk_size: int = 16,
                 ctrl_space: str = 'ee', ctrl_type: str = 'delta',
                 use_shared_memory: bool = True, memory_map_cache_size: int = 100,
                 preload_critical_data: bool = True):
        """
        初始化优化的episodic数据集
        
        Args:
            dataset_path_list: 数据集文件路径列表
            camera_names: 相机名称列表
            action_normalizers: 动作归一化器字典
            state_normalizers: 状态归一化器字典
            data_args: 数据处理参数
            chunk_size: 每个样本的时间步数
            ctrl_space: 控制空间类型
            ctrl_type: 控制类型
            use_shared_memory: 是否使用共享内存
            memory_map_cache_size: 内存映射缓存大小
            preload_critical_data: 是否预加载关键数据（如episode长度）
        """
        super(OptimizedEpisodicDataset, self).__init__()
        
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.action_normalizers = action_normalizers
        self.state_normalizers = state_normalizers
        self.data_args = data_args
        self.chunk_size = chunk_size
        self.ctrl_space = ctrl_space
        self.ctrl_type = ctrl_type
        self.use_shared_memory = use_shared_memory
        self.memory_map_cache_size = memory_map_cache_size
        self.preload_critical_data = preload_critical_data
        
        # 初始化共享内存管理器
        if self.use_shared_memory:
            self.shared_memory_manager = SharedMemoryManager()
        else:
            self.shared_memory_manager = None
            
        # 初始化内存映射文件句柄
        self.mmap_files = {}
        self._init_memory_mapped_files()
        
        # 获取episode信息
        self.episode_ids = list(range(len(self.dataset_path_list)))
        self.episode_len = self._get_episode_lengths()
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(self.episode_len) if self.episode_len else 0
        
        # 预加载关键数据到共享内存
        if self.preload_critical_data and self.use_shared_memory:
            self._preload_critical_data()
            
    def _init_memory_mapped_files(self):
        """初始化内存映射文件"""
        for i, dataset_path in enumerate(self.dataset_path_list):
            self.mmap_files[i] = MemoryMappedHDF5(dataset_path, self.memory_map_cache_size)
            
    def _get_episode_lengths(self) -> List[int]:
        """获取每个episode的长度"""
        episode_lengths = []
        
        for i, dataset_path in enumerate(self.dataset_path_list):
            try:
                with self.mmap_files[i] as mmap_file:
                    # 首先尝试获取预存储的episode长度
                    if '/episode_len' in mmap_file._file_handle:
                        length = mmap_file.load_data('/episode_len')[0].astype(int)
                    else:
                        # 回退到使用action数据推断长度
                        action_keys = ['/action', '/actions', '/action_ee', '/action_joint', '/state']
                        for key in action_keys:
                            info = mmap_file.get_dataset_info(key)
                            if info:
                                length = info['shape'][0]
                                break
                        else:
                            raise ValueError(f"无法确定episode长度: {dataset_path}")
                            
                episode_lengths.append(length)
                
            except Exception as e:
                print(f"加载episode长度时出错 {dataset_path}: {e}")
                episode_lengths.append(0)
                
        return episode_lengths
    
    def _preload_critical_data(self):
        """预加载关键数据到共享内存（如episode长度、元数据等）"""
        print("预加载关键数据到共享内存...")
        
        # 创建episode长度的共享数组
        episode_len_name = f"episode_len_{id(self)}"
        shared_episode_len = self.shared_memory_manager.create_shared_array(
            episode_len_name, (len(self.episode_len),), np.int32
        )
        shared_episode_len[:] = np.array(self.episode_len, dtype=np.int32)
        
        print("关键数据预加载完成")
    
    def _locate_transition(self, index: int) -> Tuple[int, int]:
        """将样本索引转换为episode索引和内部时间步"""
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index)
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts
    
    def _load_data_with_fallback(self, episode_id: int, key: str, slice_obj=None) -> np.ndarray:
        """从内存映射文件加载数据，带有错误回退机制"""
        try:
            return self.mmap_files[episode_id].load_data(key, slice_obj)
        except Exception as e:
            # 回退到传统的HDF5加载方式
            print(f"内存映射加载失败，回退到传统方式: {e}")
            dataset_path = self.dataset_path_list[episode_id]
            with h5py.File(dataset_path, 'r') as f:
                if slice_obj is not None:
                    return f[key][slice_obj]
                else:
                    return f[key][:]
    
    def load_onestep_from_episode(self, dataset_path: str, start_ts: int) -> Dict[str, Any]:
        """
        从episode中加载一步数据
        
        Args:
            dataset_path: 数据集文件路径
            start_ts: 起始时间步
            
        Returns:
            包含加载数据的字典
        """
        # 找到对应的episode_id
        episode_id = self.dataset_path_list.index(dataset_path)
        episode_len = self.episode_len[episode_id]
        
        # 计算实际的时间步范围
        end_ts = min(start_ts + self.chunk_size, episode_len)
        actual_chunk_size = end_ts - start_ts
        
        # 使用内存映射加载数据
        try:
            # 加载动作数据
            action_keys = ['/action', '/actions', '/action_ee', '/action_joint']
            action_data = None
            for key in action_keys:
                try:
                    action_data = self._load_data_with_fallback(
                        episode_id, key, slice(start_ts, end_ts)
                    )
                    break
                except KeyError:
                    continue
            
            if action_data is None:
                raise ValueError("无法找到动作数据")
                
            # 加载状态数据
            try:
                state_data = self._load_data_with_fallback(
                    episode_id, '/state', slice(start_ts, start_ts + 1)
                )[0]  # 只取当前时刻的状态
            except KeyError:
                state_data = np.array([])  # 如果没有状态数据
                
            # 加载图像数据
            image_dict = {}
            for cam_name in self.camera_names:
                try:
                    # 只加载当前时刻的图像
                    img_key = f'/observations/images/{cam_name}'
                    if img_key not in self.mmap_files[episode_id]._file_handle:
                        img_key = f'/{cam_name}'  # 尝试另一种路径格式
                        
                    image_data = self._load_data_with_fallback(
                        episode_id, img_key, slice(start_ts, start_ts + 1)
                    )[0]
                    image_dict[cam_name] = image_data
                except KeyError:
                    print(f"警告: 无法找到相机 {cam_name} 的图像数据")
                    # 创建dummy图像数据
                    image_dict[cam_name] = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # 加载语言指令
            try:
                lang_data = self._load_data_with_fallback(episode_id, '/language_instruction')
                if isinstance(lang_data, np.ndarray) and lang_data.dtype.kind in 'SU':
                    raw_lang = lang_data[0] if lang_data.ndim > 0 else str(lang_data)
                else:
                    raw_lang = str(lang_data)
            except KeyError:
                raw_lang = ""  # 默认空字符串
                
            # 加载推理数据（如果存在）
            try:
                reasoning_data = self._load_data_with_fallback(episode_id, '/reasoning')
                if isinstance(reasoning_data, np.ndarray):
                    reasoning = reasoning_data[0] if reasoning_data.ndim > 0 else str(reasoning_data)
                else:
                    reasoning = str(reasoning_data)
            except KeyError:
                reasoning = ""
                
        except Exception as e:
            print(f"内存映射数据加载失败: {e}")
            # 完全回退到传统加载方式
            return self._fallback_load_onestep(dataset_path, start_ts)
        
        return {
            'action': action_data,
            'image': image_dict,
            'state': state_data,
            'language_instruction': raw_lang,
            'reasoning': reasoning
        }
    
    def _fallback_load_onestep(self, dataset_path: str, start_ts: int) -> Dict[str, Any]:
        """回退的数据加载方法"""
        # 这里可以调用原始的加载逻辑
        # 或者实现一个简化版本
        raise NotImplementedError("请在子类中实现回退加载方法")
    
    def load_feat_from_episode(self, dataset_path: str, feats: List[str] = []) -> Dict[str, Any]:
        """从episode加载特征数据"""
        episode_id = self.dataset_path_list.index(dataset_path)
        
        result = {}
        for feat in feats:
            try:
                result[feat] = self._load_data_with_fallback(episode_id, feat)
            except KeyError:
                print(f"警告: 特征 {feat} 不存在")
                
        return result
    
    def extract_from_episode(self, episode_idx: int, keyname: List[str] = []) -> Dict[str, Any]:
        """从episode提取数据"""
        episode_path = self.dataset_path_list[episode_idx]
        return self.load_feat_from_episode(episode_path, keyname)
    
    def clear_cache(self):
        """清理所有缓存"""
        for mmap_file in self.mmap_files.values():
            mmap_file.clear_cache()
        gc.collect()
    
    def __len__(self) -> int:
        """返回数据集中样本的总数"""
        return sum(self.episode_len)
    
    @property
    def num_episodes(self) -> int:
        return len(self.dataset_path_list)
    
    def get_dataset_dir(self) -> str:
        """获取数据集目录路径"""
        return os.path.dirname(self.dataset_path_list[0])
    
    def get_language_instruction(self) -> str:
        """获取语言指令，需要在子类中实现"""
        raise NotImplementedError("子类必须实现 get_language_instruction")
    
    def set_action_normalizers(self, ns: dict):
        """设置动作归一化器"""
        self.action_normalizers = ns
    
    def set_state_normalizers(self, ns: dict):
        """设置状态归一化器"""
        self.state_normalizers = ns
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """获取数据集中的一个样本"""
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        
        # 加载数据
        data_dict = self.load_onestep_from_episode(dataset_path, start_ts)
        action, image_dict, state = data_dict['action'], data_dict['image'], data_dict['state']
        raw_lang = data_dict['language_instruction']
        reasoning = data_dict.get('reasoning', '')
        
        # 填充动作序列
        padded_action = np.zeros((self.chunk_size, action.shape[1]), dtype=np.float32)
        padded_action[:action.shape[0]] = action
        is_pad = np.zeros(self.chunk_size)
        is_pad[action.shape[0]:] = 1
        
        # 处理图像数据
        all_cam_images = []
        for cam_name in self.camera_names:
            if cam_name in image_dict:
                all_cam_images.append(image_dict[cam_name])
            else:
                # 创建dummy图像
                all_cam_images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        all_cam_images = np.stack(all_cam_images, axis=0)
        
        # 数据归一化
        action_normalizer = self.action_normalizers.get(self.get_dataset_dir(), None)
        if action_normalizer is not None:
            action_data = action_normalizer.normalize(padded_action, datatype='action')
        else:
            action_data = padded_action
            
        state_normalizer = self.state_normalizers.get(self.get_dataset_dir(), None)
        if state_normalizer is not None:
            state_data = state_normalizer.normalize(state, datatype='state')
        else:
            state_data = state
        
        # 转换为tensor
        image_data = torch.from_numpy(all_cam_images)
        state_data = torch.from_numpy(state_data).float()
        action_data = torch.from_numpy(action_data).float()
        is_pad = torch.from_numpy(is_pad).bool()
        
        # 调整图像维度顺序 (H, W, C) -> (C, H, W)
        image_data = torch.einsum('k h w c -> k c h w', image_data)
        
        sample = {
            'image': image_data,
            'state': state_data,
            'action': action_data,
            'is_pad': is_pad,
            'raw_lang': raw_lang,
            'reasoning': reasoning
        }
        
        return sample
    
    def __del__(self):
        """析构函数，清理资源"""
        # 关闭内存映射文件
        for mmap_file in self.mmap_files.values():
            try:
                mmap_file._close()
            except:
                pass
                
        # 清理共享内存
        if self.shared_memory_manager:
            try:
                self.shared_memory_manager.cleanup()
            except:
                pass
