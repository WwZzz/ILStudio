import torch
import tensorflow as tf
import tensorflow_datasets as tfds
from torch.utils.data import IterableDataset, DataLoader
import os
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except RuntimeError as e:
    print(e)


class WrappedTFDSDataset(IterableDataset):
    """
    一个将 tf.data.Dataset 包装为 PyTorch IterableDataset 的类。
    """
    def __init__(self, 
            dataset_path_list: list, 
            camera_names: list=[], 
            chunk_size: int = 16,  
            ctrl_space: str = 'ee', 
            ctrl_type: str = 'delta',
            *args, 
            **kwargs,
            ):
        super().__init__()
        assert len(dataset_path_list) == 1
        dataset_path = dataset_path_list[0]
        dataset_dir, data_name = os.path.split(dataset_path)
        if dataset_dir=='': dataset_dir = None
        self.dataset = tfds.load(
            data_name,
            data_dir=dataset_dir,
            split='train'
        )
        if dataset_dir=='': dataset_dir = self.dataset.data_dir
        self.dataset_dir = dataset_dir
        self.iterator = self.dataset.as_numpy_iterator()

    def __iter__(self):
        # IterableDataset 需要 __iter__ 方法返回一个迭代器
        return self

    def __next__(self):
        # 从 NumPy 迭代器中获取下一个样本
        image_np, label_np = next(self.iterator)

        # 将 NumPy 数组转换为 PyTorch 张量
        # 注意：图像的维度顺序可能需要调整
        image_tensor = torch.from_numpy(image_np)
        label_tensor = torch.from_numpy(label_np)

        # 检查并调整图像维度：PyTorch 需要 (C, H, W)
        # tfds 'mnist' 产出的图像是 (H, W, C)，我们需要转换它
        if image_tensor.ndim == 3 and image_tensor.shape[2] == 1:
             # 从 (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)

        # 将图像像素值从 [0, 255] 归一化到 [0.0, 1.0]
        image_tensor = image_tensor.to(torch.float32) / 255.0
        
        return image_tensor, label_tensor

if __name__=='__main__':
    
    DATASET_PATH = "/inspire/hdd/global_public/public_datas/Robotics_Related/Open-X-Embodiment/openx"
    DATASET_NAME = 'droid'
    ds = WrappedTFDSDataset([os.path.join(DATASET_PATH, DATASET_NAME)])

    # 2. 将其实例传递给 PyTorch DataLoader
    #    对于IterableDataset，我们不在DataLoader中设置shuffle。打乱操作应在tf.data.Dataset层面完成。
    batch_size = 4
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=0 # 在Windows或简单脚本中设为0通常更稳定
    )
    print('ok')