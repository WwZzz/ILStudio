from torch.utils.data import IterableDataset
import tensorflow_datasets as tfds
import dlimp as dl
import io
from PIL import Image
import json
import numpy as np
import os
import h5py
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

class VLAOSDataset(IterableDataset):
    def __init__(self, dataset_dir:str, dname:str, num_parallel_reads: int=4, *args, **kwargs):
        super().__init__()
        builder = tfds.builder(dname, data_dir=dataset_dir)
        self.dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads)
        self.dataset_dir = dataset_dir
        self.dname = dname
        self.reasoning_file = os.path.join(dataset_dir, dname, "reasoning.json")
        if not os.path.exists(self.reasoning_file):
            raise FileNotFoundError(f"Reasoning file not found: {self.reasoning_file}")
        self.reasoning_data = load_json(self.reasoning_file)
        
    def __iter__(self):
        for data in self.dataset.as_numpy_iterator():
            yield data
            
if __name__=='__main__':
    ds = VLAOSDataset('/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero', 'libero_object')
    data = next(iter(ds))
    print('ok')