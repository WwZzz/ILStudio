import tensorflow_datasets as tfds 
from policy.openvla.prismatic.vla.datasets.datasets import RLDSDataset

if __name__=='__main__':
    # ds = tfds.load('bridge_orig', split='train', data_dir='/inspire/hdd/project/robot-action/wangzheng-240308120196/data/bridge_data_v2')
    data = RLDSDataset(data_root_dir='/inspire/hdd/project/robot-action/wangzheng-240308120196/data/bridge_data_v2', data_mix='bridge', batch_transform=lambda x:x, resize_resolution=[256, 256])
    print('ok')