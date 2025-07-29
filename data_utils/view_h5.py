import h5py

def print_h5_structure(file_path):
    """
    打印H5文件的结构
    """
    def print_group(group, indent=0):
        """
        递归打印组和数据集
        """
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                print("  " * indent + f"Group: {key}")
                print_group(item, indent + 1)  # 递归打印子组
            elif isinstance(item, h5py.Dataset):
                print("  " * indent + f"Dataset: {key} (Shape: {item.shape}, Dtype: {item.dtype})")
            else:
                print("  " * indent + f"Unknown item: {key}")

    with h5py.File(file_path, 'r') as h5_file:
        print(f"File: {file_path}")
        print_group(h5_file)
    
print_h5_structure("/inspire/hdd/project/robot-action/public/data/openx_h5py/fmb/episode_0.hdf5.hdf5")