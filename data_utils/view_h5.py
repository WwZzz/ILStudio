import h5py
import argparse

def print_h5_structure(file_path):
    """
    View the structure of an H5 file.
    """
    def print_group(group, indent=0):
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                print("  " * indent + f"Group: {key}")
                print_group(item, indent + 1) 
            elif isinstance(item, h5py.Dataset):
                print("  " * indent + f"Dataset: {key} (Shape: {item.shape}, Dtype: {item.dtype})")
            else:
                print("  " * indent + f"Unknown item: {key}")

    with h5py.File(file_path, 'r') as h5_file:
        print(f"File: {file_path}")
        print_group(h5_file)

parser = argparse.ArgumentParser()
parser.add_argument('--file', help='Target directory name', type=str, default="/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_10/h5/episode_00335.hdf5")
args = parser.parse_args()
print_h5_structure(args.file)