import tensorflow_datasets as tfds
import dlimp as dl
import io
from PIL import Image
import json
import numpy as np
import os
import argparse
import h5py
from tqdm import tqdm

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def jump_segments(arr, thresh=0.1):
    """
    提取抓夹跳变所有区间（不忽略首索引为 0 的区间）
    Parameters
    ----------
    arr : (n,) array_like
        一维数据
    thresh : float, optional
        判断跳变的阈值，默认 0.1

    Returns
    -------
    segs : list[tuple]
        每个区间的 (start_idx, end_idx)
    """
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError("input must be a 1-D array")
    diff = np.abs(np.diff(arr))
    jump_pos = np.where(diff > thresh)[0]         # 跳变发生在 i -> i+1
    starts = np.concatenate(([0], jump_pos + 1))  # 开始索引
    ends   = np.concatenate((jump_pos, [len(arr) - 1]))  # 结束索引
    return list(zip(starts, ends))

def stationary_segments(pos,
                       vel_threshold=1e-2,   # 判定“很小移动”的距离阈值
                       min_frames=5):        # 区段至少多少帧才被接受
    """
    Parameters
    ----------
    pos : (T,3) array-like
        物体每一帧的 (x,y,z) 坐标。
    vel_threshold : float
        连续两帧间欧氏距离小于该值，就认为这一帧处于“缓慢/停留”状态。
    min_frames : int
        一个区段至少包含的帧数，低于此值将被忽略。

    Returns
    -------
    segments : list of tuple
        每个元素为 (start_idx, end_idx) ，均为 0-based 且闭区间；
        不会返回首索引等于 0 的区段。
    """

    pos = np.asarray(pos)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError('pos 必须是 (T,3) 的数组')

    # 1. 计算逐帧位移大小 (T-1,) ------------------------------
    delta = np.diff(pos, axis=0)                    # (T-1,3)
    step_len = np.linalg.norm(delta, axis=1)        # (T-1,)

    # 2. 得到与原始帧数一致的“慢速”布尔掩码  --------------------
    #    让 slow_mask[i] 描述 “第 i 帧是否处于慢速”
    slow_mask = np.concatenate([[False], step_len < vel_threshold])

    # 3. 找掩码的边缘：0->1 为段起点，1->0 为段终点 ---------------
    change = np.diff(slow_mask.astype(int))         # (T-1,)
    starts = np.where(change ==  1)[0] + 1          # 加 1 得到真正起点
    ends   = np.where(change == -1)[0]              # 此处即段终点
    if slow_mask[-1]:                               # 轨迹末尾仍在停留
        ends = np.append(ends, len(slow_mask)-1)

    # 4. 组装区段，过滤不合格区段 -------------------------------
    segments = []
    for s, e in zip(starts, ends):
        if (e - s + 1) >= min_frames:
            segments.append((s, e))
    segments = [p for p in segments if p[0]>0]
    return segments

def generate_interval_list(n, indices):
    result = []
    indices = sorted(indices)  # 保证输入的list是有序的
    indices.append(-1)  # 防止超出范围，添加一个标志值用于最后的区间
    
    for i in range(n):
        # 找到属于哪个区间
        for j in range(len(indices) - 1):
            if i < indices[j]:
                result.append(indices[j])
                break
        else:
            # 如果i落在最后一个区间之外
            result.append(-1)
    return result

def compute_rel_pose(tgt_pos_idxs, states):
    tgt_states = states[tgt_pos_idxs]
    prev_tid = 0
    all_rel_poses = []
    for tid, tstate in zip(tgt_pos_idxs, tgt_states):
        idxs = list(range(prev_tid, tid))
        for i in idxs:
            all_rel_poses.append(tstate-states[i])
        prev_tid = tid
    return np.stack(all_rel_poses)

def get_direction(dxyz, t: float=0.05) -> str:
    """
    根据三维坐标变化(dx, dy, dz)和阈值t，生成方向描述字符串。

    规则:
    1.  dx > t: "Right", dx < -t: "Left"
    2.  dy > t: "Front", dy < -t: "Back"
    3.  dz > t: "Up",    dz < -t: "Down"
    4.  如果某个轴的绝对值不大于t，则该轴为 "near"。
    5.  如果所有轴都是 "near"，结果为 "near"。
    6.  如果没有轴是 "near"，结果是三个方向的组合，顺序为 x, y, z。
    7.  如果有一或两个轴是 "near"，结果以 "near " 开头，后接非 "near" 轴的方向描述。

    Args:
        dx (float): x轴的变化量。
        dy (float): y轴的变化量。
        dz (float): z轴的变化量。
        t (float): 判断方向的阈值，应为正数。

    Returns:
        str: 最终的方向描述字符串 (英文)。
    """
    dx, dy, dz=dxyz
    if t < 0:
        t = -t # 确保阈值为正
    # 1. 单独判断每个轴的方向
    if dx > t:
        x_desc = "front"
    elif dx < -t:
        x_desc = "back"
    else:
        x_desc = "near"

    if dy > t:
        y_desc = "right"
    elif dy < -t:
        y_desc = "left"
    else:
        y_desc = "near"

    if dz > t:
        z_desc = "up"
    elif dz < -t:
        z_desc = "down"
    else:
        z_desc = "near"

    descriptions = [x_desc, y_desc, z_desc]

    # 2. 统计 "near" 的数量并筛选出非 "near" 的描述
    near_count = descriptions.count("near")
    non_near_descs = [desc for desc in descriptions if desc != "near"]

    # 3. 根据组合规则生成最终结果
    if near_count == 3:
        # 规则: 三个轴都是near，则结果只返回一个near
        return "near"
    elif near_count == 0:
        # 规则: 三个轴都不是near，则返回三个轴的结果组合的字符串
        return " ".join(descriptions)
    else: # near_count is 1 or 2
        # 规则: 有一个或两个轴满足near，在字符串最左侧补上near
        return " ".join(non_near_descs)

def process_episode(episode, reasoning_data):
    # 1. language
    text = episode['language_instruction'][0].numpy().decode()
    # 2. images
    primary_img_bytes = episode['observation']['image'].numpy()
    images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in primary_img_bytes]
    wrist_img_bytes = episode['observation']['wrist_image'].numpy()
    wrist_images = [Image.open(io.BytesIO(img_bytes)) for img_bytes in wrist_img_bytes]
    # 3. actions
    actions = episode['action'].numpy()
    actions[:,:6] = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])*actions[:,:6]
    actions[:,-1] = 0.5*(1-actions[:,-1])
    # 4. states
    states_ee = episode['observation']['state'].numpy() # ee_pos + gripper_state
    states_joint = episode['observation']['joint_state'].numpy() # joint_pos
    # Scale action ...
    episode_id = episode['traj_metadata']['episode_metadata']['file_path'][0].numpy().decode()
    
    # 5. gripper pos
    gripper_pos = np.array(reasoning_data[episode_id]['0']['features']['gripper_position'])
    
    # 6. subtasks
    subtasks = [reasoning_data[episode_id]['0']['reasoning'][str(i)]['subtask'] for i in range(actions.shape[0])]
    
    # 7. target pos
    pos_intervals = stationary_segments(states_ee[:, :3], 0.1)
    gripper_intervals = jump_segments(actions[:,-1])
    all_ends = sorted([p[1] for p in pos_intervals] + [q[1] for q in gripper_intervals])
    target_pos_idxs = generate_interval_list(actions.shape[0], all_ends)
    target_pos = gripper_pos[target_pos_idxs]
    
    # 8. rel pose
    target_poses = states_ee[target_pos_idxs][:,:3]
    delta_poses = target_poses - states_ee[:,:3]
    rel_poses = [get_direction(dp) for dp in delta_poses]
    
    return {
        'language_raw': text,
        'image': images,
        'image_wrist': wrist_images,
        'action': actions,
        'state_ee': states_ee,
        'state_joint': states_joint,
        'gripper_position': gripper_pos,
        'subtask': subtasks,
        'target_position': target_pos,
        'direction': rel_poses,
        'episode_id': episode_id,
    }

def get_dataset(name, data_dir, num_parallel_reads=8):
    builder = tfds.builder(name, data_dir=data_dir)
    full_dataset = dl.DLataset.from_rlds(builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads)
    return full_dataset

def save_data(episode_data, target_path):
    # 重新整理数据
    state_ee = episode_data['state_ee']
    if state_ee.shape[1]==8: 
        state_ee[:,6] = ((state_ee[:,6]-state_ee[:,7])/0.08-0.5)*2.
        state_ee = state_ee[:,:-1]
    state_joint = np.concatenate([episode_data['state_joint'], state_ee[:, -1:].reshape(-1, 1)], axis=1)
    img_size = episode_data['image'][0].size
    cam_width, cam_height = img_size
    gripper_pos = episode_data['gripper_position'].astype(np.float32)
    gripper_pos[:,0]/=cam_width
    gripper_pos[:,1]/=cam_height
    target_pos = episode_data['target_position'].astype(np.float32)
    target_pos[:,0]/=cam_width
    target_pos[:,1]/=cam_height
    state_dim = 7
    action_dim = 7
    max_timesteps = state_ee.shape[0]
    raw_lang = episode_data['language_raw']
    subtasks = episode_data['subtask']
    data_dict = {
        '/freq': 20,
        '/episode_len': max_timesteps,
        '/observations/state_ee': state_ee,
        '/observations/state_joint': state_joint,
        '/observations/image/primary': np.stack(([np.array(img) for img in episode_data['image']])),
        '/observations/image/wrist': np.stack(([np.array(img) for img in episode_data['image_wrist']])),
        '/action_ee': episode_data['action'],
        '/reasoning/gripper_position': gripper_pos,
        '/reasoning/target_position': target_pos,
    }
    
    # dataset_path = os.path.join(data_dir, f'episode_{total_traj_cnt}')
    with h5py.File(target_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = False
        # 创建 observations
        obs = root.create_group('observations')
        image = obs.create_group('image')
        image.create_dataset('primary', (max_timesteps, cam_height, cam_width, 3), dtype='uint8', chunks=(1, cam_height, cam_width, 3), )
        image.create_dataset('wrist', (max_timesteps, cam_height, cam_width, 3), dtype='uint8', chunks=(1, cam_height, cam_width, 3), )
        state_ee = obs.create_dataset('state_ee', (max_timesteps, 7), dtype='float32', chunks=(1, 7))
        state_joint = obs.create_dataset('state_joint', (max_timesteps, 8), dtype='float32', chunks=(1, 8))
        # 创建动作
        action = root.create_dataset('action_ee', (max_timesteps, 7), dtype='float32', chunks=(1, 7))
        # 创建reasoning
        reasoning = root.create_group('reasoning')
        reasoning.create_dataset('gripper_position', (max_timesteps, 2))
        reasoning.create_dataset('target_position', (max_timesteps, 2))
        reasoning.create_dataset('subtask', data=[d.encode('utf-8') for d in subtasks])
        reasoning.create_dataset('direction', data=[d.encode('utf-8') for d in episode_data['direction']])
        # 创建其他属性
        root.create_dataset('freq', (1,))
        root.create_dataset('episode_len', (1,))
        root.create_dataset("language_instruction", data=[raw_lang.encode('utf-8')])
        root.create_dataset("episode_id", data=[episode_data['episode_id'].encode('utf-8')])
        root.create_dataset("dataset_dir", data=[os.path.dirname(target_path).encode('utf-8')])
        root.create_dataset("robot", data=["libero_franka".encode('utf-8')])
        for name, array in data_dict.items():
            root[name][...] = array
        

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', help='Dataset root directory', type=str, default="/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero")
parser.add_argument('--name', help='Target directory name', type=str, default="libero_object")
parser.add_argument('--h5_dir', help='Target directory name', type=str, default="h5v2")
args = parser.parse_args()

reasoning_file = os.path.join(args.data_root, args.name, "reasoning.json")
reasoning_data = load_json(reasoning_file)

# if args.save_dir=="":
#     target_dir = '.'
# else:
target_dir = os.path.join(args.data_root, args.name, args.h5_dir)
os.makedirs(target_dir, exist_ok=True)

if __name__=='__main__':
    ds = get_dataset(args.name, args.data_root)
    num_traj = 0
    for episode in tqdm(ds):
        tgt_file = os.path.join(target_dir, f"episode_{num_traj:05d}.hdf5")
        if os.path.exists(tgt_file): 
            print(f"{tgt_file} already exists." )
            num_traj += 1
            continue
        try:
            episode_data = process_episode(episode, reasoning_data)
            save_data(episode_data, tgt_file)
            num_traj += 1
        except Exception as e:
            print(e)
            continue