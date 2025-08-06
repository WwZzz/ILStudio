import h5py

def extract_from_h5(file):
    with h5py.File(file) as root:
        action = root['/action'][()]
        images = root['/observations/images/top'][()]
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
    # 处理抓夹和qpos
    
    return {
        'action': action,
        'images': images,
        'qpos': qpos,
        'qvel': qvel
    }



# def save_file(data, file_path):
if __name__=='__main__':
    data = extract_from_h5('/inspire/hdd/project/robot-action/public/data/act_aloha/sim_transfer_cube_human/episode_0.hdf5')
    print("Action Gripper L:", data['action'][:,6])
    print("Action Gripper R:", data['action'][:,-1])
    print("Qpos Gripper L:", data['qpos'][:,6])
    print("Qpos Gripper R:", data['qpos'][:,-1])