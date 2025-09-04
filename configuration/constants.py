TASK_CONFIGS = {
    'example_tasks': { # for local debug
        'dataset_dir': [
            "/inspire/hdd/global_user/wangzheng-240308120196/DexVLA/data"
        ],
        'episode_len': 1000,  
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist'] # replacing with your real keys in h5py formatted data
    },
    'droid_mini':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/droid_mini"
        ],
        'episode_len': 1000,
        'camera_names': ['left', 'right']
    },
    'libero_test':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/libero_h5"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },
    'libero_object':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_object/h5v2"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },
    'libero_object_1':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_object/h5v2"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },
    'libero_spatial':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_spatial/h5"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },    
    'libero_goal':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_goal/h5"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },
    'libero_10':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_10/h5"
        ],
        'episode_len': 400,
        'camera_names': ['primary']
    },
    'libero_all':{
        'dataset_dir': [
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_10/h5",
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_object/h5",
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_goal/h5",
            "/inspire/hdd/project/robot-action/public/data/VLA-OS-Dataset/libero/libero_spatial/h5" 
        ],
        'episode_len': 400,
        'camera_names': ['top']
    },
    'transfer_cube_plus':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_transfer_cube_scripted_plus',
        ],
        'episode_len': 400,
        'camera_names': ['primary'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'sim_transfer_cube':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/act_aloha/sim_transfer_cube_human',
        ],
        'episode_len': 400,
        'camera_names': ['primary'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'sim_transfer_cube_scripted':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/act_aloha/sim_transfer_cube_scripted',
        ],
        'episode_len': 400,
        'camera_names': ['primary'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'transfer_cube_top':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_transfer_cube_scripted',
        ],
        'episode_len': 400,
        'camera_names': ['primary'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'insertion_top':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_insertion_scripted',
        ],
        'episode_len': 400,
        'camera_names': ['primary'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'aloha_transfer_cube':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/wangzheng-240308120196/act-plus-plus-main/data/sim_transfer_cube_scripted',
        ],
        'episode_len': 400,
        'camera_names': ['primary', 'wrist_left', 'wrist_right'],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'sii_transfer_cube':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/red_cube_and_blue_cup/success_3',
            '/inspire/hdd/project/robot-action/public/data/red_cube_and_blue_cup/success_2',
            '/inspire/hdd/project/robot-action/public/data/red_cube_and_blue_cup/success_1',
        ],
        'episode_len': 400,
        'camera_names': ['primary', 'wrist_left', 'wrist_right'],
        'dataset_class': 'AlohaSIIDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'robomimic_square':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/robomimic/square/ph',
            '/inspire/hdd/project/robot-action/public/data/robomimic/square/mh',
        ],
        'episode_len': 400,
        'camera_names': ['primary', ],
        'dataset_class': 'RobomimicDataset',
        'ctrl_type': 'delta',
        'ctrl_space': 'ee',
    },
    'robomimic_square_ph':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/robomimic/square/ph',
        ],
        'episode_len': 400,
        'camera_names': ['primary', ],
        'dataset_class': 'RobomimicDataset',
        'ctrl_type': 'delta',
        'ctrl_space': 'ee',
    },
    'NutAssemblySquare_Panda':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/robomimic/square/ph',
        ],
        'episode_len': 400,
        'camera_names': ['primary', ],
        'dataset_class': 'RobomimicDataset',
        'ctrl_type': 'delta',
        'ctrl_space': 'ee',
    },
    'agilex_transfer_cube': {
        'dataset_dir': [
            '/home/agilex/wz/data/aloha_sim/sim_transfer_cube_scripted',
        ],
        'camera_names': ['primary',],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'sii_cut_steak':{
        'dataset_dir': [
            '/home/agilex/zxc/cobot_magic/collect_data/cut_steak/cut_steak',
        ],
        'episode_len': 400,
        'camera_names': ['primary', 'wrist_left', 'wrist_right'],
        'dataset_class': 'AlohaSIIDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'transfer_human':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/act_aloha/sim_transfer_cube_human',
        ],
        'camera_names': ['primary',],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'insertion_human':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/act_aloha/sim_insertion_human',
        ],
        'camera_names': ['primary',],
        'dataset_class': 'AlohaSimDataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
    'sii_pick_twice':{
        'dataset_dir': [
            '/inspire/hdd/project/robot-action/public/data/sfo_train/pick_two_objects',
        ],
        'camera_names': ['primary', 'wrist_left', 'wrist_right'],
        'dataset_class': 'AlohaSIIv2Dataset',
        'ctrl_type': 'abs',
        'ctrl_space': 'joint',
    },
}