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
        'camera_names': ['primary']
    },
}

### ALOHA fixed constants
DT = 0.02
JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
START_ARM_POSE = [0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239,  0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239]
FPS = 50
# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
MASTER_GRIPPER_POSITION_OPEN = 0.02417
MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
MASTER_GRIPPER_JOINT_OPEN = 0.3083
MASTER_GRIPPER_JOINT_CLOSE = -0.6842
PUPPET_GRIPPER_JOINT_OPEN = 1.4910
PUPPET_GRIPPER_JOINT_CLOSE = -0.6213

############################ Helper functions ############################

MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
