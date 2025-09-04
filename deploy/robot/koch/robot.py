from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.robots.koch_follower import KochFollowerConfig, KochFollower
from deploy.robot.base import BaseRobot
import numpy as np
import traceback
from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

class KochFollowerWithCamera(BaseRobot):
    def __init__(self, com: str="COM8", robot_id: str="koch_follower_arm", camera_configs: dict={}, extra_args=None, **kwargs):
        import threading, queue
        super().__init__()
        # 机械臂部分
        self._robot = KochFollower(KochFollowerConfig(port=com, id=robot_id, cameras={}))
        self._motors = list(self._robot.bus.motors)
        # 相机部分
        self.cam_names = list(camera_configs.keys())
        self.cameras = {key: OpenCVCamera(OpenCVCameraConfig(**value)) for key, value in camera_configs.items()}
        # 串口操作队列和线程
        self._serial_queue = queue.Queue()
        self._serial_thread = threading.Thread(target=self._serial_worker, daemon=True)
        self._serial_thread.start()
        # 观测结果缓存
        self._obs_result = None
        self._obs_event = threading.Event()
    
    def connect(self):
        try:
            if not self._robot.is_connected:
                self._robot.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Robot already connected: {e}")
            pass
        except Exception as e:
            print(f"Failed to connect to robot due to {e}")
            traceback.print_exc()
            return False
        print("Robot connected")
        try:  
            for cam in self.cameras.values():
                cam.connect()
        except DeviceAlreadyConnectedError as e:
            print(f"Camera already connected: {e}")
            pass
        except Exception as e:
            print(f"Failed to connect to camera due to {e}")
            traceback.print_exc()
            return False
        print("Cameras connected")
        return True
    
    def get_action_dim(self):
        return len(self._motors)
    
    def get_low_dim(self, timeout=0.1):
        self._obs_event.clear()
        self._serial_queue.put(('get_observation', None))
        got = self._obs_event.wait(timeout=timeout)
        if got:
            return self._obs_result
        else:
            return self._obs_result

    def get_camera_image(self, cam_name='primary'):
        # 独立获取相机图像
        if cam_name in self.cameras:
            return self.cameras[cam_name].read()
        else:
            return None
    
    def get_observation(self):
        data_lowdim = self.get_low_dim()
        data_rgb = {cam: self.get_camera_image(cam) for cam in self.cam_names}
        data_lowdim.update({'image': data_rgb})
        return data_lowdim

    def shutdown(self):
        if self._robot.is_connected:
            self._robot.disconnect()
        for cam in self.cameras.values():
            if cam.is_connected:
                cam.disconnect()
        
    def publish_action(self, action: np.ndarray):
        action_dict = {mname+'.pos': action[i] for i, mname in enumerate(self._motors)}
        self._serial_queue.put(('publish_action', action_dict))
        
    def _serial_worker(self):
        while True:
            try:
                op, data = self._serial_queue.get()
                if op == 'publish_action':
                    self._robot.send_action(data)
                elif op == 'get_observation':
                    obs = {}
                    observation = self._robot.get_observation()
                    qpos = np.array([observation[mname+'.pos'] for mname in self._motors], dtype=np.float32)
                    obs['qpos'] = qpos
                    self._obs_result = obs
                    self._obs_event.set()
            except Exception as e:
                print(f"[SerialWorker] Error: {e}")
    
    def is_running(self):
        return self._robot.is_connected


# teleop_config = KochLeaderConfig(
#     port="COM7",
#     id="my_blue_leader_arm",
# )

# robot = KochFollower(robot_config)
# robot.connect()
# teleop_device = KochLeader(teleop_config)
# teleop_device.connect()
# teleop_device.bus.write("Drive_Mode", "elbow_flex", 0)

# input("enter to continue")
# try:
#     while True:
#         observation = robot.get_observation()
#         action = teleop_device.get_action()
#         robot.send_action(action)
# except (ConnectionError, TimeoutError, KeyboardInterrupt) as e:
#     print(f"Failed to access observations due to {e}")
#     robot.disconnect()
#     teleop_device.disconnect()