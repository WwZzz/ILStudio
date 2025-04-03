from benchmark.base import MetaEnv
import gymnasium as gym
import numpy as np
import mani_skill.envs

TASK_DESCS = {'MS-AntRun-v1': 'Ant moves in x direction at 4 m/s', 'MS-AntWalk-v1': 'Ant moves in x direction at 0.5 m/s', 'MS-CartpoleBalance-v1': 'Use the Cartpole robot to balance a pole on a cart.', 'MS-CartpoleSwingUp-v1': 'Use the Cartpole robot to swing up a pole on a cart.', 'MS-HopperHop-v1': 'Hopper robot stays upright and moves in positive x direction with hopping motion', 'MS-HopperStand-v1': 'Hopper robot stands upright', 'MS-HumanoidRun-v1': 'Humanoid moves in x direction at running pace', 'MS-HumanoidStand-v1': 'Humanoid robot stands upright', 'MS-HumanoidWalk-v1': 'Humanoid moves in x direction at walking pace', 'AssemblingKits-v1': 'The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.', 'LiftPegUpright-v1': 'A simple task where the objective is to move a peg laying on the table to any upright position on the table', 'PegInsertionSide-v1': 'Pick up a orange-white peg and insert the orange end into the box with a hole in it.', 'PickCube-v1': 'A simple task where the objective is to grasp a red cube and move it to a target goal position.', 'PickSingleYCB-v1': 'Pick up a random object sampled from the YCB dataset and move it to a random goal position', 'PlaceSphere-v1': 'Place the sphere into the shallow bin.', 'PlugCharger-v1': 'The robot must pick up one of the misplaced shapes on the board/kit and insert it into the correct empty slot.', 'PokeCube-v1': 'A simple task where the objective is to poke a red cube with a peg and push it to a target goal position.', 'PullCube-v1': 'A simple task where the objective is to pull a cube onto a target.', 'PullCubeTool-v1': 'tool to pull a cube that is out of it’s reach', 'PushCube-v1': 'A simple task where the objective is to push and move a cube to a goal region in front of it', 'PushT-v1': 'A simulated version of the real-world push-T task from Diffusion Policy: https://diffusion-policy.cs.columbia.edu/', 'RollBall-v1': 'A simple task where the objective is to push and roll a ball to a goal region at the other end of the table', 'StackCube-v1': 'The goal is to pick up a red cube and stack it on top of a green cube and let go of the cube without it falling', 'TwoRobotPickCube-v1': 'to move the cube to the goal.', 'TwoRobotStackCube-v1': 'A collaborative task where two robot arms need to work together to stack two cubes. One robot must pick up the green cube and place it on the target region, while the other robot picks up the blue cube and stacks it on top of the green cube.', 'AnymalC-Reach-v1': 'Control the AnymalC robot to reach a target location in front of it. Note the current reward function works but more needs to be added to constrain the learned quadruped gait looks more natural', 'AnymalC-Spin-v1': 'Control the AnymalC robot to spin around in place as fast as possible and is rewarded by its angular velocity.', 'UnitreeG1PlaceAppleInBowl-v1': 'Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side', 'UnitreeG1TransportBox-v1': 'A G1 humanoid robot must find a box on a table and transport it to the other table and place it there.', 'OpenCabinetDrawer-v1': 'Use the Fetch mobile manipulation robot to move towards a target cabinet and open the target drawer out.', 'Table-Top Dexterous Hand Tasks': 'Using the D’Claw robot, rotate a ROBEL valve', 'TableTopFreeDraw-v1': 'to make their own drawing tasks.', 'DrawSVG-v1': 'Instantiates a table with a white canvas on it and a svg path specified with an outline. A robot with a stick is to draw the triangle with a red line.', 'DrawTriangle-v1': 'Instantiates a table with a white canvas on it and a goal triangle with an outline. A robot with a stick is to draw the triangle with a red line.'}

class ManiEnv(MetaEnv):
    def __init__(self, task_id:str="PickCube-v1", obs_mode:str='state', control_mode:str='pd_joint_delta_pos', num_envs:int=1, seed:int=0, max_steps=-1):
        super(ManiEnv, self).__init__()
        self.max_steps = max_steps if max_steps > 0 else np.inf
        self.seed = seed
        self.task_id = task_id
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.env = gym.make(
            id=task_id,
            obs_mode=self.obs_mode,
            control_mode=self.control_mode,
            num_envs=self.num_envs,
        )

    def get_task_prompt(self):
        return TASK_DESCS.get(self.task_id, '')

    def get_random_action(self):
        return self.env.action_space.sample()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return {'obs': obs, 'reward': reward, 'done': terminated, 'truncated': truncated, 'info': info}

    def reset(self, *args, **kwargs):
        obs, _ = self.env.reset(seed=self.seed)
        return obs



