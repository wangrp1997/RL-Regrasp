import os, inspect
import time

from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import click
import math
import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from itertools import chain

import random
import pybullet_data

from kuka import kuka
from ur5 import ur5
import sys
from scenes import *  # where our loading stuff in functions are held
import threading


viewMatrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=0.3, yaw=90, pitch=-90,
                                                 roll=0, upAxisIndex=2)
projectionMatrix = p.computeProjectionMatrixFOV(fov=120, aspect=1, nearVal=0.01, farVal=10)

image_renderer = p.ER_BULLET_HARDWARE_OPENGL  # if the rendering throws errors, use ER_TINY_RENDERER, but its hella slow cause its cpu not gpu.

train_object_list1 = ['1_00', '1_02', '1_04', '1_06',
                      '1_08', '1_10', '1_12', '1_14',
                      '1_16']

def gripper_camera(obs):
    # Center of mass position and orientation (of link-7)
    pos = obs[-7:-4]
    ori = obs[-4:]  # last 4
    # rotation = list(p.getEulerFromQuaternion(ori))
    # rotation[2] = 0
    # ori = p.getQuaternionFromEuler(rotation)

    rot_matrix = p.getMatrixFromQuaternion(ori)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    # Initial vectors
    init_camera_vector = (1, 0, 0)  # z-axis
    init_up_vector = (0, 1, 0)  # y-axis
    # Rotated vectors
    camera_vector = rot_matrix.dot(init_camera_vector)
    up_vector = rot_matrix.dot(init_up_vector)
    view_matrix_gripper = p.computeViewMatrix(pos, pos + 0.1 * camera_vector, up_vector)
    img = p.getCameraImage(200, 200, view_matrix_gripper, projectionMatrix, shadow=0, flags=p.ER_NO_SEGMENTATION_MASK,
                           renderer=image_renderer)


class graspingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 30  # 控制步数

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=120,
                 isEnableSelfCollision=True,
                 renders=True,
                 arm='ur5',
                 vr=False,
                 mode_str='train'):
        print("init")

        self._timeStep = 1. / 240.
        self._urdfRoot = urdfRoot
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._vr = vr
        self.terminated = 0
        self._p = p
        self.p_bar = tqdm(ncols=0, disable=False)

        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            if self._vr:
                p.resetSimulation()
                # disable rendering during loading makes it much faster
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        else:
            p.connect(p.DIRECT)

        self._arm_str = arm

        p.resetDebugVisualizerCamera(cameraDistance=0.1, cameraYaw=90,
                                         cameraPitch=0, cameraTargetPosition=[0.5, 0, 0.05])
        if self._vr:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            p.setRealTimeSimulation(1)

        self.Lower_limit = -0.08
        self.upper_limit = 0.08
        self.grasp_space = spaces.Box(low=np.array([self.Lower_limit]),
                                       high=np.array([self.upper_limit]),
                                       dtype=np.float32)  # 定义抓取界限

        self.viewer = None
        self.done = False
        self.action_space = spaces.Box(low=np.array([-2]),
                                       high=np.array([2]),
                                       dtype=np.float32)  # 定义动作空间
        self.observation_space = spaces.Box(low=np.array([-3, -2]),
                                       high=np.array([3, 2]),
                                       dtype=np.float32)
        self.model_reset()

    def step_simulation(self):
        """
        Hook p.stepSimulation()
        """

        p.stepSimulation()
        if True:
            time.sleep(self._timeStep)
            self.p_bar.update(1)

    def model_reset(self):
        print("reset")
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        if not self._vr:
            p.setTimeStep(self._timeStep)


        p.setGravity(0, 0, -10)
        table = [
            p.loadURDF((os.path.join(urdfRoot, "table/table.urdf")), 0.0, 0.0, -0.6300, 0.000000, 0.000000, 0.0, 1.0)]

        if self._arm_str == 'rbx1':
            self._arm = rbx1(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
        elif self._arm_str == 'kuka':
            self._arm = kuka(urdfRootPath=self._urdfRoot, timeStep=self._timeStep, vr=self._vr)
        else:
            self._arm = load_arm_dim_up('ur5', dim='Z')

        self._envStepCounter = 0
        p.stepSimulation()
        # self._observation = self.getSceneObservation(abs_rel="abs",pos=[0.255,0.005,0.2,1.57,1.57,3.14,0])
        p.setRealTimeSimulation(1)

    def reset(self,mode='train',obj_id=''):
        # self.mdoel_reset(mode=mode_str,obj_id='004')
        random.shuffle(train_object_list1)
        p.removeAllUserDebugItems()
        if mode == 'train':
            self.object_id = random.choice(train_object_list1)
            self.init_grasp_pose = [0.20, self.grasp_space.sample(), 0.2, 1.57, 1.57, 3.14, 0]
        elif mode == 'test':
            self.object_id = obj_id
            self.init_grasp_pose = [0.20, 0, 0.2, 1.57, 1.57, 3.14, 0]
            if self.object_id in ['1_09', '2_09', '3_09']:
                self.init_grasp_pose = [0.20, 0.015, 0.2, 1.57, 1.57, 3.14, 0]
            if self.object_id in ['1_08', '2_08', '3_08','4_08']:
                self.init_grasp_pose = [0.20, -0.01, 0.2, 1.57, 1.57, 3.14, 0]
        if self.object_id[0] == '1':
            self.step_size = 0.32*0.4
            print('初始步长为0.32*0.4')
        if self.object_id[0] == '2':
            self.step_size = 0.36*0.4
            print('初始步长为0.36*0.4')
            self.Lower_limit = -0.1
            self.upper_limit = 0.1
        if self.object_id[0] == '3':
            self.step_size = 0.4*0.4
            print('初始步长为0.4*0.4')
            self.Lower_limit = -0.12
            self.upper_limit = 0.12
        if self.object_id[0] == '4':
            self.step_size = 0.30*0.4
            print('初始步长为0.3*0.4')
            self.Lower_limit = -0.07
            self.upper_limit = 0.07

        self._seed()

        print("抓取物体的ID:{}".format(self.object_id))
        self.object = throwing_scene(self.object_id)
        self.obj_init_pose = p.getBasePositionAndOrientation(self.object)[0]+\
                             p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.object)[1])
        print("Initial pose of object:" + str(self.obj_init_pose))
        self.step_counter = 0

        self.stable_grasp = False
        self.terminated = False
        self.current_grasp_pos = self.init_grasp_pose
        self.last_grasp_pos = self.current_grasp_pos
        self.last_y0 = self.current_grasp_pos[1]


        obs,s_c = self.getSceneObservation(abs_rel='abs',action=[0])
        self.last_obs = obs
        self.com_pos = p.getBasePositionAndOrientation(self.object)[0]
        p.removeBody(self.object)

        return obs

    def __del__(self):
        p.disconnect()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # moves motors to desired pos
    def step(self, action):
        self.object = throwing_scene(self.object_id)

        self.slip=False

        self._observation, slip_class = self.getSceneObservation(abs_rel='abs',action=action)
        self.step_counter += 1
        print("当前观测值：", self._observation)
        done, self.grasp_state = self._termination()
        reward = self._reward(self._observation)
        print('current reward:', reward)
        info = {'旋转形式': slip_class, '抓取状态': self.grasp_state}
        p.removeBody(self.object)
        self.step_size *= 0.85

        return np.array(self._observation), reward, done, info

    def _termination(self):
        # if self.obj_current_pose[2] > 0.03 and (not self.slip):
        if abs(self.current_grasp_pos[1] - self.com_pos[1]) <= 0.01:
            self.stable_grasp = True
            return True, '稳定抓取'
        elif self.step_counter >= 100:
            return True, '抓取失败'
        else:
            return False, '非稳定抓取'

    def _render(self, mode='human', close=False):
        return

    def _reward(self,observation):

        # use as a RL style reward if you like #################################
        r0 = int(not self.stable_grasp)
        r2 = int(self.stable_grasp)
        d_w = abs(observation[0]) - abs(self.last_obs[0])
        d_h = self.last_obs[1] - observation[1]
        if self.grasp_state == '抓取失败':
            reward = -500
        elif observation[1] > 0.03:
            reward = 30*r0*(-d_w -d_h) + r2*(500) + 100*(abs(self.action_reward))-20*(self.step_counter-1)
            print("高级 奖励")
        else:
            reward = 30*r0*(d_w+d_h) + r2*(500) + 100*(abs(self.action_reward))-20*(self.step_counter-1)
            print("普通 奖励")
        self.last_obs = observation

        return reward

    def get_rotation(self):
        star_time_1 = time.time()
        while True:
            time.sleep(0.01)
            self.angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.object)[1])[0]
            if abs(self.angle) >= (math.pi/2-0.1) and abs(self.angle) <= (math.pi/2+0.1):
                print(f"未发生滑动，此时的旋转角度为:{self.angle},高度为{p.getLinkState(self._arm.uid, 7)[0][2]}")
            else:
                print("到达滑动检测阈值，发生滑动！")
                self.slip = True
                break
            if time.time() - star_time_1 > 3.0 and\
                    abs(self.angle) >= (math.pi/2-0.1) and\
                    abs(self.angle) <= (math.pi/2+0.1):
                print("已超过检测时间，未发生滑动")

                break
        slip_detection_time = time.time() - star_time_1
        self.slip_v = (abs(self.obj_init_pose[3]) -
                       abs(p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.object)[1])[0])) / slip_detection_time
        print("滑动检测所花时间：", slip_detection_time)
        print("检测到的滑动速度：", self.slip_v)

    def move_up(self, new_grasp_pos, control_mode):
        start_time = time.time()
        motor_poses = self._arm.move_to(new_grasp_pos, control_mode, noise=False, clip=False)
        for _ in range(120):  # Wait for a few steps
            p.stepSimulation()
            if self.slip:
                break
            else:
                time.sleep(self._timeStep)
        print("举升所花时间：", time.time() - start_time)

    ##############################################################################################################

    def step_to(self, action, abs_rel='abs', noise=False, clip=False):
        motor_poses = self._arm.move_to(action, abs_rel, noise, clip)
        # for _ in range(120):  # Wait for a few steps
        #     self.step_simulation()
        # print(motor_poses) # these are the angles of the joints.
        for i in range(self._actionRepeat):
            p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._envStepCounter += 1

    def getSceneObservation(self, abs_rel='abs', action=[0]):

        # abs_distance = 1.0
        self._arm.resetJointPoses()
        self.current_grasp_pos[1] = self.current_grasp_pos[1] + self.step_size * action[0]  # 表示固定步长乘上一个动作
        if self.current_grasp_pos[1] > self.upper_limit:
            self.current_grasp_pos[1] = self.upper_limit
        if self.current_grasp_pos[1] < self.Lower_limit:
            self.current_grasp_pos[1] = self.Lower_limit

        self.last_grasp_pos = self.current_grasp_pos
        self.new_grasp_pos = self.current_grasp_pos[0:3] +\
                             list(p.getQuaternionFromEuler(self.current_grasp_pos[3:6])) +\
                             [self.current_grasp_pos[6]]
        self.step_to(self.new_grasp_pos, abs_rel)
        # obs = self.getSceneObservation()
        self.new_grasp_pos[2] = 0.15
        self.step_to(self.new_grasp_pos, abs_rel)

        y0 = p.getLinkState(self._arm.uid, 7)[0][1]
        print("当前抓取位置", y0)
        self.action_reward = abs(y0 - self.last_y0)  # 移动位置的奖励
        print("移动位置的奖励：", self.action_reward*100)
        self.last_y0 = y0
        print("上一次抓取位置",self.last_y0)

        self.new_grasp_pos[7] = 2.0
        # action = motorsIds
        # action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        self.step_to(self.new_grasp_pos, abs_rel)

        self.new_grasp_pos[2] = 0.19
        h0 = p.getLinkState(self._arm.uid, 7)[0][2]
        print("初始末端的高度", h0)
        t1 = threading.Thread(target=self.move_up, args=(self.new_grasp_pos, abs_rel,))
        t2 = threading.Thread(target=self.get_rotation)

        t1.start()
        t2.start()
        t1.join()
        t2.join()
        dh = p.getLinkState(self._arm.uid, 7)[0][2] - h0
        print("末端被提起的高度", dh)

        start = list(p.getLinkState(self._arm.uid, 7)[4])
        rot = np.array(p.getMatrixFromQuaternion(p.getLinkState(self._arm.uid, 7)[5])).reshape(3, 3)
        add_local = np.array([1, 0, 0]).reshape(3, 1)
        add_world = np.dot(rot, add_local).tolist()
        add_world_one = [one for token in add_world for one in token]
        end = [a + b for a, b in zip(start, add_world_one)]
        p.addUserDebugLine(start, [start[0],start[1],start[2]-0.14], [0, 0, 0])


        self.obj_current_pose = p.getBasePositionAndOrientation(self.object)[0] + \
                             p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.object)[1])
        print("当前物体的位姿为:" + str(self.obj_current_pose))
        if (self.obj_current_pose[3] - self.obj_init_pose[3]) > 0:
            print("旋转方向为逆时针")
            direction = 1
            s_c = '逆时针'
        elif (self.obj_current_pose[3] - self.obj_init_pose[3]) < 0:
            print("旋转方向为顺时针")
            direction = -1
            s_c = '顺时针'

        time.sleep(2)
        self.new_grasp_pos[2] = 0.15
        # action = motorsIds
        # action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        self.step_to(self.new_grasp_pos, abs_rel)

        self.new_grasp_pos[7] = 0.0
        # action = motorsIds
        # action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        self.step_to(self.new_grasp_pos, abs_rel)

        self.new_grasp_pos[2] = 0.19
        # action = motorsIds
        # action = action[0:3] + list(p.getQuaternionFromEuler(action[3:6])) + [action[6]]
        self.step_to(self.new_grasp_pos, abs_rel)

        # return [self.slip_v**3, 1/(dh**2)], s_c  ##进行数据的映射
        return [self.slip_v, dh], s_c  ##进行数据的映射


##############################################################################################################

def setup_controllable_camera(environment):
    environment._p.addUserDebugParameter("Camera Zoom", -15, 15, 1.674)
    environment._p.addUserDebugParameter("Camera Pan", -360, 360, 70)
    environment._p.addUserDebugParameter("Camera Tilt", -360, 360, -50.8)
    environment._p.addUserDebugParameter("Camera X", -10, 10, 0)
    environment._p.addUserDebugParameter("Camera Y", -10, 10, 0)
    environment._p.addUserDebugParameter("Camera Z", -10, 10, 0)


def setup_controllable_motors(environment, arm):
    possible_range = 3.2  # some seem to go to 3, 2.5 is a good rule of thumb to limit range.
    motorsIds = []

    for tests in range(0, environment._arm.numJoints):  # motors

        jointInfo = p.getJointInfo(environment._arm.uid, tests)
        # print(jointInfo)
        qIndex = jointInfo[3]

        if arm == 'kuka':
            if qIndex > -1 and jointInfo[0] != 7:
                motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                                      -possible_range,
                                                                      possible_range,
                                                                      0.0))
        else:
            motorsIds.append(environment._p.addUserDebugParameter("Motor" + str(tests),
                                                                  -possible_range,
                                                                  possible_range,
                                                                  0.0))

    return motorsIds


def update_camera(environment):
    if environment._renders:
        # Lets reserve the first 6 user debug params for the camera
        p.resetDebugVisualizerCamera(environment._p.readUserDebugParameter(0),
                                     environment._p.readUserDebugParameter(1),
                                     environment._p.readUserDebugParameter(2),
                                     [environment._p.readUserDebugParameter(3),
                                      environment._p.readUserDebugParameter(4),
                                      environment._p.readUserDebugParameter(5)])


def send_commands_to_motor(environment, motorIds):
    done = False

    while (not done):
        action = []

        for motorId in motorIds:
            action.append(environment._p.readUserDebugParameter(motorId))
        print(action)
        state, reward, done, info = environment.step(action)
        obs = environment.getSceneObservation()
        update_camera(environment)

    environment.terminated = 1


def control_individual_motors(environment, arm):
    motorIds = setup_controllable_motors(environment, arm)
    send_commands_to_motor(environment, motorIds)


###################################################################################################
def make_dir(string):
    try:
        os.makedirs(string)
    except FileExistsError:
        pass  # directory already exists


#####################################################################################

def str_to_bool(string):
    if str(string).lower() == "true":
        string = True
    elif str(string).lower() == "false":
        string = False

    return string


def env_test(mode, arm, render):
    print(arm)

    env = graspingEnv(renders=str_to_bool(render), arm=arm)

    if env._renders:
        setup_controllable_camera(env)

    if mode == 'xyz':
        y_ = 0.075
        env.reset(mode='test')
        for j in range(9):
            action = [-0.5]
            obs, reward, done, info = env.step(action=action)
            print(info)
            print('current reward:', reward)
            # y_ -= 0.02
    else:
        env._arm.active = True
        control_individual_motors(env, arm)


def train(mode, arm, render):
    from spinup.utils.run_utils import setup_logger_kwargs
    # from ddpg.ddpg import ddpg
    # import ddpg.core as core
    from td3.td3 import td3
    import td3.core as core
    # from sac.sac import sac
    # import sac.core as core

    # env = graspingEnv(renders=str_to_bool(render), arm=arm)
    #
    # if env._renders:
    #     setup_controllable_camera(env)

    logger_kwargs = setup_logger_kwargs('td3-ur5-ComGrasp', 0, data_dir=r'./logs/')

    td3(graspingEnv, actor_critic=core.MLPActorCritic, ac_kwargs=dict(hidden_sizes=[16,16]), seed=0,
        steps_per_epoch=30, epochs=200, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=75,
        update_after=30, update_every=30, act_noise=0.1, target_noise=0.2,
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=5,
        logger_kwargs=logger_kwargs, save_freq=1)

def evaluate():
    import torch
    from spinup.utils.run_utils import setup_logger_kwargs
    from td3.td3 import td3
    import td3.core as core

    env = graspingEnv(renders=True, arm='ur5')
    ac = core.MLPActorCritic(env.observation_space, env.action_space, **dict(hidden_sizes=[16,16]))
    checkpoint = torch.load(r".\logs\td3-ur5-ComGrasp\td3-ur5-ComGrasp_s0\pyt_save\model.pt")
    ac.load_state_dict(checkpoint['model'])
    print('ac={}'.format(ac))

    temp = ['1_01','1_03','1_05','1_07','1_09','1_11','1_13', '1_15', '1_17']
    for test_obj in temp:
        sum_reward = 0
        success = 0
        step_counter = {}
        for i in range(20):
            print('共20次探索，第{}次探索'.format(i + 1))
            while True:
                obs = env.reset(mode='test',obj_id=test_obj)  # 020： 100%→100%  991： 100%→100% 604：100%  007 100%
                if env.slip:
                    print('初始化成功！')
                    break
                else:
                    print('初始化失败，重新初始化中···')
                    p.removeBody(env.object)
            for step in range(100):
                actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
                print("td3 output origin action:", actions)
                actions[0] = -1*(obs[0]/abs(obs[0])) * abs(actions[0])
                obs, reward, done, info = env.step(actions)
                sum_reward += reward
                if done:
                    if env.stable_grasp:
                        print('抓取成功')
                        success += 1
                        step_counter[i] = ["成功", step + 1]
                        success_rate = (success / 20) * 100
                        print('成功率：{}%'.format((success / 20) * 100))
                    else:
                        print('抓取失败')
                        step_counter[i] = ["失败"]
                        print('成功率：{}%'.format((success / 20) * 100))
                    break
            p.removeBody(env.object)

        print('sum reward={}'.format(sum_reward))
        print(step_counter)
        steps_list = []
        for key, value in step_counter.items():
            if value[0] == '成功':
                steps_list.append(value[1])
        print(np.array(steps_list))
        print("平均需要次数：", np.mean(steps_list))
        np.savetxt(rf'./results/{test_obj}.txt', steps_list)
        with open(rf'./results/{test_obj}.txt', 'a') as f:
            f.write('成功率:' + str(success_rate))
            f.write('\n')
            f.write('平均尝试次数：' + str(np.mean(steps_list)))


def real_exp():
    import ppo.core as core
    import subprocess
    import torch
    from ur_control import ur5 as ur

    success_counter = 0
    step_counter_list = []
    object_name = input("请输入实验物品名称：")
    env = graspingEnv(renders=False, arm='ur5')
    for i in range(20):
        step_counter = 0
        wait = input('调整夹爪完毕请按回车')
        dv = 0.05
        while True:
            subprocess.call(
                '"D:\Anaconda3\envs\py36_32bit\python" E:\\Renpeng\\三种模态对比最终版本\\最最终版本\\real_experiment\\main_tt.py')

            ac = core.MLPActorCritic(env.observation_space, env.action_space, **dict(hidden_sizes=[16] * 2))
            checkpoint = torch.load(r".\logs\ppo-ur5-ComGrasp\ppo-ur5-ComGrasp_s0\pyt_save\model.pt")
            ac.load_state_dict(checkpoint['model'])

            obs = np.loadtxt(r'E:\Renpeng\三种模态对比最终版本\最最终版本\obs.txt',dtype=np.float32)
            # obs[1] = obs[1] * 2
            print(obs)
            if obs[4] == 1:
                print("抓取成功")
                success_counter += 1
                print(f'当前成功率{(success_counter / 20) * 100}%')
                step_counter_list.append(step_counter)
                print(step_counter_list)
                break
            # if step_counter == 0:
            #     w0 = obs[0]
            #     actions = ac.step(torch.as_tensor(obs[0:-1], dtype=torch.float32))
            # else:
            #     if abs(obs[0]) >= abs(w0):
            #         w1 = (abs(obs[0]) / obs[0]) * ((abs(w0) / (abs(w0) + abs(obs[0]))) * abs(w0))
            #         obs[0] = w1
            #     else:
            #         #     w1 = (abs(obs[0])/obs[0])*(((abs(w0) + abs(obs[0])*abs(w0))/abs(obs[0])))
            #         #     obs[0] = w1
            #         w0 = obs[0]
            #     print("处理后的观测值为：", obs)
            #     actions = ac.step(torch.as_tensor(obs[0:-1], dtype=torch.float32))
            actions = ac.step(torch.as_tensor(obs[0:2], dtype=torch.float32))
            dv *= 0.8
            while True:
                move_dist = dv * actions[0]
                a = input(f"是否移动{move_dist}？")
                if a == ' ':
                    tcp = ur.get_current_tcp()
                    tcp[0] += move_dist
                    ur.move_to_tcp(tcp)
                    break
                if a == 'n':
                    break

            step_counter += 1
            if step_counter > 6 or a == 'n':
                print("抓取失败")
                break

    print(f'成功率：{success_counter / 20 * 100}%')
    print(f"抓取成功平均尝试次数{np.mean(step_counter_list)}")
    with open(r"./results/{}.txt".format(object_name), 'a') as f:
        f.write(f"实验物品{object_name}\n")
        f.write(str(step_counter_list))
        f.write("\n")
        f.write((f'成功率：{success_counter / 20 * 100}%\n'))
        f.write(f"抓取成功平均尝试次数{np.mean(step_counter_list)}")
        f.write('\n')



@click.command()
@click.option('--mode', type=str, default='xyz',
              help='motor: control individual motors, xyz: control xyz/rpw of gripper, demos: collect automated demos')
# @click.option('--abs_rel', type=str, default='abs',
#               help='absolute or relative positioning, abs doesnt really work with rbx1 yet')
@click.option('--arm', type=str, default='ur5', help='rbx1 or kuka')
@click.option('--render', type=bool, default=True, help='rendering')
def main(**kwargs):

    # env_test(**kwargs)
    # train(*kwargs)
    evaluate()
    # real_exp()

if __name__ == "__main__":
    main()

