
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
import sys
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from itertools import chain
from collections import deque

import random
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
import time
import itertools
from cntrl import *
from pyRobotiqGripper import *
real = False

def setup_sisbot(p, uid):
    # controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
    #                  "elbow_joint", "wrist_1_joint",
    #                  "wrist_2_joint", "wrist_3_joint",
    #                  "robotiq_85_left_knuckle_joint"]
    controlJoints = ["shoulder_pan_joint","shoulder_lift_joint",
                     "elbow_joint", "wrist_1_joint",
                     "wrist_2_joint", "wrist_3_joint", 'left_gripper_motor', 'right_gripper_motor']

    jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
    numJoints = p.getNumJoints(uid)
    jointInfo = namedtuple("jointInfo", 
                           ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity","controllable"])
    joints = AttrDict()
    for i in range(numJoints):
        info = p.getJointInfo(uid, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        controllable = True if jointName in controlJoints else False
        info = jointInfo(jointID,jointName,jointType,jointLowerLimit,
                         jointUpperLimit,jointMaxForce,jointMaxVelocity,controllable)
        if info.type=="REVOLUTE": # set revolute joint to static
            p.setJointMotorControl2(uid, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        joints[info.name] = info
    controlRobotiqC2 = False
    mimicParentName = False
    return joints, controlRobotiqC2, controlJoints, mimicParentName



class ur5:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01, vr = False):



        self.robotUrdfPath = "./urdf/real_arm.urdf"
        # self.robotUrdfPath = "./urdf/ur5.urdf"
        self.robotStartPos = [0,0,0]
        self.robotStartOrn = p.getQuaternionFromEuler([1.885,1.786,0.132])
        # self.robotStartOrn = p.getQuaternionFromEuler([0,0,0])

        self.xin = self.robotStartPos[0]
        self.yin = self.robotStartPos[1]

        self.zin = self.robotStartPos[2]
        self.lastJointAngle = None
        self.active = False

        if real:
            self.s = init_socket()

            if True:
                self.grip=RobotiqGripper("COM8")
                #grip.resetActivate()
                self.grip.reset()
                #grip.printInfo()
                self.grip.activate()
                #grip.printInfo()
                #grip.calibrate()




        self.reset()
        self.timeout = 0


    def reset(self):

        print("----------------------------------------")
        print("Loading robot from {}".format(self.robotUrdfPath))
        self.uid = p.loadURDF(self.robotUrdfPath, self.robotStartPos, self.robotStartOrn,
                             flags=p.URDF_USE_INERTIA_FROM_FILE)
        for i in range(1,7):
            p.enableJointForceTorqueSensor(self.uid, jointIndex=i, enableSensor=1)
        self.joints, self.controlRobotiqC2, self.controlJoints, self.mimicParentName = setup_sisbot(p, self.uid)
        self.endEffectorIndex = 7 # ee_link
        self.numJoints = p.getNumJoints(self.uid)
        self.active_joint_ids = []
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            self.active_joint_ids.append(joint.id)
        c = p.createConstraint(self.uid,
                               8,
                               self.uid,
                               9,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=1, erp=0.1, maxForce=50)





    def getActionDimension(self):
        # if (self.useInverseKinematics):
        #     return len(self.motorIndices)
        return 8  # position x,y,z and ori quat and finger angle
    def getObservationDimension(self):
        return len(self.getObservation())

    def setPosition(self, pos, quat):

        p.resetBasePositionAndOrientation(self.uid,pos,
                                          quat)

    def resetJointPoses(self):
                # move to this ideal init point
        self.active = False
        self.reset_pose = [0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897,
                           1.5824233979484994, 0.19581299859677043,0.012000000476837159, -0.012000000476837159]
        for i in range(0,50000):
            self.action([0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994, 0.19581299859677043, 0.012000000476837159, -0.012000000476837159])
        self.active = True

        if real:
             

             movej(self.s,[0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994, 0.19581299859677043, 0.012000000476837159, -0.012000000476837159],a=0.01,v=0.05,t=10.0)
             time.sleep(10)

        self.lastJointAngle = [0.15328961509984124, -1.8, -1.5820032364177563, -1.2879050862601897, 1.5824233979484994, 0.19581299859677043]




    def getObservation(self):
        observation = []
        state = p.getLinkState(self.uid, self.endEffectorIndex, computeLinkVelocity = 1)
       #print('state',state)
        pos = state[0]
        orn = state[1]
        


        observation.extend(list(pos))
        observation.extend(list(orn))

        joint_states = p.getJointStates(self.uid, self.active_joint_ids)
        
        joint_positions = list()
        joint_velocities = list()
        

        for joint in joint_states:
            
            joint_positions.append(joint[0])
            joint_velocities.append(joint[1])
            

  
        return joint_positions + joint_velocities + observation


    def action(self, motorCommands):
        #print(motorCommands)
        
        poses = []
        indexes = []
        forces = []


        # if self.lastJointAngle == None:
        #     self.lastJointAngle =  motorCommands[0:6]

        # rel_a = np.array(motorCommands[0:6]) - np.array(self.lastJointAngle) 

        # clipped_a = np.clip(rel_a, -0.1, 0.1)
        # motorCommands[0:6] = list(clipped_a+self.lastJointAngle)
        # self.lastJointAngle =  motorCommands[0:6]

        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]

            poses.append(motorCommands[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)
        l = len(poses)


        # p.setJointMotorControlArray(self.uid, indexes, p.POSITION_CONTROL, targetPositions=poses, targetVelocities =[0]*l, positionGains = [0.03]*l, forces = forces)
        #holy shit this is so much faster in arrayform!
        p.setJointMotorControlArray(self.uid, indexes, p.POSITION_CONTROL, targetPositions=poses, targetVelocities =[0.0]*l, positionGains = [0.03]*l, forces=forces)


        if real and self.active:
            
            if time.time() > self.timeout+0.05:
                servoj(self.s,poses[0:6],a=0,v=0,t=0.05, gain = 100, lookahead_time = 0.05)
                self.timeout = time.time()


                grip_angle =  max(0, min(255,int(poses[6]*255/0.04)))  # range 0 - 0.04
                self.grip.goTo(grip_angle)


    def move_to(self, position_delta, mode = 'abs', noise = False, clip = False):

        #at the moment UR5 only absolute

        x = position_delta[0]
        y = position_delta[1]
        z = position_delta[2]
        
        orn = position_delta[3:7]
        finger_angle = position_delta[7]

        # define our limtis. 
        z = max(0.14, min(0.7,z))
        x = max(-0.25, min(0.3,x))
        y =max(-0.4, min(0.4,y))

        jointPose = list(p.calculateInverseKinematics(self.uid,
                                                      self.endEffectorIndex,
                                                      [x, y, z], orn, restPoses=self.reset_pose, maxNumIterations=100
                                                      ))

        # print(jointPose)
        # print(self.getObservation()[:len(self.controlJoints)]) ## get the current joint positions
        
        jointPose[7] = -finger_angle/25 
        jointPose[6] = finger_angle/25
        
        self.action(jointPose)
        #print(jointPose)
        return jointPose

    def get_FT(self):
        return p.getJointState(self.uid, 6)

    def get_joints_FT(self):

        rotation_axes = [
            np.array([0, 0, 0, 0, 0, 1]),  # shoulder_pan_joint
            np.array([0, 0, 0, 0, 1, 0]),  # shoulder_lift_joint
            np.array([0, 0, 0, 0, 1, 0]),  # elbow_joint
            np.array([0, 0, 0, 0, 1, 0]),  # wrist_1_joint
            np.array([0, 0, 0, 0, 0, 1]),  # wrist_2_joint
            np.array([0, 0, 0, 0, 1, 0]),  # wrist_3_joint
        ]
        joint_states = p.getJointStates(self.uid, range(1, 7))
        joint_T = []

        for joint_index, state in enumerate(joint_states):
            # joint_name = p.getJointInfo(self.uid, joint_index+1)[1].decode("utf-8")  # 关节名称
            joint_force = state[2]  # 关节的力（扭矩）信息
            # print(f"Joint: {joint_name}, Joint Torque: {rotation_axes[joint_index]@ np.array(joint_force).reshape(6, 1)}")
            joint_T.append((rotation_axes[joint_index] @ np.array(joint_force).reshape(6, 1))[0])

        return [np.array([0, 0, item]) for item in joint_T]

