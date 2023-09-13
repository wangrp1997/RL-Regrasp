import pybullet as p
import pybullet_data as pd
import math
import time
import numpy as np
import pybullet_robots.panda.panda_sim as panda_sim
import pybullet_data

p.connect(p.GUI)
p.setGravity(0, 0, -10)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
object_id = p.loadURDF(
    r"./urdf/objects/" + "1_16" + ".urdf", basePosition=[0.7, 0.7, 0.03],
    baseOrientation=p.getQuaternionFromEuler([math.pi, 0, 0]))

p.stepSimulation()
# self._observation = self.getSceneObservation(abs_rel="abs",pos=[0.255,0.005,0.2,1.57,1.57,3.14,0])
# p.setRealTimeSimulation(10)

p.resetDebugVisualizerCamera(cameraDistance=0.3, cameraYaw=120,
                             cameraPitch=-45, cameraTargetPosition=[0.8, 0.8, 0.1])

while 1:
    # p.stepSimulation()
    time.sleep(1 / 240)
    p.getCameraImage(320, 240)
