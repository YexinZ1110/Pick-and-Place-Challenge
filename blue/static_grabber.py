import sys
import numpy as np
from copy import deepcopy
from math import pi

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from lib.IK_position_null import IK
from lib.calculateFK import FK

class StaticGrabber():
   def __init__(self, detector, arm, team, ik, fk):
      self.detector = detector
      self.arm = arm
      self.team = team
      self.ik = ik
      self.fk = fk
      self.count = 0
      self.H_ee_camera = detector.get_H_ee_camera()

      self.set_point = np.array([
      [ -0.16097003682152108 , 0.26515757940645446 , -0.1520784952404254 , -2.001213012437274 , 0.05158319488593262 , 2.262800639256027 , 0.4446596833710525 ],
      [ -0.13935156015134775 , 0.1964201426640972 , -0.17559119610373727 , -1.9456625998192407 , 0.040452582424290014 , 2.1387720194572366 , 0.45199989565677234 ],
      [ -0.13891269108791018 , 0.14670670561348462 , -0.17613379276037092 , -1.8659150800310416 , 0.028306696834171346 , 2.010234698503467 , 0.4601643408218827 ],
      [ -0.1427846183733849 , 0.11842974444092638 , -0.1723885513762473 , -1.760564820993247 , 0.021258528277594765 , 1.8771954838928218 , 0.46499997458907394 ]     
        ])
   
   def moveTo(self,q):
      self.arm.safe_move_to_position(q)

   def blockDetect(self):
      """
      Detect blocks and sort them according to there distance to world frame origin.
      return: sorted poses in world frame nx4x4
      """
      # (name, H_camera_block)
      staticBlocks = self.detector.get_detections()
      H_ee_camera = self.detector.get_H_ee_camera()
      _,H = self.fk.forward(self.arm.get_positions())
      H_camera2world = H @ H_ee_camera 
      H_End = [] # poses in world frame
      H_rank = [] # sorted displacement
      for (_, pose) in staticBlocks:
         # block pose in world
         current = H_camera2world @ pose
         H_End.append(current)
         displacement = np.linalg.norm(current[:3,3])
         H_rank.append(displacement)
      sorted = np.argsort(H_rank)
      H_Sorted = []
      for i in range(len(H_End)):
         H_Sorted.append(H_End[sorted[i]])

      # print("There are ",len(H_End), " blocks detected! \n",H_Sorted)

      return H_Sorted

   def blockPose(self,H):
      """
      :param H: block pose 4x4
      """
      axis = []
      for i in range(3):
         test = H[0][i]*H[1][i]
         if test < 0.001 and test > -0.001 :
            continue
         else:
            axis.append([H[0][i],H[1][i],H[2][i]])
      targetAxis = axis[1]
      if axis[0][1] < axis[1][1]:
         targetAxis = axis[0]
      x = targetAxis
      z = np.array([0,0,-1])
      y = np.cross(z,x)

      H_block = np.eye(4)
      H_block[:3,3] = H[:3,3]
      H_block[:3,0] = x
      H_block[:3,1] = y
      H_block[:3,2] = z
      print("H_block is \n",H_block)
      return H_block

   def moveUp(self):
      q = self.arm.get_positions()
      q[1] -= 0.3
      q[3] += 0.3
      q[5] -= 0.3
      self.arm.safe_move_to_position(q)


   def grab(self):
      
      self.arm.exec_gripper_cmd(0.04, 80)
      self.moveUp()
      self.arm.safe_move_to_position(self.set_point[self.count,:])
      self.count += 1
      self.arm.open_gripper()
      self.moveUp()