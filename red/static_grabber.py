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
      # self.H_ee_camera = detector.get_H_ee_camera()
      self.set_point = np.array([[ 0.19025 , 0.31056 , -0.03039 , -1.90188 , 0.01161 , 2.21227 , 0.93976 ],
[ 0.18041 , 0.24891 , -0.01995 , -1.84123 , 0.00568 , 2.09009 , 0.94366 ],
[ 0.1654 , 0.20995 , -0.00371 , -1.76426 , 0.00086 , 1.97421 , 0.94684 ],
[ 0.15387 , 0.1888 , 0.00894 , -1.66473 , -0.00173 , 1.85352 , 0.94854 ],
[ 0.1493 , 0.18752 , 0.01425 , -1.53908 , -0.00267 , 1.72658 , 0.94911 ],
[ 0.15224 , 0.21016 , 0.01142 , -1.37971 , -0.00237 , 1.58986 , 0.94885 ],
[ 0.1633 , 0.26678 , -0.00166 , -1.16767 , 0.00046 , 1.43445 , 0.94716 ],
[ 0.17898 , 0.39971 , -0.02792 , -0.82469 , 0.01157 , 1.22429 , 0.94259 ]])

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
      # print("H_block is \n",H_block)
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