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
      self.set_point = np.array([
                                [ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517] ,
                                [ 0.1986,  0.1423,  0.0645, -2.0026, -0.0109,  2.1445,  1.0538] ,
                                [ 0.1755,  0.0966,  0.0882, -1.9295, -0.0095,  2.0257,  1.0528] ,
                                [ 0.1619,  0.0684,  0.102 , -1.8359, -0.0074,  1.904 ,  1.0514] ,
                                [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507] ,
                                [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512] ,
                                [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524] ,
                                [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525] ,
                                ])

   def moveTo(self,q):
      self.arm.safe_move_to_position(q)

   def blockDetect(self):
      staticBlocks = self.detector.get_detections()
      H_ee_camera = self.detector.get_H_ee_camera()
      _,H = self.fk.forward(self.arm.get_positions())
      H_camera2world = H @ H_ee_camera 
      H_End = []
      H_rank = []
      for (_, pose) in staticBlocks:
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