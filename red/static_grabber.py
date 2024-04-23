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
                              [ 0.20149,  0.16689,  0.06193, -2.03366, -0.01272,  2.2002,   1.05545],
                              [ 0.19154,  0.09093,  0.07196, -1.92797, -0.00724,  2.01865,  1.05174] ,
                              [ 0.18033,  0.05624,  0.08327, -1.76436, -0.00481,  1.8204,   1.05005] ,
                              [ 0.17466,  0.06443,  0.09028, -1.59918,-0.00583,  1.66357,  1.05069] ,
                              [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507] ,
                              [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512] ,
                              [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524] ,
                              [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525] ,
                                ])
   # [ 0.20149  0.16689  0.06193 -2.03366 -0.01272  2.2002   1.05545]
   # [ 0.19154  0.09093  0.07196 -1.92797 -0.00724  2.01865  1.05174]
   # [ 0.18033  0.05624  0.08327 -1.76436 -0.00481  1.8204   1.05005]
   # [ 0.17466  0.06443  0.09028 -1.59918 -0.00583  1.66357  1.05069]
   def moveTo(self,q):
      self.arm.safe_move_to_position(q)

   def blockDetect(self):
      staticBlocks = self.detector.get_detections()
      H_ee_camera = self.detector.get_H_ee_camera()
      print("H_block is \n",staticBlocks)
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
      targetAxis = axis[0]
      if np.abs(axis[1][1]) < np.abs(axis[0][1]):
         targetAxis = axis[1]
      x = targetAxis
      if targetAxis[0] < 0 :
         x[0] = x[0] * -1
         x[1] = x[1] * -1
         x[2] = x[2] * -1
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
