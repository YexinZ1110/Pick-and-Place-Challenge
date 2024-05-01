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

      self.set_point = np.array([[ -0.25713726502200174 , 0.09504443741635296 , -0.2590920067873705 , -2.1524953148900834 , 0.031108568237637652 , 2.2441529661705686 , 0.25088750289490785 ],
[ -0.2544872503569543 , 0.03297907778035211 , -0.25999661311000327 , -2.091158049869565 , 0.00995741737823692 , 2.1230106688163017 , 0.2658299347085468 ],
[ -0.25557210483248627 , -0.010663835630216634 , -0.2558842889840574 , -2.010451545469407 , -0.0029677073497847394 , 2.000136553089767 , 0.27519508743284854 ],
[ -0.2560165764353278 , -0.03523925082391412 , -0.25245198600129726 , -1.9099224657171938 , -0.009225476389465245 , 1.8757904478008844 , 0.279854126356452 ],
[ -0.2554511897671769 , -0.04008324102737134 , -0.25140184253113884 , -1.7880119687474252 , -0.010128798919366843 , 1.749181835425805 , 0.2805399782459023 ],
[ -0.255111398907641 , -0.023655717764447085 , -0.2532722850990789 , -1.6411319026440718 , -0.005933021324009739 , 1.6182330760728487 , 0.2773676807273076 ],
[ -0.2584171328373537 , 0.017845137845794348 , -0.25763775286997254 , -1.4614196642052477 , 0.004566669142845643 , 1.478680205217465 , 0.269806595791745 ],
[ -0.2749487806478152 , 0.09445185420194438 , -0.2608313396274776 , -1.2294101471857872 , 0.02510533106472033 , 1.3207641507036083 , 0.2569461831460255 ],
         
                          # [ -0.14144917546923008 , 0.13106221756602918 , -0.17351063478712656 , -1.821340439888635 , 0.02429279454467291 , 1.950358497331739 , 0.4628988992732331 ],
      # [ -0.14483622081232564 , 0.11320207991998145 , -0.17079061018542888 , -1.7034408475382845 , 0.019786655519115806 , 1.8149670991461944 , 0.4660635899711677 ]        
  
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

      print("There are ",len(H_End), " blocks detected! \n")

      return H_Sorted

   def blockPose(self,H):
      """
      :param H: block pose 4x4
      """
      axis = []
      for i in range(3):
         test = H[0][i]*H[1][i]
         if test < 0.002 and test > -0.002 :
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
