import sys
import numpy as np
from copy import deepcopy
from math import pi
from time import sleep

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds
from lib.IK_position_null import IK
from lib.calculateFK import FK
from labs.final.blue.static_grabber import StaticGrabber

if __name__ == "__main__":
    try:
        team = rospy.get_param("team") # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()
    # default_state=[-9.83582087e-06, -7.84994017e-01,  2.60828669e-05, -2.35598679e+00,7.83132808e-06,  1.56998697e+00,  7.84997389e-01]
    origin_state=arm.get_positions()
    print("origin_state:",origin_state)
    # start_position = np.array(origin_state)
    # print("start_position:",start_position)
    # arm.safe_move_to_position(start_position) # on your mark!
    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!
    arm.open_gripper()
    
    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    # arm.exec_gripper_cmd(0.04, 80)
    # input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!
    ik = IK()
    fk = FK()
    static_grabber = StaticGrabber(detector, arm, team, ik, fk)
    dynamic_standy=[-0.08542891,  0.1200711 ,-0.11651164,-1.34435869,  0.01400446 , 1.46363719, 0.58579098]
    static_grabber.moveTo( dynamic_standy)

    # dynamic observation
    dynamic_standy=[ -0.12598619584967605 , -0.084194363957534 , -0.11354374627065772 , -1.8120089071472794 , -0.009648593609467578 , 1.7283502238929493 , 0.54778476893618 ]
    static_grabber.moveTo(dynamic_standy)
    H_Sorted_dynamic = static_grabber.blockDetect()
    ee_poses_dynamic = []
    for obj_pose in H_Sorted_dynamic:
        obj_pose = static_grabber.blockPose(obj_pose)
        ee_poses_dynamic.append(obj_pose)

    # static observation
    #static_standy=[ 0.20030833776641954 , 0.059256771640907174 , 0.17570900006170106 , -1.3694812469257056 , -0.010458889457973086 , 1.427835575521786 , 1.1596232927188619 ]
    static_standy=[ 0.33265238,  0.17196839,  0.17188321 ,-1.22208596 ,-0.02974835 , 1.39164772 , 1.28214468]
    static_grabber.moveTo( static_standy)
    H_Sorted = static_grabber.blockDetect()
    ee_poses = []
    for obj_pose in H_Sorted:
        obj_pose = static_grabber.blockPose(obj_pose)
        ee_poses.append(obj_pose)
        # seed = eepose for hot start

    # static grab
    arm.open_gripper()
    for ee_pose in ee_poses:
        ee_pose[2][3] += 0.1
        q, rollout, success, message = ik.inverse( ee_pose, arm.get_positions(), "J_pseudo", 0.5)
        jointPositions, T0e = fk.forward(q)
        static_grabber.moveTo(q)
        # move down to grasp
        ee_pose[2][3] -= 0.1
        q1, rollout, success, message = ik.inverse( ee_pose, q, "J_pseudo", 0.5)
        static_grabber.moveTo(q1)
        print("arm.get_positions(): ",arm.get_positions())
        static_grabber.grab()


    # dynamic grab
    arm.open_gripper()
    dynamic_standy=[-0.08542891,  0.1200711 ,-0.11651164,-1.34435869,  0.01400446 , 1.46363719, 0.58579098]
    static_grabber.moveTo( dynamic_standy)
    for ee_pose in ee_poses_dynamic:
        ee_pose[2][3] += 0.1
        q, rollout, success, message = ik.inverse( ee_pose, arm.get_positions(), "J_pseudo", 0.5)
        jointPositions, T0e = fk.forward(q)
        static_grabber.moveTo(q)
        # move down to grasp
        ee_pose[2][3] -= 0.1
        q1, rollout, success, message = ik.inverse( ee_pose, q, "J_pseudo", 0.5)
        static_grabber.moveTo(q1)
        print("arm.get_positions(): ",arm.get_positions())
        static_grabber.grab()
        dynamic_standy=[-0.08542891,  0.1200711 ,-0.11651164,-1.34435869,  0.01400446 , 1.46363719, 0.58579098]
        static_grabber.moveTo( dynamic_standy)


        
        

    # grabber move to above block
    #for (pose) in ee_pose:
        
        
