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
from labs.final.static_grabber import StaticGrabber

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

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    # arm.exec_gripper_cmd(0.04, 80)
    input("\nWaiting for start... Press ENTER to begin!\n") # get set!
    print("Go!\n") # go!
    ik = IK()
    fk = FK()
    static_grabber = StaticGrabber(detector, arm, team, ik, fk)
    # observation
    # T0e=np.array([[1,0,0,0.52],[0,-1,0,0.2],[0,0,-1,0.47],[0,0,0,1]])
    static_standy = [0.14073182017042138 , -0.016202042086755374 , 0.22430339736949562 , -1.722431588772413 , 0.003637551798808322 , 1.706634383411021 , 1.1499123550701502]
    static_grabber.moveTo( static_standy)
    #detect static blocks: sorted poses in world frame nx4x4
    H_Sorted = static_grabber.blockDetect()
    # ee's supposed pose
    ee_poses = []
    for obj_pose in H_Sorted:
        obj_pose = static_grabber.blockPose(obj_pose)
        ee_poses.append(obj_pose)
        # seed = eepose for hot start
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

        
        

    # grabber move to above block
    #for (pose) in ee_pose:
        
        