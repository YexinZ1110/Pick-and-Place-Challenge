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

    start_position = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    arm.safe_move_to_position(start_position) # on your mark!

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

    # STUDENT CODE HERE
    # arm.exec_gripper_cmd(0.04, 80)

    # dynamic observation
    static_grabber.moveTo( [-0.20638,  0.03282, -0.21015, -1.71503 , 0.00695,  1.74713,  0.36776])
    H_Sorted_dynamic_1 = static_grabber.blockDetect()
    H_Sorted_dynamic = []
    for (pose) in H_Sorted_dynamic_1:
        pose = static_grabber.blockPose(pose)
        H_Sorted_dynamic.append(pose)
        # seed = eepose for hot start

    # static observation
    static_grabber.moveTo( [-0.20638,  0.03282, -0.21015, -1.71503 , 0.00695,  1.74713,  0.36776])
    H_Sorted_1 = static_grabber.blockDetect()
    H_Sorted = []
    for (pose) in H_Sorted_1:
        pose = static_grabber.blockPose(pose)
        H_Sorted.append(pose)
        # seed = eepose for hot start
    
    # static grabber
    arm.open_gripper()
    for (pose) in H_Sorted:
        pose[2][3] += 0.1
        q, rollout, success, message = ik.inverse( pose, arm.get_positions(), "J_pseudo", 0.5)
        jointPositions, T0e = fk.forward(q)
        static_grabber.moveTo(q)
        # move down to grasp
        pose[2][3] -= 0.1
        q1, rollout, success, message = ik.inverse( pose, q, "J_pseudo", 0.5)
        static_grabber.moveTo(q1)
        static_grabber.grab()
        
    # dynamic grabber
    arm.open_gripper()
    for (pose) in H_Sorted_dynamic:
        pose[2][3] += 0.1
        q, rollout, success, message = ik.inverse( pose, arm.get_positions(), "J_pseudo", 0.5)
        jointPositions, T0e = fk.forward(q)
        static_grabber.moveTo(q)
        # move down to grasp
        pose[2][3] -= 0.1
        q1, rollout, success, message = ik.inverse( pose, q, "J_pseudo", 0.5)
        static_grabber.moveTo(q1)
        static_grabber.grab()

    # grabber move to above block
    #for (pose) in H_Sorted:
        
        
