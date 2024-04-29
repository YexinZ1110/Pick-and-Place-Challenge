import sys
import numpy as np
from copy import deepcopy
from math import pi, sin, cos

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

    # get the transform from camera to panda_end_effector
# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

from lib.IK_position_null import IK
from lib.calculateFK import FK

from scipy.spatial.transform import Rotation as R

from labs.final.static_grabber import StaticGrabber



if __name__ == "__main__":
    # try:
    #     team = rospy.get_param("team") # 'red' or 'blue'
    #     simorreal = rospy.get_param("sr") # sim or real
    # except KeyError:
    #     print('Team must be red or blue - make sure you are running final.launch!')
    #     exit()
    team="red"
    simorreal="sim"
    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()
    ik = IK(max_steps=1000)
    fk = FK()
    static_grabber = StaticGrabber(detector, arm, team, ik, fk)

    # initialize the environment
    class Env():
        def __init__(self):
            self.radius = .305
            self.height = .2
            self.center_red_static = np.array([.562, -1.159, self.height])
            self.center_blue_static = np.array([.562, 1.159, self.height])
            self.center_red = np.array([.562, -.82, self.height])
            self.center_blue = np.array([.562, .82, self.height])
            self.platform_width = .25
            self.robot_base_red = np.array([0.0, -.99, 0.0])
            self.robot_base_blue = np.array([0.0, .99, 0.0])
            if simorreal == 'real':
                self.turntable_angular_speed = np.pi/50. # real
            else:
                self.turntable_angular_speed = np.pi/100. # sim
    env = Env()

    start_time = time_in_seconds()
    ts = [(time_in_seconds() - start_time, 'start')]
    def blockDetect():
      """
      Detect blocks and sort them according to there distance to world frame origin.
      return: sorted poses in world frame nx4x4
      """
      # (name, H_camera_block)
      staticBlocks = detector.get_detections()
      H_ee_camera = detector.get_H_ee_camera()
      _,H = fk.forward(arm.get_positions())
      H_camera2world = H @ H_ee_camera 
      H_End = [] # poses in world frame
      H_rank = [] # sorted displacement
      for (_, pose) in staticBlocks:
         # block pose in world
         current = H_camera2world @ pose
         H_End.append(current)
        #  displacement = np.linalg.norm(pose[:3,3])
         displacement = pose[0, 3]
         H_rank.append(displacement)
      sorted = np.argsort(H_rank)
      H_Sorted = []
      for i in range(len(H_End)):
         H_Sorted.append(H_End[sorted[i]])

      # print("There are ",len(H_End), " blocks detected! \n",H_Sorted)

      return H_Sorted
    
    def blockPose(H):
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

    def get_poses():
        H_Sorted_1 = blockDetect()
        # ee's supposed pose
        H_Sorted = []
        for (pose) in H_Sorted_1:
            pose = blockPose(pose)
            H_Sorted.append(pose)
            # seed = eepose for hot start
        return H_Sorted
    
    def get_poses_many_times(max_itr = 10, interval = 1):
        '''
        keep detecting blocks position
        if not detected, detect until reaching max_itr

        interval - time between each detection
        '''
        poses = []
        i = 0
        while i < max_itr:
            poses = get_poses()
            if len(poses) > 0:
                break
            i += 1
            rospy.sleep(interval)
        return poses
    
    def select_block(poses):
        index = 0 if team == 'red' else -1
        return poses[index]
    
    def predict_pose(advance_time, pose):
        # predict rotation
        vec_ang = np.array([0., 0., 1.]) * advance_time * env.turntable_angular_speed
        rotation_predict = R.from_rotvec(vec_ang)
        # print('rotation predicct:\n', rotation_predict.as_matrix())
        rotation_base = R.from_matrix(pose[:3, :3])
        # print('rotation_base:\n', rotation_base.as_matrix())
        rotation = rotation_predict * rotation_base
        # print('rotation:\n', rotation.as_matrix())

        # predict center
        center = rotation_predict.apply(pose[:3, 3])
        center[2] += 0.05
        # print('center:\n', center)

        # get gripper rotation
        pose_x = rotation.as_matrix()[:, 0]
        pose_y = rotation.as_matrix()[:, 1]
        radius_direction = np.array([-center[0], -center[1], 0.])
        if abs(np.dot(radius_direction, pose_x)) > abs(np.dot(radius_direction, pose_y)):
            y = pose_x
        else:
            y = pose_y
        # z = np.array([center[1], -center[0], -np.linalg.norm(center[:2])])
        if np.dot(radius_direction, y) < 0:
            y = -y
        
        z = np.array([0., 0., -1.])
        x = np.cross(y, z)
        x = x/np.linalg.norm(x)
        y = y/np.linalg.norm(y)
        z = z/np.linalg.norm(z)
        rotation_matrix = np.hstack((x.reshape(3,1), y.reshape(3,1), z.reshape(3,1)))

        pose = np.hstack((rotation_matrix, center.reshape(3,1)))
        pose = np.vstack((pose, np.array([0, 0, 0, 1])))
        print('predicted pose:\n', pose)
        return pose

    def get_grip_configurations():
        ts.append((time_in_seconds() - start_time, 'ready to select block'))
        # poses = get_poses()
        poses = get_poses_many_times()
        if len(poses) == 0:
            rospy.logwarn("No block detected!!!!!!!")
            return None, None
        selected_pose = select_block(poses)
        selected_pose_in_world = selected_pose
        selected_pose_in_world[:3, 3] += env.robot_base_red
        # print('pose to be gripped:\n', selected_pose_in_world)
        ts.append((time_in_seconds() - start_time, 'block selected'))

        predicted_pose = predict_pose(advance_time, selected_pose_in_world)
        predicted_pose[:3, 3] -= env.robot_base_red
        ts.append((time_in_seconds() - start_time, 'pose predicted'))
        
        # predicted_configuration = get_configuration(predicted_pose)
        # ts.append((time_in_seconds() - start_time, 'predicted configuration found'))

        pose_to_grip = predicted_pose
        pose_to_grip[2, 3] -= 0.05
        grip_configuration, success = get_configuration(pose_to_grip)
        cnt = 0
        while cnt < 5 and not success:
            poses = get_poses_many_times()
            if len(poses) == 0:
                rospy.logwarn("No block detected!!!!!!!")
                return None, None
            selected_pose = select_block(poses)
            selected_pose_in_world = selected_pose
            selected_pose_in_world[:3, 3] += env.robot_base_red
            # print('pose to be gripped:\n', selected_pose_in_world)
            ts.append((time_in_seconds() - start_time, 'block selected'))

            predicted_pose = predict_pose(advance_time, selected_pose_in_world)
            predicted_pose[:3, 3] -= env.robot_base_red
            ts.append((time_in_seconds() - start_time, 'pose predicted'))
            
            # predicted_configuration = get_configuration(predicted_pose)
            # ts.append((time_in_seconds() - start_time, 'predicted configuration found'))

            pose_to_grip = predicted_pose
            pose_to_grip[2, 3] -= 0.05
            grip_configuration, success = get_configuration(pose_to_grip)
            cnt += 1
        ts.append((time_in_seconds() - start_time, 'grip configuration found'))

        # return predicted_configuration, grip_configuration
        return grip_configuration, success
    
    def grip():
        for i in range(5):
            rospy.sleep(2)
            arm.exec_gripper_cmd(0.03, 80)
            gripper_state = arm.get_gripper_state()
            gripper_positions = gripper_state['position']
            gripper_forces = gripper_state['force']
            gripper_distance = gripper_positions[0] + gripper_positions[1]
            # print('gripper force:', gripper_forces)
            # if gripper_distance < 0.04 or gripper_forces[0] < 1 or gripper_forces[1] < 1:
            if gripper_distance < 0.04:
                print("current diatance:", gripper_distance)
                print("nothing grabbed for " + str(i) + " times\n")
                arm.open_gripper()
                continue
            else:
                print("current diatance:", gripper_distance)
                print("successfully grabbed one block")
                break
    
    # def grip(configuration):
    #     # arm.safe_move_to_position(configuration)
    #     # ts.append((time_in_seconds() - start_time, 'moved to grip configuration'))
    #     # arm.close_gripper()
    #     arm.exec_gripper_cmd(0.04, 80)



    # seed = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # seed = current configuration
    seed = arm.get_positions()

    # position = np.array([0., -env.radius, env.height + 0.02])
    # position = env.center_red + np.array([-env.platform_width/2 + 0.03, -env.platform_width/2 + 0.03, 0.1])
    
    # BLUE standby position 
    # position = np.array([2.82641223e-09, -7.35000000e-01, 3.99999997e-01])
    # standby red position [[ 2.44244176e-10  1.00000000e+00  2.78591434e-09  2.82641223e-09]
        # [ 1.00000000e+00 -2.44244016e-10 -3.60451028e-09  7.35000000e-01]
        # [-3.60451035e-09  2.78591439e-09 -1.00000000e+00  3.99999997e-01]
        # [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]

    # position = env.center_red + np.array([-env.platform_width/2 + 0.03, 0., 0.03]) ############ to be changed, maybe higher
    # position = env.center_red + np.array([0.0, 0.0, 0.03])
    # position = np.array([(env.radius + 0.05)/np.sqrt(2), -(env.radius + 0.05)/np.sqrt(2), env.height + 0.1])
    # position = np.array([0.0, -env.radius + 0.05, env.height + 0.2])
    # position = np.array([0.0, env.radius - 0.05, env.height + 0.2])
    position = env.center_red + np.array([0., 0., 0.2])
    
    # # BLUE - place dynamic blocks positions
    # # position 1
    # position = env.center_blue + np.array([-(env.platform_width/2 - 0.03), -(env.platform_width/2 - 0.03), 0.03])
    # # position 2
    # position = env.center_blue + np.array([0, -(env.platform_width/2 - 0.03), 0.03])
    # # position 3
    # position = env.center_blue + np.array([(env.platform_width/2 - 0.03), -(env.platform_width/2 - 0.03), 0.03])
    # # position 4
    # position = env.center_blue + np.array([-(env.platform_width/2 - 0.03), 0, 0.03])
    # # position 5 (backup position)
    # position = env.center_blue + np.array([-(env.platform_width/2 - 0.03), (env.platform_width/2 - 0.03), 0.03])

    # BLUE - stack dynamic blocks positions
    # position 1
    # position 2
    # position 3
    # position 4
    
    # position[1] = 0.

    # x = np.array([1., 0., 0.]) ############ to be changed, cause we do not grip from the top
    # y = np.array([0., -1., 0.])
    z = np.array([0., 0., -1.])
    y = np.array([1., 0., 0.])
    # y = np.array([(env.radius + 0.05)/np.sqrt(2), (env.radius + 0.05)/np.sqrt(2) + env.radius, 0.])
    # z = np.array([-env.radius, 0.0, env.height-0.2]) - position
    # z = np.array([0., 1., -1.])
    x = np.cross(y, z)

    x = x/np.linalg.norm(x)
    y = y/np.linalg.norm(y)
    z = z/np.linalg.norm(z)

    position -= env.robot_base_red
    # position -= env.robot_base_blue

    rotation_matrix = np.hstack((x.reshape(3,1), y.reshape(3,1), z.reshape(3,1)))
    pose = np.hstack((rotation_matrix, position.reshape(3,1)))
    pose = np.vstack((pose, np.array([0, 0, 0, 1])))

#     pose = np.array([[-0.99895404 , 0.04572557 , 0.     ,     0.16468932],
#  [ 0.04572557 , 0.99895404 , 0.   ,      -0.18403141],
#  [ 0.        ,  0.      ,   -1.    ,      0.27560536],
#  [ 0.       ,   0.     ,     0.     ,     1.        ]])
#     pose[:3, 3] -= env.robot_base_red
    

    # pose = np.array([[-2.47197398e-01, 2.95481027e-01, -9.22813312e-01,  2.24580742e-01 - 0.1],
    #         [ 7.65627001e-02, 9.55348602e-01,  2.85389561e-01,  7.26039988e-01 - 0.05],
    #         [ 9.65935608e-01, -1.05521908e-04, -2.58782514e-01 , 2.19967753e-01],
    #         [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])

    # right_configuration = np.array([ 0.84993119,  1.06922655 , 0.76135993 ,-1.21502817 , 0.54872997 , 1.7833866,-1.18264383] )

    
    # rotation_matrix_current = pose[:3,:3]

    # axis = np.array([0.0, 1.0, 0.0])
    # axis = axis/np.linalg.norm(axis)
    # angle = -np.deg2rad(20.0)
    # rotvec = axis * angle
    # rotation_matrix_trans = R.from_rotvec(rotvec).as_matrix()

    # rotation_matrix_try = rotation_matrix_current @ rotation_matrix_trans

    # pose[:3, :3] = rotation_matrix_try

    # standby_configuration = np.array([ 0.68893224 , 1.06267568 , 0.67994595 ,-0.90568375 , 0.53786226 , 1.74134063, -0.96663082] )
    if team == 'red':
        standby_configuration = np.array([-1.05472241 ,-0.99948921  ,1.83233867, -0.82716498 , 0.97981148  ,1.36334427,  0.48816522])
        # standby_configuration = np.array([ 1.32766334 , 0.28948362  ,0.36751803 ,-1.14773112 ,-0.1298407  , 1.65771037,  0.95206491])
    # BLUE
    # standby_configuration = np.array([ 1.04830745 -1.02905062 -1.7685155  -0.82269323 -1.04061906  1.3413959 -2.04853074])

        grip_configuration = np.array([ 0.9728,  1.2579,  0.75 , -0.8674 ,  0.5607,  1.6436 , -1.1162] )

        temp_configuration = np.array([[ 0.32305623 , 0.09319256 , 0.1947716  ,-2.22563592, -0.02452935  ,2.31692631,  1.31905428],
                                   [ 0.40204208 , 0.3206562  , 0.04152907, -1.90099434 ,-0.01644505 , 2.22132744,  1.23681461],
                                   [ 0.41590124  ,0.58667391 ,-0.04133152 ,-1.4814851  , 0.02602317 , 2.06762328,  1.15446726],
                                   [ 0.17834827 ,-0.01674468 , 0.17067261 ,-2.36400503 , 0.00398733  ,2.34749961, -0.4391956 ]
                                #    [-0.0310031 , -0.08418885 , 0.19028863 ,-2.44350807 , 0.02259693 , 2.36070422,  2.49877044]
                                    ])
    # BLUE
    # temp_configuration = np.array([-0.32300043,  0.09319359, -0.1948279,  -2.22563588,  0.02453662,  2.31692619, -1.31905947],
    #                               [-0.40205964,  0.32065596, -0.04151016, -1.90099436,  0.01643755,  2.22132752, -1.23680969 + pi/2],
    #                               [-0.41591353,  0.5866742,   0.04134714, -1.48148506, -0.02603302,  2.06762312, -1.15446184 + pi/2],
    #                               [-0.16460494, -0.01678596, -0.18440821, -2.36400477, -0.00431533,  2.3474987, -1.13136116],
    #                               [-0.03945169, -0.08327459, -0.11981713, -2.44353794, -0.01412641,  2.36080777, -0.93422053])

        temp_rise_configuration = np.array([[ 0.272114   , 0.02135111 , 0.24498653 ,-2.17354963 ,-0.00637817 , 2.19425375,  1.30616896],
                                        # [ 0.36831182 , 0.26303237 , 0.07840437 ,-1.84881634 ,-0.02374747 , 2.11095267,  1.24164285],
                                        # [ 0.40272532 , 0.54391048 ,-0.02518702 ,-1.41971274 , 0.01410652  ,1.96344759,  1.16117192],
                                        # [ 0.04758803 ,-0.09761197 , 0.29764292 ,-2.31002702 , 0.03578967 , 2.21637367,  2.67855118],
                                        # [-0.11997488 ,-0.16954142  ,0.27428131 ,-2.387858  ,  0.05756867 , 2.22373957,  2.47175981]
                                        [ 0.31095929,  0.24569294 , 0.14096387 ,-1.82214864 ,-0.03883035 , 2.06519424,  1.25157375],
                                          [ 0.37162451 , 0.53178973 , 0.01551921, -1.3873994 , -0.0083719  , 1.91913658,  1.17325675],
                                          [ 0.22467561 ,-0.11935326 , 0.12187234 ,-2.28318777 , 0.01746645  ,2.1646397, -0.44947776]
                                        #   [ 1.68774581e-01, -1.89402082e-01, -9.26975983e-03 ,-2.36060094e+00, -2.11524473e-03 , 2.17120351e+00, -6.24527496e-01]
                                          ])
        
        observation_configuration = np.array([ 0.13956188 , 0.08149442 , 0.16024589 ,-1.78180635 ,-0.0135612  , 1.8622359, -0.48221643])
    
    # BLUE
    # temp_rise_configuration = np.array([-2.56339926e-01, -2.66639424e-03, -2.59650951e-01, -2.14731754e+00, -8.15200687e-04,  2.14474037e+00, -1.30094555e+00],
    #                                    [-0.35749753,  0.24437221, -0.09033726, -1.82223207,  0.02480327,  2.06551824, -1.24234106 + pi/2],
    #                                    [-0.39855263,  0.53182979,  0.02000159, -1.38739406, -0.01079056,  1.91911774, -1.16302806 + pi/2],
    #                                    [-0.11155569, -0.12177318, -0.23259093, -2.28310256, -0.0337821, 2.16431276, -1.10898462],
    #                                    [-1.08712371e-03, -1.91677512e-01, -1.53718282e-01, -2.36047405e+00, -3.53473630e-02,  2.17071146e+00, -9.17469524e-01])

        # key_point_for_dynamic_put = np.array([[ 0.31095929,  0.24569294 , 0.14096387 ,-1.82214864 ,-0.03883035 , 2.06519424,  1.25157375],
        #                                   [ 0.37162451 , 0.53178973 , 0.01551921, -1.3873994 , -0.0083719  , 1.91913658,  1.17325675],
        #                                   [ 0.22467561 ,-0.11935326 , 0.12187234 ,-2.28318777 , 0.01746645  ,2.1646397, -0.44947776],
        #                                   [ 1.68774581e-01, -1.89402082e-01, -9.26975983e-03 ,-2.36060094e+00, -2.11524473e-03 , 2.17120351e+00, -6.24527496e-01]])
    
    elif team == 'blue':
        # standby_configuration = np.array([ 1.04830745, -1.02905062 ,-1.7685155 , -0.82269323, -1.04061906 , 1.3413959 ,-2.04853074])
        standby_configuration = np.array([-1.35671779  ,0.71245488, -0.40794795, -0.87960244 , 0.26239033  ,1.55072233,  0.68876784])

        temp_configuration = np.array([[-0.32300043,  0.09319359, -0.1948279,  -2.22563588,  0.02453662,  2.31692619, -1.31905947 + pi/2],
                                  [-0.40205964,  0.32065596, -0.04151016, -1.90099436,  0.01643755,  2.22132752, -1.23680969 + pi/2],
                                  [-0.41591353,  0.5866742,   0.04134714, -1.48148506, -0.02603302,  2.06762312, -1.15446184 + pi/2],
                                  [-0.16460494, -0.01678596, -0.18440821, -2.36400477, -0.00431533,  2.3474987, -1.13136116 + pi]
                                #   [-0.03945169, -0.08327459, -0.11981713, -2.44353794, -0.01412641,  2.36080777, -0.93422053]
                                  ])
        
        temp_rise_configuration = np.array([[-2.56339926e-01, -2.66639424e-03, -2.59650951e-01, -2.14731754e+00, -8.15200687e-04,  2.14474037e+00, -1.30094555e+00 + pi/2],
                                       [-0.35749753,  0.24437221, -0.09033726, -1.82223207,  0.02480327,  2.06551824, -1.24234106 + pi/2],
                                       [-0.39855263,  0.53182979,  0.02000159, -1.38739406, -0.01079056,  1.91911774, -1.16302806 + pi/2],
                                       [-0.11155569, -0.12177318, -0.23259093, -2.28310256, -0.0337821, 2.16431276, -1.10898462 + pi]
                                    #    [-1.08712371e-03, -1.91677512e-01, -1.53718282e-01, -2.36047405e+00, -3.53473630e-02,  2.17071146e+00, -9.17469524e-01]
                                       ])

    # pose = fk.forward(configuration)[1]

    # pose[]
    # print(arm.get_positions())
    arm.safe_move_to_position(arm.neutral_position())

    def get_configuration(pose):
        seed = arm.get_positions()
        configuration, rollout, success, message = ik.inverse(pose, seed, method='J_pseudo', alpha = .5)
        print(message)
        return configuration, success

    # configuration = np.array([ 0.9728-0.3 + 10./180.*np.pi,  1.2579,  0.75  , -0.8674,  0.5607,  1.6436, -1.1162] )
    # _, pose = fk.forward(configuration)
    # configuration = np.array([ 0.9728,  1.2579,  0.75  , -0.8674,  0.5607,  1.6436, -1.1162] )
    # configuration = np.array([ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517])
    # pose = fk.forward(configuration)[1]
    # configurations = np.array([
    #                             [-0.0975,  0.2073, -0.1692, -2.0558,  0.0449,  2.2597,  0.4937] ,
    #                             [-0.1087,  0.1437, -0.1578, -2.0025,  0.0268,  2.1443,  0.506 ] ,
    #                             [-0.1168,  0.0973, -0.1489, -1.9295,  0.016 ,  2.0256,  0.5133] ,
    #                             [-0.1218,  0.0688, -0.1432, -1.8359,  0.0104,  1.904 ,  0.5173] ,
    #                             [-0.1241,  0.0593, -0.1411, -1.7201,  0.0085,  1.7789,  0.5186] ,
    #                             [-0.1248,  0.0708, -0.1425, -1.578 ,  0.0101,  1.6482,  0.5177] ,
    #                             [-0.1261,  0.1076, -0.1469, -1.401 ,  0.0158,  1.5075,  0.5143] ,
    #                             [-0.1344,  0.1814, -0.1515, -1.1668,  0.0279,  1.3463,  0.5082] ,
    #                             ])


    # configuration = get_configuration(pose)
    # print('pose:', pose)
    # print('configuration:', configuration)
    
    # arm.close_gripper()

    # arm.safe_move_to_position(arm.neutral_position())
    # target= transform( np.array([0.3, 0.745, 0.162]), np.array([ 0,pi,pi]) )
    # seed= np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])
    # q, rollout, success, message = ik.inverse(target, seed, method='J_pseudo', alpha=.5)
    # #print(q)
    # q[4]=q[4]+pi/2-0.45
    # q[6] = q[6]-3*pi/4 - pi/8 - pi/20
    # q[3] = q[3] - pi/10
    # q[5]-=pi/4 -0.3 -0.3 
    # q[0]-=pi/24

    # # q =[-1.76167489, -1.40459303  ,1.94353466 ,-0.89720455 , 2.28922666 , 1.14506434, -1.14923264]

    # q[5]+= pi/10
    # # q[4]-=pi/25
    # # q[4]-=pi/7
        
    # q[5]-=pi/5
    # q[0]+=pi/11
    
    # arm.safe_move_to_position(q)
    # print(q)


    if simorreal == 'sim':
        advance_time = 10.
    else:
        advance_time = 4.0
    # configuration = np.array([-0.20638,  0.03282, -0.21015, -1.71503 , 0.00695,  1.74713,  0.36776])
    # arm.safe_move_to_position(configuration)
    # arm.safe_move_to_position(standby_configuration)
    # for i in range(4):
    #     arm.safe_move_to_position(temp_rise_configuration[i])
    #     arm.safe_move_to_position(temp_configuration[i])
    #     arm.safe_move_to_position(temp_rise_configuration[i])
    # test_config = np.array([-0.40205964,  0.32065596, -0.04151016, -1.90099436,  0.01643755,  2.22132752, -1.23680969 + pi/2])
    # arm.safe_move_to_position(test_config)

    # arm.safe_move_to_position(standby_configuration)
    # arm.safe_move_to_position(grip_configuration)

    ts.append((time_in_seconds() - start_time, 'ready'))
    arm.open_gripper()
    ts.append((time_in_seconds() - start_time, 'gripper opened'))
    for i in range(4):
        arm.safe_move_to_position(standby_configuration)
        ts.append((time_in_seconds() - start_time, 'standby{}'.format(i)))
        predict_start_time = time_in_seconds()
        # # press ENTER to continue
        # input("\nWaiting for start... Press ENTER to begin!\n") # get set!
        # ts.append((time_in_seconds() - start_time, 'get grip configuration{}'.format(i)))
        # predicted_configuration, grip_configuration = get_grip_configurations()
        grip_configuration, success = get_grip_configurations()
        # if predicted_configuration is None or grip_configuration is None:
        if grip_configuration is None or not success:
            rospy.logwarn("Skip this block! Because not detected!")
            continue
        ts.append((time_in_seconds() - start_time, 'ready to move{}'.format(i)))
        # arm.safe_move_to_position(predicted_configuration)
        arm.safe_move_to_position(grip_configuration)
        ts.append((time_in_seconds() - start_time, 'ready to grip{}'.format(i)))
        # # press ENTER to grip
        # input("\nPress ENTER to grip!\n") # get set!
        ready_to_grip_time = time_in_seconds()
        if simorreal == 'sim':
            if ready_to_grip_time - predict_start_time < (advance_time-5.8):
                rospy.sleep(advance_time - 5.8 - (ready_to_grip_time - predict_start_time))
            else:
                print('advance time is too short')
        else:
            if ready_to_grip_time - predict_start_time < (advance_time-0.5):
                rospy.sleep(advance_time - 0.5 - (ready_to_grip_time - predict_start_time))
            else:
                print('advance time is too short')
        ts.append((time_in_seconds() - start_time, 'start to grip{}'.format(i)))
        # grip(grip_configuration)
        grip()
        # wait_unitl_grabbed()
        ts.append((time_in_seconds() - start_time, 'gripped{}'.format(i)))
        if i != 0:
            arm.safe_move_to_position(temp_rise_configuration[i])
        arm.safe_move_to_position(temp_configuration[i])
        ts.append((time_in_seconds() - start_time, 'reach temp{}'.format(i)))
        arm.open_gripper()
        ts.append((time_in_seconds() - start_time, 'unload{}'.format(i)))
        arm.safe_move_to_position(temp_rise_configuration[i])
        ts.append((time_in_seconds() - start_time, 'rise{}'.format(i)))

    arm.safe_move_to_position(observation_configuration)
    dynamic_poses = static_grabber.blockDetect()
    



    # # observation
    # static_grabber.moveTo( [-0.20638,  0.03282, -0.21015, -1.71503 , 0.00695,  1.74713,  0.36776])
    # #detect static blocks: sorted poses in world frame nx4x4
    # H_Sorted_1 = static_grabber.blockDetect()
    # # ee's supposed pose
    # H_Sorted = []
    # for (pose) in H_Sorted_1:
    #     pose = static_grabber.blockPose(pose)
    #     H_Sorted.append(pose)
    #     # seed = eepose for hot start
    # arm.open_gripper()
    # for (pose) in H_Sorted:
    #     pose[2][3] += 0.1
    #     q, rollout, success, message = ik.inverse( pose, arm.get_positions(), "J_pseudo", 0.5)
    #     jointPositions, T0e = fk.forward(q)
    #     static_grabber.moveTo(q)
    #     # move down to grasp
    #     pose[2][3] -= 0.1
    #     q1, rollout, success, message = ik.inverse( pose, q, "J_pseudo", 0.5)
    #     static_grabber.moveTo(q1)
    #     static_grabber.grab()

    
    # H_Sorted_1 = dynamic_poses
    # # ee's supposed pose
    # H_Sorted = []
    # for (pose) in H_Sorted_1:
    #     pose = static_grabber.blockPose(pose)
    #     H_Sorted.append(pose)
    #     # seed = eepose for hot start
    # arm.open_gripper()
    # for i, pose in enumerate(H_Sorted):
    #     pose[2][3] += 0.1
    #     q, rollout, success, message = ik.inverse( pose, arm.get_positions(), "J_pseudo", 0.5)
    #     jointPositions, T0e = fk.forward(q)
    #     static_grabber.moveTo(q)
    #     # move down to grasp
    #     pose[2][3] -= 0.1
    #     if i == H_Sorted_1.shape[0] - 1:
    #         temp = pose[:3, 1].copy()
    #         pose[:3, 1] = pose[:3, 0].copy()
    #         pose[:3, 0] = -temp
    #     q1, rollout, success, message = ik.inverse( pose, q, "J_pseudo", 0.5)
    #     static_grabber.moveTo(q1)
    #     static_grabber.grab()

    

    for t in ts:
        print('time: \n', t, '\n')

    # save ts to file
    save_path = '/home/kaihan/meam520_ws/src/meam520_labs/ts.txt'
    with open(save_path, 'w') as f:
        for t in ts:
            f.write(str(t) + '\n')
        print('file saved to:', save_path)

    
    # for configuration in configurations:
    #     print('configuration:', configuration)
    #     arm.safe_move_to_position(configuration)
    #     rospy.sleep(1.0)

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")