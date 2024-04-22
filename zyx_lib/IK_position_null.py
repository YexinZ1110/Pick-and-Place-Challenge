import numpy as np
from math import pi, acos
from scipy.linalg import null_space

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
from lib.IK_velocity import IK_velocity
# from lib.IK_velocity import IK_velocity  #optional


class IK:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK()

    def __init__(self,linear_tol=1e-4, angular_tol=1e-3, max_steps=1000, min_step_size=1e-5):
        """
        Constructs an optimization-based IK solver with given solver parameters.
        Default parameters are tuned to reasonable values.

        PARAMETERS:
        linear_tol - the maximum distance in meters between the target end
        effector origin and actual end effector origin for a solution to be
        considered successful
        angular_tol - the maximum angle of rotation in radians between the target
        end effector frame and actual end effector frame for a solution to be
        considered successful
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # solver parameters
        self.linear_tol = linear_tol
        self.angular_tol = angular_tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################

    @staticmethod
    def displacement_and_axis(target, current):
        """
        Helper function for the End Effector Task. Computes the displacement
        vector and axis of rotation from the current frame to the target frame

        This data can also be interpreted as an end effector velocity which will
        bring the end effector closer to the target position and orientation.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        current - 4x4 numpy array representing the "current" end effector orientation

        OUTPUTS:
        displacement - a 3-element numpy array containing the displacement from
        the current frame to the target frame, expressed in the world frame
        axis - a 3-element numpy array containing the axis of the rotation from
        the current frame to the end effector frame. The magnitude of this vector
        must be sin(angle), where angle is the angle of rotation around this axis
        """

        ## STUDENT CODE STARTS HERE
        displacement = np.zeros(3)
        R_des=target[:3,:3]
        R_curr=current[:3,:3]
        axis=calcAngDiff(R_des,R_curr)

        T_des=target[:3,-1]
        T_curr=current[:3,-1]
        displacement = T_des-T_curr

        ## END STUDENT CODE
        return displacement, axis

    @staticmethod
    def distance_and_angle(G, H):
        """
        Helper function which computes the distance and angle between any two
        transforms.

        This data can be used to decide whether two transforms can be
        considered equal within a certain linear and angular tolerance.

        Be careful! Using the axis output of displacement_and_axis to compute
        the angle will result in incorrect results when |angle| > pi/2

        INPUTS:
        G - a 4x4 numpy array representing some homogenous transformation
        H - a 4x4 numpy array representing some homogenous transformation

        OUTPUTS:
        distance - the distance in meters between the origins of G & H
        angle - the angle in radians between the orientations of G & H
        """
        
        ## STUDENT CODE STARTS HERE
        distance = 0
        angle = 0
        T_G=G[:3,-1]
        T_H=H[:3,-1]
        disp = T_G - T_H
        distance = np.linalg.norm(disp)

        R_G=G[:3,:3]
        R_H=H[:3,:3]
        delta_R=R_G.T@R_H
        inside = np.clip((np.trace(delta_R)-1)/2, -1, 1)
        angle = acos(inside)
        ## END STUDENT CODE
        return distance, angle

    def is_valid_solution(self,q,target):
        """
        Given a candidate solution, determine if it achieves the primary task
        and also respects the joint limits.

        INPUTS
        q - the candidate solution, namely the joint angles
        target - 4x4 numpy array representing the desired transformation from
        end effector to world

        OUTPUTS:
        success - a Boolean which is True if and only if the candidate solution
        produces an end effector pose which is within the given linear and
        angular tolerances of the target pose, and also respects the joint
        limits.
        """

        ## STUDENT CODE STARTS HERE
        success = True
        found=""
        reason=""
        _, T0e = self.fk.forward(q)
        assert T0e.shape == (4, 4)
        distance, angle = self.distance_and_angle(T0e, target)
        if distance > self.linear_tol:
            success=False
            reason="Distance between the achieved and target end effector position is less than linear_tol."
        if angle > self.angular_tol:
            success=False
            reason="The joint angles are out of the joint limits."
        if all(q<self.lower) and all(q > self.upper):
            success = False
            reason="The magnitude of the angle between the achieved and target end effector orientations is less than angular.tol"
        if not success:
            found="Solution not found. "
        else:
            found="Solution found. "
        message=found+reason
        ## END STUDENT CODE
        return success, message

    ####################
    ## Task Functions ##
    ####################

    @staticmethod
    def end_effector_task(q,target,method):
        """
        Primary task for IK solver. Computes a joint velocity which will reduce
        the error between the target end effector pose and the current end
        effector pose (corresponding to configuration q).

        INPUTS:
        q - the current joint configuration, a "best guess" so far for the final answer
        target - a 4x4 numpy array containing the desired end effector pose
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm
        
        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        ## STUDENT CODE STARTS HERE
        dq = np.zeros(7)
        fk=FK()
        _,current=fk.forward(q)
        distance, axis=IK.displacement_and_axis(target,current)
        dq=IK_velocity(q,distance,axis,method)
        ## END STUDENT CODE
        return dq

    @staticmethod
    def joint_centering_task(q,rate=5e-1): 
        """
        Secondary task for IK solver. Computes a joint velocity which will
        reduce the offset between each joint's angle and the center of its range
        of motion. This secondary task acts as a "soft constraint" which
        encourages the solver to choose solutions within the allowed range of
        motion for the joints.

        INPUTS:
        q - the joint angles
        rate - a tunable parameter dictating how quickly to try to center the
        joints. Turning this parameter improves convergence behavior for the
        primary task, but also requires more solver iterations.

        OUTPUTS:
        dq - a desired joint velocity to perform this task, which will smoothly
        decay to zero magnitude as the task is achieved
        """

        # normalize the offsets of all joints to range from -1 to 1 within the allowed range
        offset = 2 * (q - IK.center) / (IK.upper - IK.lower)
        dq = rate * -offset # proportional term (implied quadratic cost)

        return dq
        
    ###############################
    ## Inverse Kinematics Solver ##
    ###############################

    def inverse(self, target, seed, method, alpha):
        """
        Uses gradient descent to solve the full inverse kinematics of the Panda robot.

        INPUTS:
        target - 4x4 numpy array representing the desired transformation from
        end effector to world
        seed - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], which
        is the "initial guess" from which to proceed with optimization
        method - a boolean variable that determines to use either 'J_pseudo' or 'J_trans' 
        (J pseudo-inverse or J transpose) in your algorithm

        OUTPUTS:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6], giving the
        solution if success is True or the closest guess if success is False.
        success - True if the IK algorithm successfully found a configuration
        which achieves the target within the given tolerance. Otherwise False
        rollout - a list containing the guess for q at each iteration of the algorithm
        """
        q = seed
        rollout = []


        ## STUDENT CODE STARTS HERE

        
        ## gradient descent:
        while True:
            rollout.append(q)
            # Primary Task - Achieve End Effector Pose
            dq_ik = IK.end_effector_task(q,target, method)

            # Secondary Task - Center Joints
            dq_center = IK.joint_centering_task(q)

            ## Task Prioritization
            J=calcJacobian(q)
            J_psu_inv=np.linalg.pinv(J)
            dq_second=(np.eye(J_psu_inv.shape[0])-J_psu_inv@J)@dq_center
            dq=dq_ik+dq_second
            # Check termination conditions
            if(np.linalg.norm(dq_ik)<self.min_step_size or len(rollout)>self.max_steps):
                break
            # update q
            q=q+alpha*dq
            q=q.reshape(-1)

        ## END STUDENT CODE

        success, message = self.is_valid_solution(q,target)
        return q, rollout, success, message

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    ik = IK()

    # # matches figure in the handout
    # seed = np.array([-0.01779206, -0.76012354,  0.01978261, -2.34205014, 0.02984053, 1.54119353+pi/2, 0.75344866])

    # target = np.array([
    #     [1,0,0,0.5],
    #     [0,-1,0,-0.2],
    #     [0,0,-1,.7],
    #     [0,0,0, 1],
    # ])

    # # Using pseudo-inverse 
    # q_pseudo, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(target, seed, method='J_pseudo', alpha=.5)

    # for i, q_pseudo in enumerate(rollout_pseudo):
    #     joints, pose = ik.fk.forward(q_pseudo)
    #     d, ang = IK.distance_and_angle(target,pose)
    #     print('iteration:',i,' q =',q_pseudo, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # # Using pseudo-inverse 
    # q_trans, rollout_trans, success_trans, message_trans = ik.inverse(target, seed, method='J_trans', alpha=.5)

    # for i, q_trans in enumerate(rollout_trans):
    #     joints, pose = ik.fk.forward(q_trans)
    #     d, ang = IK.distance_and_angle(target,pose)
    #     print('iteration:',i,' q =',q_trans, ' d={d:3.4f}  ang={ang:3.3f}'.format(d=d,ang=ang))

    # # compare
    # print("\n method: J_pseudo-inverse")
    # print("   Success: ",success_pseudo, ":  ", message_pseudo)
    # print("   Solution: ",q_pseudo)
    # print("   #Iterations : ", len(rollout_pseudo))
    # print("\n method: J_transpose")
    # print("   Success: ",success_trans, ":  ", message_trans)
    # print("   Solution: ",q_trans)
    # print("   #Iterations :", len(rollout_trans),'\n')


    fk = FK()
    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0., 0., 0., 0., 0., 0., 0.])
    # q=np.array([0,-pi/2,-pi/4,pi/2,pi,pi/4,0])
    q = np.array([
            [ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517] ,
            [ 0.1986,  0.1423,  0.0645, -2.0026, -0.0109,  2.1445,  1.0538] ,
            [ 0.1755,  0.0966,  0.0882, -1.9295, -0.0095,  2.0257,  1.0528] ,
            [ 0.1619,  0.0684,  0.102 , -1.8359, -0.0074,  1.904 ,  1.0514] ,
            [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507] ,
            [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512] ,
            [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524] ,
            [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525] ,
            ])
    T0es=[]
    qs=[]
    _, T0e = fk.forward(q[0])
    # print(q.shape)
    for i in range(q.shape[0]):
        # print("qi:",q[i])
        # print("Joint Positions:\n",joint_positions)
        # print("End Effector Pose \n",i,":",T0e)
        # T0es.append(T0e)
        T0e[2][3]=T0e[2][3]+0.03
        print("T0e tall",T0e)
        q_t, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T0e, np.array(q[i]), method='J_pseudo', alpha=.5)
        qs.append(q_t)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
    print(qs)