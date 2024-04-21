import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-4, max_steps=500, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the 
        target joint position and the current joint position 

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        att_f = np.zeros((3, 1)) 
        d = 0.2 # distance threshold
        gain = 5.0 # gain

        if np.linalg.norm(target - current) > d:
            att_f = (target - current) / np.linalg.norm(target - current)
        else:
            att_f = gain * (target - current)

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box 

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        rep_f = np.zeros((3, 1))
        d_critical = 0.2 # distance threshold
        gain = 5.0 # gain

        d, unitvec = PotentialFieldPlanner.dist_point2box(current.T, obstacle)
        if d[0] < d_critical:
            rep_f = -gain * (1.0/d[0] - 1.0/d_critical) * (1.0/(d[0]**2)) * unitvec.reshape(3, 1)
        ## END STUDENT CODE

        # print('rep_f', rep_f)
        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        joint_forces = np.zeros((3, 9)) 
        for i in range(9):
            joint_forces[:, i] = PotentialFieldPlanner.attractive_force(target[:, i].reshape(3, 1), current[:, i].reshape(3, 1)).reshape(3)
            for j in range(len(obstacle)):
                # print(obstacle[j, :], current[:, i].reshape(3, 1))
                joint_forces[:, i] += PotentialFieldPlanner.repulsive_force(obstacle[j, :], current[:, i].reshape(3, 1)).reshape(3)

        ## END STUDENT CODE
        # print('joint_forces', joint_forces)
        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each 
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint ????????? why 1x9
        """

        ## STUDENT CODE STARTS HERE
        joint_torques = np.zeros((1, 7)) 

        joint_positions, T0e = PotentialFieldPlanner.fk.forward_expanded(q)
        z0 = T0e[:, :3, 2] # 10x3

        for i in range(7): # force on last 6 joints and the left virtual joint
            jac_v_temp = np.zeros((3, 7))
            for j in range(i+1):
                jac_v_temp[:, j] = np.cross(z0[j+1], joint_positions[i+1]-joint_positions[j])
            joint_torques += jac_v_temp.T @ joint_forces[:, i]
        for i in range(7, 9): # force on the right virtual joint and the end effector
            jac_v_temp = np.zeros((3, 7))
            for j in range(7):
                jac_v_temp[:, j] = np.cross(z0[j+1], joint_positions[i+1]-joint_positions[j])
            joint_torques += jac_v_temp.T @ joint_forces[:, i]

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE
        joint_positions_current, _ = PotentialFieldPlanner.fk.forward_expanded(current)
        joint_positions_target, _ = PotentialFieldPlanner.fk.forward_expanded(target)
        distance = np.linalg.norm(joint_positions_target[1:] - joint_positions_current[1:])
        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal 
        configuration 

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task. 
        """

        ## STUDENT CODE STARTS HERE
        dq = np.zeros((1, 7))
        gain = np.ones((1, 7)) * 0.1
        
        joint_positions_current, _ = PotentialFieldPlanner.fk.forward_expanded(q)
        joint_positions_target, _ = PotentialFieldPlanner.fk.forward_expanded(target) # 10x3
        joint_forces = PotentialFieldPlanner.compute_forces(joint_positions_target[1:].T, map_struct.obstacles, joint_positions_current[1:].T) # 3x9
        print('joint_forces', joint_forces)
        # print('joint_positions_current', joint_positions_current[1:].T)
        # print('joint_positions_target', joint_positions_target[1:].T)
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q) # 1x7
        print('joint_torques', joint_torques)
        dq = joint_torques * gain
        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes 
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles. 
        """

        q_path = np.array([]).reshape(0,7)
        q_path = np.vstack((q_path, start))

        while True:

            ## STUDENT CODE STARTS HERE
            
            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code 
            
            # Compute gradient 
            # TODO: this is how to change your joint angles 
            dp = PotentialFieldPlanner.compute_gradient(start, goal, map_struct)

            # Termination Conditions
            threshold = 1e-3
            if PotentialFieldPlanner.q_distance(goal, start) < threshold: # TODO: check termination conditions
                print('goal reached!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
                break # exit the while loop if conditions are met!

            # check if exceed joint limits
            if (start + dp < PotentialFieldPlanner.lower).any() or (start + dp > PotentialFieldPlanner.upper).any():
                print('exceed joint limits')
                # dp = np.clip(dp, PotentialFieldPlanner.lower - start, PotentialFieldPlanner.upper - start)
                print('dp', dp)
                print('p', start + dp)
                print('lower', PotentialFieldPlanner.lower)
                print('upper', PotentialFieldPlanner.upper)
                print('joint low', (start + dp < PotentialFieldPlanner.lower))
                print('joint high', (start + dp > PotentialFieldPlanner.upper))
                break

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function 
            def detectCollision_helper(dp):
                res = True

                for i in range(map_struct.obstacles.shape[0]):
                    next_joint_positions, _ = PotentialFieldPlanner.fk.forward_expanded(start + dp)
                    linept1 = np.zeros((9, 3))
                    linept2 = np.zeros((9, 3))

                    linept1[-1] = next_joint_positions[-1] # end effector
                    linept2[-1] = next_joint_positions[-2] # right virtual joint

                    linept1[-2] = next_joint_positions[-1]
                    linept2[-2] = next_joint_positions[-3] # left virtual joint
                    
                    linept1[-3] = next_joint_positions[-2] # right virtual joint
                    linept2[-3] = next_joint_positions[-4] # wrist

                    linept1[-4] = next_joint_positions[-3] # left virtual joint
                    linept2[-4] = next_joint_positions[-4] # wrist

                    linept1[:-4] = next_joint_positions[1:-4]
                    linept2[:-4] = next_joint_positions[:-5]

                    res = res and np.array(detectCollision(linept1, linept2, map_struct.obstacles[i]), dtype=bool).any()
            
                return res
            if detectCollision_helper(dp):
                print('collision')
                break

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            local_minima_threshold = 1e-3
            if np.linalg.norm(dp) < local_minima_threshold:
                print('local minima')
                dp = np.random.rand(1, 7) * 0.1
                while detectCollision_helper(dp):
                    dp = np.random.rand(1, 7) * 0.1

            print('dp', dp)
            start = start + dp
            q_path = np.vstack((q_path, start))
        q_path = np.vstack((q_path, goal))
            ## END STUDENT CODE

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
