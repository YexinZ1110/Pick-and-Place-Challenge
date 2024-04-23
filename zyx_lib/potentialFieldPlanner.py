import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calcJacobian import calcJacobian
from lib.calculateFKJac import FK_Jac
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # JOINT LIMITS
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=1e-2, max_steps=500, min_step_size=1e-5):
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
        self.fk=FK_Jac()
        self.tol = tol
        self.max_steps = 700
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

        att_f = np.zeros((3, 1)) 
        norm=np.linalg.norm(current-target)
        d=1
        if norm> d: # Conic Well
            att_f=-(current-target)/norm
        else: # Parabolic Well
            att_f=-20*(current-target)

        return att_f.reshape(3,1)

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
        rho_0=0.2
        rep_f = np.zeros((3, 1)) 
        dist, unit=PotentialFieldPlanner.dist_point2box(current,obstacle)
        if dist<1e-6:
            rep_f=1e6*(-unit)
        elif dist<rho_0:
            rep_f=(1/dist-1/rho_0)*(1/(dist**2))*(-unit)
        
        return rep_f.reshape(3,1)

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
        p=p.reshape(1,-1)
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

        obs_n=len(obstacle)
        joint_n=9
        joint_forces = np.zeros((3, joint_n)) 
        for i in range(joint_n):
            rep_f=np.zeros((3,1))
            att_f=PotentialFieldPlanner.attractive_force(target[:,i],current[:,i])
            for j in range(obs_n):
                rep_f+=PotentialFieldPlanner.repulsive_force(obstacle[j],current[:,j])
            total_f=rep_f+att_f
            joint_forces[:,i]=total_f.reshape(-1)

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
        joint_torques - 1x9 numpy array representing the torques on each joint 
        """
        joint_torques = np.zeros((9,1)) 
        for i in range(1,10):
            Jv=PotentialFieldPlanner.calcJacobian_n(q,i) #(3,9)
            # print(i,"\n",Jv)
            torque=Jv.T@joint_forces[:, i-1].reshape(3,1) # (9,3)@(3,1)
            joint_torques +=torque #(7,1)
        return joint_torques.T
    
    @staticmethod
    def calcJacobian_n(q_in,n):
        """
        Calculate the full Jacobian of the n joint in a given configuration
        :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
        :return: J - 6 x 7 matrix representing the Jacobian, where the first three
        rows correspond to the linear velocity and the last three rows correspond to
        the angular velocity, expressed in world frame coordinates
        """
        joint_num=10
        fk = FK_Jac()
        rot_axis=fk.get_axis_of_rotation(q_in).T # (9,3)
        joints_pos,_=fk.forward_expanded(q_in) # (10,3)
        o0e=joints_pos[n] # (3,)
        joints_pos[7]=joints_pos[6]
        joints_pos[8]=joints_pos[6]

        o=o0e-joints_pos # (n,3)

        Jv=np.zeros((3,joint_num-1))
        for i in range(1,joint_num):
            if i<n:
                Jv[:,i-1]=np.cross(rot_axis[i-1],o[i-1])
            else:
                break
        return Jv # (3,9)

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
        distance = np.linalg.norm(target-current)
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

        dq = np.zeros((1, 7))
        fk=FK_Jac()
        jointPos, _=fk.forward_expanded(q) #(10,3)
        target_joint, _=fk.forward_expanded(target) #(10,3)

        current=jointPos.T[:,1:]  #(3,9)
        target=target_joint.T[:,1:] #(3,9)
        obstacle=map_struct.obstacles #(n,6)
        forces=PotentialFieldPlanner.compute_forces(target,obstacle,current)
        torques=PotentialFieldPlanner.compute_torques(forces,q)[0,:7] # (1,7)
        dq=torques/np.linalg.norm(torques)
        print("dq:",dq)
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
        q_path=np.append(q_path, start.reshape((1, 7)), axis=0)
        cur_q=start
        cnt=0
        rate=0.01
        while True:
            # Compute gradient 
            gradient=PotentialFieldPlanner.compute_gradient(cur_q.reshape(7),goal,map_struct)
            # Termination Conditions
            if self.q_distance(goal,cur_q)<self.tol or cnt>self.max_steps:
                break
            step=rate*gradient

            # CHECK FOR COLLISIONS WITH OBSTACLES
            jointPos, _=self.fk.forward_expanded(cur_q.reshape(-1)) #(10,3)
            line_segments_pt1 = jointPos[:-1] # Each joint position except the last
            line_segments_pt2 = jointPos[1:] # Each joint position except the first
            for obs in map_struct.obstacles:
                if np.any(detectCollision(line_segments_pt1, line_segments_pt2, obs)):
                    continue

            # when detect a local minima, implement a random walk
            if np.linalg.norm(step)<self.min_step_size:
                 step = np.random.randn(1, 7)

            cur_q=cur_q+step
            q_path=np.vstack((q_path,cur_q.reshape(1,7)))
            cnt+=1

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner(tol=1e-2, max_steps=700, min_step_size=1e-5)
    
    # inputs 
    map_struct = loadmap("maps/map3.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print("path length:", len(q_path))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal[:])
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)


