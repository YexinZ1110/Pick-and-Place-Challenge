
import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.joints = np.array([[0, 0, 0.141, 0],
                                [0, -pi/2, 0.192, 0],
                                [0, pi/2, 0, 0],
                                [0.0825, pi/2, 0.195 + 0.121, 0],
                                [-0.0825, -pi/2, 0, 0],
                                [0, pi/2, 0.259 + 0.125, 0],
                                [0.088, pi/2, 0, 0],
                                [0, 0, 0.051 + 0.159, -pi/4]])
        
        self.offset = [0, 0, 0.195, 0, 0.125, -0.015, 0.051, 0]

        # pass

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)
        
        for i, joint in enumerate(self.joints):
            a, alpha, d, theta = joint
            if i > 0:
                theta += q[i-1]
            T0e = T0e @ self.Ai(a, alpha, d, theta)
            jointPositions[i,:] = (T0e @ np.array([0, 0, self.offset[i], 1]))[:3]

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1
    def Ai(self, a, alpha, d, theta):
        return np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])        
    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        axis_of_rotation_list = np.zeros((3, 7))
        T = np.identity(4)
        for i in range(7):
            a, alpha, d, theta = self.joints[i]
            if i > 0: 
                theta = q[i-1] + self.joints[i, 3]
            else:
                theta = self.joints[i, 3]

            T = T @ self.Ai(a, alpha, d, theta)

            z = T[:3, 2]
            axis_of_rotation_list[:, i] = z

        return axis_of_rotation_list
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        T0e = np.identity(4)
        Ai = []
        for i, joint in enumerate(self.joints):
            a, alpha, d, theta = joint
            if i > 0:
                theta += q[i-1]
            T0e = T0e @ self.Ai(a, alpha, d, theta)
            Ai.append(T0e)

        return Ai
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0,0,0,0,0,0,0])
    # q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.021, 0.102, 0.0])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)