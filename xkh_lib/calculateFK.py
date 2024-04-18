import numpy as np
from math import pi
import matplotlib.pyplot as plt

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.a = np.array([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
        self.alpha = np.array([-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0.])
        self.d = np.array([0.333, 0., 0.316, 0., 0.384, 0., 0.21])
        pass

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

        a = self.a
        alpha = self.alpha
        d = self.d
        theta = np.array(q, dtype=float)
        theta[-1] = theta[-1] - pi/4
        
        A = np.zeros((7, 4, 4))
        A[:, 0, 0] = np.cos(theta)
        A[:, 0, 1] = -np.sin(theta) * np.cos(alpha)
        A[:, 0, 2] = np.sin(theta) * np.sin(alpha)
        A[:, 0, 3] = a * np.cos(theta)
        A[:, 1, 0] = np.sin(theta)
        A[:, 1, 1] = np.cos(theta) * np.cos(alpha)
        A[:, 1, 2] = -np.cos(theta) * np.sin(alpha)
        A[:, 1, 3] = a * np.sin(theta)
        A[:, 2, 0] = 0
        A[:, 2, 1] = np.sin(alpha)
        A[:, 2, 2] = np.cos(alpha)
        A[:, 2, 3] = d
        A[:, 3, 0] = 0
        A[:, 3, 1] = 0
        A[:, 3, 2] = 0
        A[:, 3, 3] = 1

        T0 = np.zeros((7, 4, 4))
        T0[0] = A[0]
        for i in range(6):
            T0[i+1] = T0[i] @ A[i+1]
        
        jointPositions[0] = np.array([0, 0, 0.141])
        jointPositions[1] = (T0[0] @ np.array([0, 0, 0, 1]))[:3]
        jointPositions[2] = (T0[1] @ np.array([0, 0, 0.195, 1]))[:3]
        jointPositions[3] = (T0[2] @ np.array([0, 0, 0, 1]))[:3]
        jointPositions[4] = (T0[3] @ np.array([0, 0, 0.125, 1]))[:3]
        jointPositions[5] = (T0[4] @ np.array([0, 0, -0.015, 1]))[:3]
        jointPositions[6] = (T0[5] @ np.array([0, 0, 0.051, 1]))[:3]
        jointPositions[7] = (T0[6] @ np.array([0, 0, 0, 1]))[:3]
        T0e = T0[-1]

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
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
        
        return()
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0, 0, 0, 0, 0, 0, 0])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(joint_positions[:, 0], joint_positions[:, 1], joint_positions[:, 2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()
