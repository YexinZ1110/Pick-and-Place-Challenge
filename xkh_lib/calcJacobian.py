import numpy as np
# from lib.calculateFK import FK
# from calculateFK import FK
from math import pi

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    ## STUDENT CODE GOES HERE

    # FK
    a = np.array([0., 0., 0.0825, -0.0825, 0., 0.088, 0.])
    alpha = np.array([-pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0.])
    d = np.array([0.333, 0., 0.316, 0., 0.384, 0., 0.21])
    theta = np.array(q_in, dtype=float)
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

    # Jacobian
    jac = np.zeros((6, 7))
    z0 = np.zeros((8, 3))
    z0[0] = np.array([0, 0, 1])
    z0[1:] = T0[:, :3, 2]
    o0 = np.zeros((8, 3))
    o0[0] = 0
    o0[1:] = T0[:, :3, 3]
    jac[:3] = np.cross(z0[:-1], o0[-1]-o0[:-1]).T
    jac[3:] = z0[:-1].T
    
    return jac

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))
