import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))

    v_omega = np.vstack((v_in, omega_in))
    valid_indices = []
    for i, value in enumerate(v_omega):
        if not np.isnan(value):
            valid_indices.append(i)
    
    v_omega = v_omega[valid_indices]
    
    J = calcJacobian(q_in)
    J = J[valid_indices]

    J_pseudo_inverse = np.linalg.pinv(J) # (7,6)
    I=np.eye(J_pseudo_inverse.shape[0])
    null=(I-J_pseudo_inverse@J)@b
    dq = J_pseudo_inverse @ v_omega # (7,6)@(6,1)+((7,7)-(7,6)@(6,7))
    dq=dq.T
    return dq

