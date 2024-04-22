import numpy as np 
from lib.calcJacobian import calcJacobian
from lib.FK_velocity import FK_velocity


def IK_velocity(q_in, v_in, omega_in,method):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
    v_omega = np.vstack((v_in.reshape(3,1), omega_in.reshape(3,1)))
    valid_indices = []
    for i, value in enumerate(v_omega):
        if not np.isnan(value):
            valid_indices.append(i)
    
    v_omega = v_omega[valid_indices]
    
    J = calcJacobian(q_in)
    J = J[valid_indices]
    if method == 'J_pseudo':
        J_pseudo_inverse = np.linalg.pinv(J)
        dq = J_pseudo_inverse @ v_omega # (7,6)@(6,1)

    elif method == 'J_trans':
        J_T=J.T
        dq = J_T @ v_omega # (7,6)@(6,1)

    
    return dq.reshape(1,7)
