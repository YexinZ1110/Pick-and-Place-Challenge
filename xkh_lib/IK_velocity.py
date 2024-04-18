import numpy as np 
from lib.calcJacobian import calcJacobian
# from calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
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

     dq = np.zeros((1, 7))

     v_in = v_in.reshape((3,1))
     omega_in = omega_in.reshape((3,1))

     jac_old = calcJacobian(q_in)

     command_old = np.vstack((v_in, omega_in))
     nan_mask = np.isnan(command_old)
     command = command_old[~nan_mask]
     nan_mask = nan_mask.flatten()
     jac = jac_old[~nan_mask]


     # jac_expanded = np.hstack((jac, command))
     
     # # if np.linalg.matrix_rank(jac_expanded / np.linalg.norm(jac_expanded, axis=0)) > np.linalg.matrix_rank(jac / np.linalg.norm(jac, axis=0)):
     # if np.linalg.matrix_rank(jac_expanded) > np.linalg.matrix_rank(jac):
     #      dq = np.linalg.lstsq(jac, command, rcond=None)[0]
     # else:
     #      jac_inv = np.linalg.pinv(jac)
     #      dq = jac_inv @ command
    
     dq = np.linalg.lstsq(jac, command, rcond=None)[0]

     return dq

if __name__ == '__main__':
     q = np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4], dtype=float)
     # v = np.array([0, 1, 0], dtype=float)
     v = np.array([np.nan, np.nan, np.nan], dtype=float)
     omega = np.array([0, 0, 1], dtype=float)
     dq = IK_velocity(q, v, omega)
     print('dq: ', dq)