import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK()
    jointPositions, T0e = fk.forward(q_in)
    z = fk.get_axis_of_rotation(q_in)
    Ai = fk.compute_Ai(q_in)
    On0 = jointPositions[-1, :]

    for i in range(7):
        if i == 0:
            O_iminus1 = np.array([0, 0, 0])
        else:
            O_iminus1 = Ai[i][:3, 3]
       
        z_i1 = z[:,i]

        J[:3, i] = np.cross(z_i1, np.array(On0 - O_iminus1))
        J[3:, i] = z_i1

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))