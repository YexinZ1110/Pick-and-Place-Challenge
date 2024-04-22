import numpy as np
from math import pi,cos,sin,sqrt
from lib.calculateFK import FK
from lib.calculateFKJac import FK_Jac

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    joint_num=7
    J = np.zeros((6, joint_num))

    # # STUDENT CODE GOES HERE
    fk = FK()
    rot_axis=fk.get_axis_of_rotation(q_in).T # (7,3)
    joints_pos,_=fk.forward(q_in)

    o0e=joints_pos[-1] # (3,)
    o=o0e-joints_pos[:-1,:] # (7,3)

    Jv=np.zeros((3,joint_num))
    Jw=np.zeros((3,joint_num))
    for i in range(joint_num):
        Jv[:,i]=np.cross(rot_axis[i],o[i])
        Jw[:,i]=rot_axis[i]
    J=np.vstack((Jv,Jw))

    return J

def calcJacobian_n(q_in,n):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """
    joint_num=7

    # # STUDENT CODE GOES HERE
    fk = FK_Jac()
    rot_axis=fk.get_axis_of_rotation(q_in).T # (7,3)
    joints_pos,_=fk.forward_expanded(q_in) # (10,3)

    o0e=joints_pos[n] # (3,)
    o=o0e-joints_pos[:-1,:] # (7,3)

    Jv=np.zeros((3,joint_num))
    Jw=np.zeros((3,joint_num))
    for i in range(joint_num):
        Jv[:,i]=np.cross(rot_axis[i],o[i])
        Jw[:,i]=rot_axis[i]
    J=np.vstack((Jv,Jw))

    return Jv

if __name__ == '__main__':
    q= np.array([pi/4, -1, 1, -np.pi/2, 1, np.pi/2, 2])
    for i in range(9):
        print(i,"\n",np.round(calcJacobian_n(q,i),3))
    # print("\n",np.round(calcJacobian_n(q,7),3))

    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, -np.pi/4])
    # print(np.round(calcJacobian(q),3))

    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 2])
    # q = np.array([0., 0., 0., 0., 0., 0., 0.])
    # print(np.round(calcJacobian(q),3))
