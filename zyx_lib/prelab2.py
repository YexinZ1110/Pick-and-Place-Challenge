import numpy as np
from math import pi,cos,sin,sqrt
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

    # STUDENT CODE GOES HERE
    fk = FK()
    jointPositions,T0e=fk.forward(q_in)
    Ts=np.array([fk.T01,fk.T02,fk.T03@fk.t3,fk.T04,fk.T05@fk.t5,fk.T06@fk.t6,fk.T07@fk.t7]) # (7, 4, 4)

    Ts=np.array([fk.T01,fk.T02,fk.T03,fk.T04,fk.T05,fk.T06,fk.T07]) # (7, 4, 4)
    # print(np.round(Ts,3))
    R0iz=Ts[:,:3, 2] # (7, 3)


    o0e=T0e[:3,-1] # (3,)
    o0i=Ts[:,:3,-1] # (8, 3)
    o=o0e-o0i
    Jv=np.zeros((3,7))
    Jw=np.zeros((3,7))
    # J[:,0]=np.cross(np.array( [0,0,1]).transpose(),o[0])
    for i in range(7):
        Jv[:,i]=np.cross(R0iz[i],o[i])
        Jw[:,i]=R0iz[i]
    J=np.vstack((Jv,Jw))
    return J



if __name__ == '__main__':
    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q = np.array([0., 0., 0., 0., 0., 0., 0.])
    print(np.round(calcJacobian(q),3))
