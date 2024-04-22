import numpy as np
from math import pi,cos,sin,sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.T01=np.identity(4)
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
        
        self.T01=self.DH(0             , 0.141, 0     , 0    )
        T12=self.DH(q[0]          , 0.192, 0     ,-pi/2 )
        T23=self.DH(q[1]          , 0    , 0     , pi/2 )
        T34=self.DH(q[2]          , 0.316, 0.0825, pi/2 )
        T45=self.DH(q[3]+pi/2+pi/2, 0    , 0.0825, pi/2 )
        T56=self.DH(q[4]          , 0.384, 0     ,-pi/2 ) 
        T67=self.DH(q[5]-pi/2-pi/2, 0    , 0.088 , pi/2 )
        T7e=self.DH(q[6]-pi/4     , 0.21 , 0     , 0    )

    
        jointPositions[0,:]=self.T01[0:3,3]

        self.T02=self.T01@T12
        jointPositions[1,:]=self.T02[0:3,3]

        self.t3=self.trans(0,0,0.195)
        self.T03=self.T02@T23
        jointPositions[2,:]=(self.T03@self.t3)[0:3,3]

        self.T04=self.T03@T34
        jointPositions[3,:]=self.T04[0:3,3]

        self.t5=self.trans(0,0,0.125)
        self.T05=self.T04@T45
        jointPositions[4,:]=(self.T05@self.t5)[0:3,3]

        self.t6=self.trans(0,0,-0.015)
        self.T06=self.T05@T56
        jointPositions[5,:]=(self.T06@self.t6)[0:3,3]

        self.t7=self.trans(0,0,0.051)
        self.T07=self.T06@T67
        jointPositions[6,:]=(self.T07@self.t7)[0:3,3]

        self.T0e=self.T07@T7e
        jointPositions[7,:]=self.T0e[0:3,3]
        # Your code ends here

        return jointPositions, self.T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    def rot_x(self,theta):
        R=np.array([[1,0,0,0],
                    [0,cos(theta),-sin(theta),0],
                    [0,sin(theta),cos(theta),0],
                    [0,0,0,1]])
        return R
                         
    def rot_z(self,theta):
        R= np.array([[cos(theta),-sin(theta),0,0],
                    [sin(theta),cos(theta),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])
        return R
        
    def trans(self,x,y,z):
        R= np.array([[1,0,0,x],
                    [0,1,0,y],
                    [0,0,1,z],
                    [0,0,0,1]])
        return R

    def DH(self,theta,d,a,alpha):
        A=self.rot_z(theta)@self.trans(0,0,d)@self.trans(a,0,0)@self.rot_x(alpha)
        return A

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
        self.forward(q)
        axis_of_rotation_list=np.vstack((self.T01[:3,2],self.T02[:3,2],
                                         self.T03[:3,2],self.T04[:3,2],
                                         self.T05[:3,2],self.T06[:3,2],
                                         self.T07[:3,2]))

        return axis_of_rotation_list.T
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return self.T0e
    
if __name__ == "__main__":

    fk = FK()
    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0., 0., 0., 0., 0., 0., 0.])
    # q=np.array([0,-pi/2,-pi/4,pi/2,pi,pi/4,0])
    q = np.array([
                [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507] ,
                [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512] ,
                [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524] ,
                [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525] ,
                ])
    T0es=[]
    qs=[]
    for i in range(q.shape[0]):
        joint_positions, T0e = fk.forward(q[i])
        
        # print("Joint Positions:\n",joint_positions)
        print("End Effector Pose \n",i,":",T0e)
        # T0es.append(T0e)
        T0e[2][3]+=i*0.05
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
    print(qs)