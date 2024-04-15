    
from lib.calculateFK import FK
from lib.IK_position_null import IK
import numpy as np


def q_stack_calc(q,h):
    """
    :param1 q: nx7 vector, configuration of seed 
    :paraml h: int, stack height
    :return qs: stack configuration
    """
    np.set_printoptions(suppress=True,precision=5)

    ik = IK()
    fk = FK()

    q_stack=[]
    _, T0e = fk.forward(q[0])
    for i in range(q.shape[0]):
        T0e[2][3]=T0e[2][3]+h
        q_t, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T0e, np.array(q[i]), method='J_pseudo', alpha=.5)
        q_stack.append(q_t)

    return np.array(q_stack)


def main():
    h=0.03
    q = np.array([
        [ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517] ,
        [ 0.1986,  0.1423,  0.0645, -2.0026, -0.0109,  2.1445,  1.0538] ,
        [ 0.1755,  0.0966,  0.0882, -1.9295, -0.0095,  2.0257,  1.0528] ,
        [ 0.1619,  0.0684,  0.102 , -1.8359, -0.0074,  1.904 ,  1.0514] ,
        [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507] ,
        [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512] ,
        [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524] ,
        [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525] ,
        ])
    q_stack=q_stack_calc(q,h)
    print("Stack as this configuration:\n",q_stack)

main()