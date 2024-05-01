from lib.calculateFK import FK
from lib.IK_position_null import IK
import numpy as np
def ik_cal(T0e, seed,alpha):
    ik = IK()
    q_t, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T0e, seed, method='J_pseudo', alpha=.5)
    print(message_pseudo)
    return q_t
def fk_cal(q):
    fk = FK()
    _, T0e = fk.forward(q)
    return T0e
def q_stack_calc(q,h,n):
    """
    :param q: 1x7 vector, configuration of seed 
    :param h: int, stack height
    :param n: int, stack number
    :return qs: stack configuration
    """
    ik = IK()
    fk = FK()
    q_stack=[q]
    _, T0e = fk.forward(q)
    seed=q
    print_q(q)
    for i in range(n):
        T0e[2][3]=T0e[2][3]+h
        q_t, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T0e, np.array(seed), method='J_pseudo', alpha=.5)
        q_stack.append(q_t)
        print_q(q_t)
        seed=q_t
    return np.array(q_stack)
def main(first_block_stack):
    h=0.055
    n=3
    q = np.array(first_block_stack)
    q_stack=q_stack_calc(q,h,n)
    # print("Stack as this configuration:\n")
def print_q(q):
    print("[",q[0],",",q[1],",",q[2],",",q[3],",",q[4],",",q[5],",",q[6],"]")

static_stand =[ 0.33265238,  0.17196839,  0.17188321 ,-1.22208596 ,-0.02974835 , 1.39164772 , 1.28214468]
T0e=fk_cal(static_stand)
print(T0e)
T0e[1][3]=T0e[1][3]-0.1
T0e[2][3]=T0e[2][3]
q=ik_cal(T0e,static_stand,0.5)
print_q(q)

