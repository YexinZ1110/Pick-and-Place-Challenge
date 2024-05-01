    
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
    T0e[2][3]=T0e[2][3]+0.005
    for i in range(n):
        T0e[2][3]=T0e[2][3]+h
        # print("T0e",T0e)
        q_t, rollout_pseudo, success_pseudo, message_pseudo = ik.inverse(T0e, np.array(seed), method='J_pseudo', alpha=.5)
        q_stack.append(q_t)
        print_q(q_t)
        seed=q_t

    return np.array(q_stack)


def main(first_block_stack):
    h=0.05
    n=7

    q = np.array(first_block_stack)
    q_stack=q_stack_calc(q,h,n)
    # print("Stack as this configuration:\n")

def print_q(q):
    q=np.round(q,5)
    print("[",q[0],",",q[1],",",q[2],",",q[3],",",q[4],",",q[5],",",q[6],"]")



## blue original pos
# start=np.array([8.98141289e-06, -7.85005888e-01,-2.81193385e-05, -2.35601398e+00,-8.11740800e-06,  1.57001310e+00,  7.85002585e-01])
# print("origin state: ",fk_cal(start))
# T0e=np.array([[1,0,0,0.52],[0,-1,0,0.2],[0,0,-1,0.47],[0,0,0,1]])
# q=ik_cal(T0e,start,0.5)
# print_q(q)


## static_standy = [0.14073182017042138 , -0.016202042086755374 , 0.22430339736949562 , -1.722431588772413 , 0.003637551798808322 , 1.706634383411021 , 1.1499123550701502]

# black center
# pickup location of first block
# first_block=[ 3.59927194e-02, -1.26240268e-03,  1.97398241e-01, -2.35900199e+00,
#   3.51082039e-04,  2.35777170e+00,  2.89730026e+00]
# T0e=fk_cal(first_block)
# T0e=np.array([[1,0,0,0.57],[0,-1,0,-0.18],[0,0,-1,0.25],[0,0,0,1]])
# q=ik_cal(T0e,first_block,0.5)

# # # blue desk 1st position
# main(q)
# static_stand = [0.14073182017042138 , -0.016202042086755374 , 0.22430339736949562 , -1.722431588772413 , 0.003637551798808322 , 1.706634383411021 , 1.1499123550701502]
# T0e=fk_cal(static_stand)
# print(T0e)
# T0e[2][3]=T0e[2][3]+0.1
# q=ik_cal(T0e,static_stand,0.5)
# print_q(q)


# # dynamic block obs position
# dynamic_obs=[ 0.20521129491312645 , 0.032997411691558634 , 0.21157256832206484 , -1.7148264863939717 , -0.007043809736550042 , 1.7470857641033648 , 1.2033066537853696 ]
# T0e=fk_cal(dynamic_obs)
# print(T0e)

# T0e[2][3]=T0e[2][3]+0.05
# q=ik_cal(T0e,dynamic_obs,0.5)
# print_q(q)

# # stack position
static_obs=[ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517]
T0e=fk_cal(static_obs)
print(T0e)
T0e[0][3]=T0e[0][3]+0.05
T0e[1][3]=T0e[1][3]-0.05

q=ik_cal(T0e,static_obs,0.5)
# print_q(q)

main(q)

# # red desk 1st position

# static_stand = [ 0.14028 , 0.04232 , 0.05282 , -2.2751 , -0.00302 , 2.31737 , 0.98051 ]
# main(static_stand)