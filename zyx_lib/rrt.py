import numpy as np
import random
from copy import deepcopy
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from lib.calculateFK import FK
from lib.rrt_node import Node
# from detectCollision import detectCollision
# from loadmap import loadmap
# from calculateFK import FK
# from rrt_node import Node

def detectRobotCollision(map, joint_pos):
    '''
    Check if the robot will collide with the obstacles
    '''
    obs = np.array(map.obstacles)
    if obs.size == 0:
        return False
    # print(obs.reshape(-1))
    # print("shape:", obs.reshape(-1).shape)
    collision_vec = detectCollision(joint_pos[1:], joint_pos[:-1], obs.reshape(-1))
    if any(collision_vec):
        return True
    return False

def detectPathCollision(map, start, goal, step_size):
    '''
    Check if the path between two configurations will collide with the obstacles
    '''
    fk = FK()
    dist = goal - start
    num_step = int(np.linalg.norm(dist) / step_size)
    for i in range(num_step):
        new_node = start + i * step_size * dist
        new_node_pos, _ = fk.forward(new_node)
        if detectRobotCollision(map, new_node_pos):
            return True
    return False

def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []
    iter_num=10000
    eps=1e-2
    step_size=0.1

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])
    
    # Initialization:
    fk = FK()
    root = Node(start)
    obstacles = map.obstacles
    
    # we need to check 1. if the robot will collide with the obstacles
    #                  2. if the path between two configurations will collide with the obstacles
    
    start_pos, _ = fk.forward(start)
    goal_pos, _ = fk.forward(goal)
    start_collide = detectRobotCollision(map, start_pos)
    goal_collide = detectRobotCollision(map, goal_pos)
    if start_collide or goal_collide:
        return np.array(path)
    
    for _ in range(iter_num):
        # q_rand = random configuration in Q_free
        q_rand = np.random.uniform(lowerLim, upperLim)
        # q_rand_pos, _ = fk.forward(q_rand)
        # q_a = closest node in start configuration
        q_a, _ = root.find_closest_node(q_rand)
        
        # If NOT collide(q_rand, q_a):
        #     Add (q_rand, q_a) to start configuration
        if not detectPathCollision(map, q_rand, q_a.q, step_size):
            # q_b = closest node in goal configuration
            q_b = Node(q_rand)
            q_a.add_node(q_b)
            # If NOT collide(q_rand, q_b):
            # Add (q_rand, q_b) to goal configuration
            if not detectPathCollision(map, q_rand, goal, step_size):
                q_goal = Node(goal)
                q_b.add_node(q_goal)
                path = q_goal.get_path()
                # If q_rand is connected to start configuration and goal configuration:
                # break
                break
        
    return np.array(path)

if __name__ == '__main__':
    # map_struct = loadmap("../maps/map1.txt")
    map_struct = loadmap("/Users/yeshuchen/Desktop/MEAM5200/Lab4/meam520_labs/maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print(len(path))