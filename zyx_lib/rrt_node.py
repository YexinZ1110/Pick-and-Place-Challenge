import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self, q):
        self.q=q
        self.parent=None
        self.children=[]

    def find_closest_node(self,sample):
        """
        :param sample:  (1,7) sample configuration
        """
        node=self
        dist=np.linalg.norm(self.q-sample)
        for child in self.children:
            closest,d=child.find_closest_node(sample)
            if d<dist:
                dist=d
                node=closest

        return node,dist
    
    def step_forward(self,target, step_size):
        dir=(target-self.q)
        dir=dir/np.linalg.norm(dir)
        new_node=Node(self.q+step_size*dir)
        return new_node

    def add_node(self, node):
        node.parent=self
        self.children.append(node)

    def get_path(self):
        path=[]
        current=self
        while current:
            path.insert(0,current.q)
            current=current.parent
        return path
        

    def plot_tree(node, ax, parent_position=None):
        """
        Recursively plot a tree of nodes.

        :param node: The current node in the tree
        :param ax: The matplotlib axes object where the tree is to be plotted
        :param parent_position: The position of the parent node
        """
        current_position = node.q[:2]  # Assume only the first two dimensions for plotting
        if parent_position is not None:
            ax.plot([parent_position[0], current_position[0]], [parent_position[1], current_position[1]], 'ro-')

        for child in node.children:
            Node.plot_tree(child, ax, current_position)

    def draw_tree(root):
        """
        Initialize the plotting of a tree.

        :param root: The root node of the tree
        """
        fig, ax = plt.subplots()
        Node.plot_tree(root, ax)
        ax.set_aspect('equal')
        plt.show()