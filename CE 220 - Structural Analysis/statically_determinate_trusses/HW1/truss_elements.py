import numpy as np

node_id_counter = 0
element_id_counter = 0


class Node:
    def __init__(self, x, y, x_restrained=False, y_restrained=False, id=None):
        self.id = id
        self.id = self.get_id()
        self.x = x
        self.y = y
        self.x_restrained = x_restrained
        self.y_restrained = y_restrained
        self.element_force_list = []

    def get_id(self):
        if self.id is None:
            global node_id_counter
            node_id_counter += 1
            return node_id_counter
        else:
            return self.id

free_dof_counter = 0
class FreeDOF:
    def __init__(self, node, direction: int, dof_id=None):
        self.node = node
        self.direction = direction
        self.f_dof_id =self.get_dof_id(dof_id)

    def get_dof_id(self, dof_id):
        if dof_id is not None:
            return dof_id
        global free_dof_counter
        free_dof_counter += 1
        return free_dof_counter


class Element:
    def __init__(self, initial_node, final_node, id=None):
        self.id = id
        self.id = self.get_id()
        self.initial_node = initial_node
        self.final_node = final_node
        self.length = np.sqrt((initial_node.x - final_node.x) ** 2 + (initial_node.y - final_node.y) ** 2)
        self.dx = final_node.x - initial_node.x
        self.dy = final_node.y - initial_node.y
        self.x_unit_comp = self.dx / self.length
        self.y_unit_comp = self.dy / self.length
        self.name = str(f"{initial_node.id}-{final_node.id}")

    def get_id(self):
        if self.id is None:
            global element_id_counter
            element_id_counter += 1
            return element_id_counter
        else:
            return self.id

    def add_element_force_contribution(self):
        self.initial_node.element_force_list[self.id-1] = (-self.x_unit_comp, -self.y_unit_comp)
        self.final_node.element_force_list[self.id-1] = (self.x_unit_comp, self.y_unit_comp)