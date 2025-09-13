import json
import numpy as np
import sys
import argparse

from truss_elements import Node, Element, DOF
from plot_truss import plot_elements_plotly


# --- Parse command-line argument ---
parser = argparse.ArgumentParser(description="Solve a statically determinate truss from JSON input.")
parser.add_argument(
    "input_file",
    nargs="?",  # makes it optional
    default="input_json_files/HW1.json", # Debugging
    help="Path to the input JSON file (default: input.json)"
)
args = parser.parse_args()



with open("input_json_files/HW1.json") as f:
    data = json.load(f)

nodes_list = [Node(**n) for n in data["nodes"]]
elements_list = [Element(nodes_list[e[0]-1], nodes_list[e[1]-1]) for e in data["elements"]]



free_dof_list = []
restrained_dof_list = []
for node in nodes_list:
    if node.x_restrained is False:
        free_dof_list.append(DOF(node, 1))
    else:
        restrained_dof_list.append(DOF(node, 1, is_free_dof=False))
    if node.y_restrained is False:
        free_dof_list.append(DOF(node, 2))
    else:
        restrained_dof_list.append(DOF(node, 2, is_free_dof=False))



n_q = len(elements_list)



for node in nodes_list:  # Intializing element forces to zero for all nodes
    node.element_force_list = [(0,0)] * n_q

for element in elements_list: # Adding element forces to nodes
    element.add_element_force_contribution()


n_f = len(free_dof_list)
n_d = len(restrained_dof_list)

# Define P from JSON
P = np.zeros((n_f, 1))
for load in data.get("loads", []):
    node_id = load["node"]
    for free_node in free_dof_list:
        if free_node.node.id == node_id and free_node.direction == load["direction"]:
                P[free_node.f_dof_id-1] = load["value"]

R = np.zeros((n_d, 1))

def get_B_matrix():
    rows_list = []
    for free_dof in free_dof_list:
        node = free_dof.node
        direction = free_dof.direction - 1
        row = [0] * n_q
        for i, element_force  in enumerate(node.element_force_list):
            row[i] = element_force[direction]
        rows_list.append(row)
    return np.array(rows_list)


def get_Bd_matrix():
    rows_list = []
    for restrained_dof in restrained_dof_list:
        node = restrained_dof.node
        direction = restrained_dof.direction - 1
        row = [0] * n_q
        for i, element_force  in enumerate(node.element_force_list):
            row[i] = element_force[direction]
        rows_list.append(row)
    return np.array(rows_list)


def verify_equilibrium(P, R):
    sumM = 0
    sumFx = 0
    sumFy = 0

    for i, free_dof in enumerate(free_dof_list):
        if P[i] != 0:
            x = free_dof.node.x
            y = free_dof.node.y
            Fx, Fy = 0, 0
            if free_dof.direction == 1:
                Fx = P[i]
            else:
                Fy= P[i]
            sumFx += Fx
            sumFy += Fy
            sumM += -Fx*y + Fy*x

    for j, restrained_dof in enumerate(restrained_dof_list):
        if R[j] != 0:
            x = restrained_dof.node.x
            y = restrained_dof.node.y
            Fx, Fy = 0, 0
            if restrained_dof.direction == 1:
                Fx = R[j]
            else:
                Fy= R[j]
            sumFx += Fx
            sumFy += Fy
            sumM += -Fx*y + Fy*x

    print((sumFx, sumFy, sumM))
    tolerance = 1e-3
    return all(abs(x.item()) < tolerance for x in (sumFx, sumFy, sumM))



if __name__ == "__main__":
    plot_elements_plotly(nodes_list, elements_list, free_dof_list)
    print(f"P = {P}")
    B = get_B_matrix()
    print(f"Bf = {B}")
    B_inverse = np.linalg.inv(B)
    np.set_printoptions(precision=3, suppress=True)
    print(f"Bf inverse = {B_inverse}")
    Q = B_inverse @ P
    print(f"Q = {Q}")
    Bd = get_Bd_matrix()
    print(f"Bd = {Bd}")
    R = Bd @ Q
    print(f"R = {R}")
    print(verify_equilibrium(P, R))
