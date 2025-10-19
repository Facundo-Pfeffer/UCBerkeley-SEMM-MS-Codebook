import numpy as np

def run_assignment_1():

    Af = (1 / np.sqrt(5)) * np.array([
        [0, 0, 0, 0, np.sqrt(5), 0, 0, 0, 0],
        [0, 2, 1, 0, 0, 0, 0, 0, 0],
        [2, -2, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, np.sqrt(5), 0, 0],
        [0, 2, -1, -2, 1, 0, 0, 0, 0],
        [0, 0, -np.sqrt(5), 0, 0, 0, 0, 0, np.sqrt(5)],
        [0, -2, -1, 0, 0, 2, 1, 0, 0],
        [0, 0, 0, -2, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, -1, 2, 1]
    ])

    # Thermal expansion
    alpha = 5e-6  # coefficient of thermal expansion [1/°C]
    delta_T = 150  # temperature change [°C]
    L = 10 * np.sqrt(5)  # element lengthn
    delta_L = alpha * delta_T * L

    V = np.array([
        0,
        delta_L,
        delta_L,
        0,
        0,
        0,
        0,
        0,
        0
    ]).reshape(-1, 1)
    print(V)
    U = np.linalg.inv(Af) @ V
    print("U =\n", U)


if __name__ == "__main__":
    run_assignment_1()