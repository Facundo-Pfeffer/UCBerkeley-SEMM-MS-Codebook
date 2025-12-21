import numpy as np
L=10
EI = 10000
k0 = 0.001
def get_K():
    Af = np.array([[-1/(L/2), 1],
                  [1/(L/2), 1]])
    Ks = np.array([[3*EI/(L/2), 0],
                   [0, 3*EI/(L/2)]])
    K = Af.T @ Ks @ Af
    QO = np.array([-k0*EI, k0*EI])
    P0 = Af.T @ QO
    U = -np.linalg.inv(K) @ P0
    V = Af @ U
    Q =  Ks @ V + QO
    print(Q)


def problem_2():
    Af = np.array([[1/6,0,0,0,0],
                   [1 / 6, 1, 0, 0, 0],
                   [0, 1, 0, 0, -1/8],
                   [0, 0, 1, 0, -1/8],
                   [1/6, 0, 0, 1, -1/6],
                  ])
    Ks = 100000 * np.array(
        [[4/6,2/6,0,0,0],
        [2/6,4/6, 0, 0, 0],
         [0,0, 4/8, 2/8, 0],
         [0,0, 2/8, 4/8, 0],
         [0,0, 0, 0, 3/6]]
    )
    K = Af.T @ Ks @ Af
    Vd = np.array([0,0,0,0,-8*0.001])
    P0 = Af.T @ Ks @ Vd
    U = -np.linalg.inv(K) @ P0
    V = Af @ U
    Q =  Ks @ V
    print(Q)


import numpy as np

def problem_2_with_Vd_first(U5=0.001):
    A = np.array([
        [1/6, 0,   0, 0,    0   ],
        [1/6, 1,   0, 0,    0   ],
        [0,   1,   0, 0,   -1/8 ],
        [0,   0,   1, 0,   -1/8 ],
        [1/6, 0,   0, 1,   -1/6 ],
    ], dtype=float)

    Ks = 100000 * np.array([
        [4/6, 2/6, 0,   0,   0  ],
        [2/6, 4/6, 0,   0,   0  ],
        [0,   0,   4/8, 2/8, 0  ],
        [0,   0,   2/8, 4/8, 0  ],
        [0,   0,   0,   0,   3/6],
    ], dtype=float)

    # free dofs: U1..U4, prescribed dof: U5
    free = [0, 1, 2, 3]
    prescribed = [4]

    Af = A[:, free]              # 5x4
    Ad = A[:, prescribed]        # 5x1
    Ud = np.array([U5], float)   # 1x1

    # 1) obtain Vd first
    Vd = (Ad @ Ud).reshape(-1)   # basic deformations due only to support displacement

    # 2) solve for free displacements
    Kff = Af.T @ Ks @ Af
    P0 = Af.T @ Ks @ Vd
    Uf = np.linalg.solve(Kff, -P0)

    # 3) reconstruct full U, then V and Q
    U = np.zeros(5, float)
    U[free] = Uf
    U[prescribed] = Ud

    V = A @ U
    Q = Ks @ V

    print("Vd =", Vd)
    print("U  =", U)
    print("V  =", V)
    print("Q  =", Q)


if __name__ == "__main__":
    get_K()
    problem_2_with_Vd_first(U5=-0.001*8)