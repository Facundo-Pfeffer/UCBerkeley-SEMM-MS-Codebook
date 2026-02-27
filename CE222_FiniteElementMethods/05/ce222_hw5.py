import numpy as np
import matplotlib.pyplot as plt
import math
from time import perf_counter


N1 = lambda xi: 0.50*xi*(xi-1)
N2 = lambda xi: -(xi**2-1)
N3 = lambda xi: 0.50*xi*(xi+1)
N_eval = lambda xi: np.array([[N1(xi)], [N2(xi)], [N3(xi)]])

dN1 = lambda xi: xi-0.50
dN2 = lambda xi: -2 * xi
dN3 = lambda xi: xi+0.50
B_eval = lambda xi: np.array([[dN1(xi)], [dN2(xi)], [dN3(xi)]])
# Define variables of the problem, domain and other parameters as needed
L = 10  # length of domain
AE = 100.0 # Stiffness
u0 = 0.0  # boundary values
uL = -0.10  # boundary values


def b(x):
    return math.sin(4 * math.pi * x / L)  # distributed loading function

# Create datastructures to define the mesh
N = 2  # number of elements, in this case we will have three nodes per element.
n = 3 # number of nodes per element
p = n - 1 # polynomial order
x = np.linspace(0, L, N+1+(n-2)*N)  # Nodal coordinates
total_node_number = N*p+1
base = p * np.arange(N, dtype=int)        # 0, p, 2p, ...
location_matrix = np.vstack([base + j for j in range(n)])  # shape (n, N)

# Define arrays to specify essential boundary conditions and the values
bc_ess = [0, total_node_number-1]  # Zero based indexing, node numbers for ess bcs
bc_val = [u0, uL]  # The values at the ess nodes

# Generate the exact solution for plotting
xtrue = np.linspace(0, L, 200)
utrue = (L**2/(16*np.pi**2*AE))*np.sin(4*np.pi*xtrue/L) - 0.1*(xtrue/L)


def assemble_K(x, location_matrix, total_node_number, AE):
    K = np.zeros((total_node_number, total_node_number))

    for element_id in np.arange(location_matrix.shape[1]): # i element index
        h = x[location_matrix[2, element_id]] - x[location_matrix[0, element_id]]
        J = h/2
        B1 = B_eval(1/math.sqrt(3))
        B2 = B_eval(-1/math.sqrt(3))
        klocal = (AE / J) * (B1 @ B1.T + B2 @ B2.T)  # Using gaussian quadrature
        rows = location_matrix[:, element_id]
        cols = rows
        K[np.ix_(rows, cols)] = K[np.ix_(rows, cols)] + klocal

    return K


def assemble_F(x, location_matrix, total_node_number):
    F = np.zeros(total_node_number)

    gauss_pts = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
    gauss_wts = np.array([1.0, 1.0])

    for i in np.arange(location_matrix.shape[1]):
        rows = location_matrix[:, i]
        x_e = x[rows]

        flocal = np.zeros(rows.size)
        for xi, w in zip(gauss_pts, gauss_wts):
            Nq = N_eval(xi).ravel()
            Bq = B_eval(xi).ravel()
            xq = float(Nq @ x_e)
            Jq = float(Bq @ x_e)
            flocal += w * Nq * b(xq) * Jq

        F[rows] = F[rows] + flocal

    return F


def solve_system(K, F, bc_ess, bc_val):
    u = np.zeros(K.shape[0])

    u[bc_ess] = bc_val  # Set essential boundary values

    # Create mask for active degrees of freedom
    mask = np.ones(K.shape[0], dtype=bool)
    mask[bc_ess] = False

    u[mask] = np.linalg.solve(K[np.ix_(mask, mask)],
                              F[mask] - K[np.ix_(mask, bc_ess)] @ bc_val)

    return u


def time_call(fn, repeats=50):
    t0 = perf_counter()
    out = None
    for _ in range(repeats):
        out = fn()
    t1 = perf_counter()
    return (t1 - t0) / repeats, out


repeats = 200  # increase if N is small and timings are noisy

tK, K = time_call(lambda: assemble_K(x, location_matrix, total_node_number, AE), repeats=repeats)
tF, F = time_call(lambda: assemble_F(x, location_matrix, total_node_number), repeats=repeats)
tS, u = time_call(lambda: solve_system(K, F, bc_ess, bc_val), repeats=max(10, repeats // 10))

print(f"K assembly time (avg): {tK*1e3:.6f} ms")
print(f"F assembly time (avg): {tF*1e3:.6f} ms")
print(f"Solve time (avg):      {tS*1e3:.6f} ms")
print(f"Total (avg):           {(tK+tF+tS)*1e3:.6f} ms")

def fem_curve(x, location_matrix, u, n_per_elem=50):
    xis = np.linspace(-1.0, 1.0, n_per_elem)
    x_plot = []
    u_plot = []

    for e in range(location_matrix.shape[1]):
        rows = location_matrix[:, e]
        x_e = x[rows]
        u_e = u[rows]

        for k, xi in enumerate(xis):
            if e > 0 and k == 0:
                continue  # avoid duplicating shared node
            N = N_eval(xi).ravel()
            x_plot.append(float(N @ x_e))
            u_plot.append(float(N @ u_e))

    return np.array(x_plot), np.array(u_plot)


x_fem, u_fem = fem_curve(x, location_matrix, u, n_per_elem=80)

plt.plot(x_fem, u_fem, '-', label='FEA (quadratic interp)', color='red')
plt.plot(x, u, 's', label='FEA nodes', color='red')
plt.plot(xtrue, utrue, label='exact', color='blue')
plt.legend()
plt.show()