import numpy as np
import matplotlib.pyplot as plt
import math

# Define variables of the problem, domain and other paramters as needed
L = 10  # length of domain
AE = 100.0 # Stiffness
u0 = 0.0  # boundary values
uL = -0.10  # boundary values


def b(x):
    return math.sin(4 * math.pi * x / L)  # distributed loading function

# Create datastructures to define the mesh
N = 10  # number of elements, might change to generate different plots.
x = np.linspace(0, L, N+1)  # Nodal coordinates
lm = np.vstack((
    np.arange(0, N, dtype=int),
    np.arange(1, N+1, dtype=int)
))

# Define arrays to specify essential boundary conditions and the values
bc_ess = [0, N]  # Zero based indexing, node numbers for ess bcs
bc_val = [u0, uL]  # The values at the ess nodes

# Generate exact solution for plotting
xtrue = np.linspace(0, L, 200)
utrue = (L**2/(16*np.pi**2*AE))*np.sin(4*np.pi*xtrue/L) - 0.1*(xtrue/L)

# Loop over elements to construct K matrix
K = np.zeros((N + 1, N + 1))
for i in np.arange(N): # i element index
    h = x[i+1] - x[i]
    klocal = (AE / h) * np.array([[1.0, -1.0],
                                  [-1.0, 1.0]])
    rows = lm[:, i]
    cols = rows
    K[np.ix_(rows, cols)] = K[np.ix_(rows, cols)] + klocal

# Loop over elements, contstruct load vector F using Gauss Quadrature
F = np.zeros(N + 1)
gauss_pts = np.array([-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)])
gauss_wts = np.array([1.0, 1.0])

for i in np.arange(N):
    h = x[i+1] - x[i]
    xmid = 0.5 * (x[i] + x[i+1])

    flocal = np.zeros(2)
    for xi, w in zip(gauss_pts, gauss_wts):
        N1 = 0.5 * (1.0 - xi)
        N2 = 0.5 * (1.0 + xi)
        xq = xmid + 0.5 * h * xi
        flocal += w * np.array([N1, N2]) * b(xq) * (0.5 * h)

    rows = lm[:, i]
    F[rows] = F[rows] + flocal

# Solve the equations accounting for the essential boundary conditions
u = np.zeros(N + 1)

u[bc_ess] = bc_val  # Set essential boundary values

# Create mask for active degrees of freedom
mask = np.ones(N + 1, dtype=bool)
mask[bc_ess] = False

u[mask] = np.linalg.solve(K[np.ix_(mask, mask)],
                          F[mask] - K[np.ix_(mask, bc_ess)] @ bc_val)

# Plot FEA and exact solutions
plt.plot(x, u, 's-', label='FEA', color='red')
plt.plot(xtrue, utrue, label='exact', color='blue')
plt.legend()
plt.show()