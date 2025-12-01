import numpy as np

division_lines_count = 30


def run_assignment_2():
    print("Assignment 2 Results")

    # Stress tensor from problem statement
    sigma = np.array([
        [10, -5, 20],
        [-5, 0, 0],
        [20, 0, 10]
    ])
    print("σ =\n", sigma)

    # (a) Principal stresses and principal directions
    eigvals, eigvecs = np.linalg.eig(sigma)
    idx = np.argsort(eigvals)[::-1]  # sort descending for clarity
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    print("\n(a) Principal stresses =", eigvals)
    print("\nPrincipal directions (eigenvectors) =\n", eigvecs)

    # (b) Maximum shear stress
    sigma_max, sigma_min = np.max(eigvals), np.min(eigvals)
    tau_max = 0.5 * (sigma_max - sigma_min)
    print("\n(b) Maximum shear stress =", tau_max)

    # (c) Deviatoric stress and mean normal stress
    sigma_m = np.trace(sigma) / 3
    I = np.identity(3)
    sigma_dev = sigma - sigma_m * I
    print("\n(c) Mean normal stress =", sigma_m)
    print("\nDeviatoric stress tensor σ' =\n", sigma_dev)


if __name__ == "__main__":
    run_assignment_2()
