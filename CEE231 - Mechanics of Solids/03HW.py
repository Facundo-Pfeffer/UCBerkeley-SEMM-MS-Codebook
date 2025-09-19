import numpy as np

division_lines_count = 30


def run_assignment_2():
    # Parameters
    a0 = 6.0  # Å
    a = 4.55  # Å
    b = 5.45  # Å
    c = 6.02  # Å
    beta = 87 * np.pi / 180  # radians

    # Cubic lattice vectors
    a1 = a0 * np.array([1, 0, 0])
    a2 = a0 * np.array([0, 1, 0])
    a3 = a0 * np.array([0, 0, 1])

    # Monoclinic lattice vectors
    m1 = a * np.array([
        np.cos(np.pi / 2 - beta) - np.sin(np.pi / 2 - beta),
        np.cos(np.pi / 2 - beta) + np.sin(np.pi / 2 - beta),
        0
    ])
    m2 = c * np.array([0, 1, 0])
    m3 = b * np.array([0, 0, 1])

    # Form matrices with vectors as columns
    A = np.column_stack((a1, a2, a3))
    M = np.column_stack((m1, m2, m3))

    # Obtain F from solving the equation AF = M

    F = M @ np.linalg.inv(A)
    print(f"F= {F}")
    # Compute C = F^T F
    C = F.T @ F
    print("\nC =\n", C)

    eigenvalues, eigenvectors = np.linalg.eig(C)
    print("\nEigenvalues =", eigenvalues)
    print("\nEigenvectors =\n", eigenvectors)

    # Spectral decomposition: C = Σ λ_i r_i ⊗ r_i
    U = np.zeros_like(C)

    for i in range(len(eigenvalues)):
        r = eigenvectors[:, i]
        U += np.sqrt(eigenvalues[i]) * np.outer(r, r)

    print("\nU (reconstructed from spectral decomposition of C) =\n", U)
    C_verification = U @ U

    # Check closeness
    if np.allclose(C, C_verification):
        print("\nVerification successful: U @ U ≈ C ✅")
    else:
        print("\nVerification failed: U @ U does not match C ❌")

    E_biot = U - np.identity(3)
    print("\nFINAL RESPONSE\nE_biot =U-1\n", E_biot, f"\nPrincipal strains:\n {[round(float(x)-1,2) for x in eigenvalues]}")


def run_assignment_3():
    print("Assignment 3 Results")

    # Strain tensor
    eps = np.array([
        [1.0, 2.0, 0.0],
        [2.0, 0.5, 0.0],
        [0.0, 0.0, -1.5]
    ])
    print("ε =\n", eps)

    # (a) Volumetric strain = trace(ε)
    vol_strain = np.trace(eps)
    print("\n(a) Volumetric strain =", vol_strain)

    # (b) Principal strains
    eigvals, eigvecs = np.linalg.eig(eps)
    idx_max, idx_min = np.argmax(eigvals), np.argmin(eigvals)
    print("\n(b) Max principal strain =", eigvals[idx_max], "dir =", eigvecs[:, idx_max])
    print("    Min principal strain =", eigvals[idx_min], "dir =", eigvecs[:, idx_min])

    # (c) Normal strain in n = (1,1,1)/√3
    n = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    nEn = n.T @ eps @ n
    print("\n(c) Normal strain (n) =", nEn)

    # (d) Tensorial shear strain e_nm = 2 nᵀ ε m
    m = np.array([0.0, 1.0, -1.0]) / np.sqrt(2)
    gamma_nm = (n.T @ eps @ m)
    gamma_mn = (m.T @ eps @ n)
    print("\n(d) Shear strain e_nm =", gamma_nm)
    print("Verifies e_mn=e_nm", np.allclose(gamma_nm, gamma_mn))


if __name__ == "__main__":
    run_assignment_2()
    run_assignment_3()

