import numpy as np


def run_assignment_3():
    print("Assignment 5 Results: Stress State\n")

    # Strain vector (Eq. 7)
    eps_voigt = -1 * np.array([[1], [1], [1], [0], [0], [0]])  # take Δ = 1

    # Elasticity matrices (in GPa)
    C_Alb = np.array([
        [69.1, 34.0, 30.8, 5.1, -2.4, -0.9],
        [34.0, 183.5, 5.5, -3.9, -7.7, -5.8],
        [30.8, 5.5, 179.5, -8.7, 7.1, -9.8],
        [5.1, -3.9, -8.7, 24.9, -2.4, -7.2],
        [-2.4, -7.7, 7.1, -2.4, 26.8, 0.5],
        [-0.9, -5.8, -9.8, -7.2, 0.5, 33.5]
    ])

    C_Fe = np.array([
        [231.4, 134.7, 134.7, 0, 0, 0],
        [134.7, 231.4, 134.7, 0, 0, 0],
        [134.7, 134.7, 231.4, 0, 0, 0],
        [0, 0, 0, 116.4, 0, 0],
        [0, 0, 0, 0, 116.4, 0],
        [0, 0, 0, 0, 0, 116.4]
    ])

    # Compute σ_voigt = C_voigt * ε_voigt
    sigma_Alb = C_Alb @ eps_voigt
    sigma_Fe = C_Fe @ eps_voigt

    print("ε_voigt =Δ\n", eps_voigt)
    print("\nσ_voigt^Albite (GPa) =Δ\n", sigma_Alb)
    print("\nσ_voigt^Fe (GPa) =Δ\n", sigma_Fe)


if __name__ == "__main__":
    run_assignment_3()
