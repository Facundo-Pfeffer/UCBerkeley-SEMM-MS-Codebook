import numpy as np

division_lines_count = 30

def run_assignment_1():
    S = np.array([
        [1, 3, 4],
        [3, 2, 5],
        [4, 5, 3]
    ])

    e1_star = np.array([1, 0, 0])
    e2_star = np.array([0, np.cos(np.pi/4), np.sin(np.pi/4)])

    S12_star = e1_star @ S @ e2_star

    print("Assignment 1 Results\n" + "-" * division_lines_count)
    print(f"Transformed tensor component S*₁₂ = {S12_star:.4f}")

if __name__ == "__main__":
    run_assignment_1()
