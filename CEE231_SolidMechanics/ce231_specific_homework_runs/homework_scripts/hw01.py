import numpy as np

division_lines_count = 30

def run_assignment_2():
    v = np.array([3, 4, 0])
    w = np.array([0, 2, 2])

    z = np.cross(w, v)

    s1 = np.dot(v, w)
    s2 = np.dot(v, z)

    print("Assignment 2 Results\n" + "-" * division_lines_count)
    print("a) Cross product z = w × v")
    print(f"\tz₂ = {z[1]}")
    print(f"\tz₃ = {z[2]}\n")

    print("b) Scalar products")
    print(f"\ts₁ = v · w = {s1}")
    print(f"\ts₂ = v · z = {s2}")

S = np.array([  # Defining S as a variable in the global scope bc is used in #3 and #5
    [1, 3, 2],
    [3, 0, 1],
    [2, 1, 0]
])

def run_assignment_3():

    v = np.array([1, 2, 3])

    Sv = S @ v
    vTSv = v @ S @ v
    S_double_dot_S = np.sum(S * S)
    SS = S @ S
    skw_S_def = 0.5 * (S - S.T)

    print("Assignment 3 Results")
    print("-" * division_lines_count)

    print("(a) Sv = {Sv}\n")

    print(f"(b) v · Sv ={vTSv}")

    print(f"(c) S : S = {S_double_dot_S}\n")

    print("(d) SS = S · S")
    print("=", SS, "\n")

    print(f"(e) skw(S)=\n{skw_S_def} (by definition).")

def run_assignment_5():
    eigvals, eigvecs = np.linalg.eig(S)
    print("-" * division_lines_count)
    print("Assignment 5 Results")

    print("Eigenvalues:\n", eigvals)
    print("Normalized eigenvectors (columns):\n", eigvecs)

    dot_products_list = [
        np.dot(eigvecs[index], eigvecs[index+1 if index+1 <3 else 0]) for index in range(3)]
    print("Are eignenvectors orthogonal:\n", "Yes" if all([round(x,3) ==0 for x in dot_products_list]) else "No")

if __name__ == "__main__":
    run_assignment_2()
    run_assignment_3()
    run_assignment_5()

