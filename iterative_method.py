import numpy as np

def iterative_method(A, B0, max_iterations=100, tolerance=1e-6):
    Bk = B0.copy()
    I = np.eye(A.shape[0])

    for i in range(max_iterations):
        E = np.dot(A, Bk) - I
        Bk_new = np.dot(Bk, (I - E + np.dot(E, E)))
        if np.linalg.norm(E) < tolerance:
            return Bk_new
        Bk = Bk_new
    print("Iterative inverse did not converge within max iterations.")
    return None

if __name__ == '__main__':
    A = np.array([[1, 10, 1],
                  [2, 0, 1],
                  [3, 3, 2]])
    B = np.array([[0.4, 2.4, -1.4],
                  [0.14, 0.14, -0.14],
                  [-0.85, -3.8, 2.8]])

    max_iterations = 100
    tolerance = 1e-6

    refined_inverse = iterative_method(A, B, max_iterations, tolerance)

    if refined_inverse is not None:
        print("Refined Inverse:")
        print(refined_inverse)