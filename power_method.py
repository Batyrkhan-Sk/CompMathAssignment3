import numpy as np

def power_method(A, v0, num_iterations: int):
    b_k = v0.copy()
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    return b_k


def estimate_eigenvalue(A, eigenvector):
    return (np.dot(eigenvector.T, np.dot(A, eigenvector))) / np.dot(eigenvector.T, eigenvector)

if __name__ == '__main__':
    A = np.array([[2, -1, 0],
                  [-1, 2, -1],
                  [0, -1, 2]])
    v0 = np.array([1, 0, 0])
    num_iterations = 100

    dominant_eigenvector = power_method(A, v0, num_iterations)
    dominant_eigenvalue = estimate_eigenvalue(A, dominant_eigenvector)

    print("Dominant Eigenvalue:", dominant_eigenvalue)
    print("Dominant Eigenvector:", dominant_eigenvector)