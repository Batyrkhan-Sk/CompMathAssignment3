import numpy as np

def lu_decomposition(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()

    for i in range(n):
        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, :] -= factor * U[i, :]
    return L, U

def solve_linear_system(L, U, b):
    n = L.shape[0]
    y = np.zeros_like(b, dtype=float)
    x = np.zeros_like(b, dtype=float)
    
    y[0] = b[0]
    for i in range(1, n):
       y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def matrix_inverse_lu(A):
    n = A.shape[0]
    L, U = lu_decomposition(A)
    A_inv = np.zeros_like(A, dtype=float)
    identity = np.eye(n)

    for i in range(n):
      A_inv[:, i] = solve_linear_system(L, U, identity[:, i])
    return A_inv


A = np.array([[50, 107, 36],
              [35, 54, 20],
              [31, 66, 21]])

A_inv = matrix_inverse_lu(A)

print("Original matrix A:\n", A)
print("\nInverse matrix \n", A_inv)