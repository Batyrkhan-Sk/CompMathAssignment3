import numpy as np

def jacobi_method_example(A, tolerance=1e-8, max_iterations=100):
    n = A.shape[0]
    A = A.astype(float)
    eigenvectors = np.eye(n)
    
    for _ in range(max_iterations):
        off_diag_max = 0.0
        p, q = 0, 0
        
        for i in range(n):
            for j in range(i + 1, n):
                if abs(A[i, j]) >= off_diag_max:
                    off_diag_max = abs(A[i, j])
                    p, q = i, j
        
        if off_diag_max < tolerance:
            break
            
        if A[p, p] == A[q, q]:
             theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A[p, q] / (A[q, q] - A[p, p]))
        
        J = np.eye(n)
        J[p, p] = np.cos(theta)
        J[q, q] = np.cos(theta)
        J[p, q] = -np.sin(theta)
        J[q, p] = np.sin(theta)
        
        A = np.dot(np.dot(J.T, A), J)
        eigenvectors = np.dot(eigenvectors, J)
        
    eigenvalues = np.diag(A)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    return eigenvalues, eigenvectors

if __name__ == '__main__':
    A = np.array([[1, np.sqrt(2), 2],
                  [np.sqrt(2), 3, np.sqrt(2)],
                  [2, np.sqrt(2), 1]])
    
    eigenvalues, eigenvectors = jacobi_method_example(A)
    
    np.set_printoptions(precision=8, suppress=True)

    print("Eigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)