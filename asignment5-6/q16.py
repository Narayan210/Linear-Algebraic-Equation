import numpy as np

def gauss_jordan_elimination(A, b):
    # Augmenting the matrix A with vector b
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])
    n = len(b)

    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Make the diagonal element 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Eliminate the ith column elements of other rows
        for j in range(n):
            if j != i:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]

    return augmented_matrix[:, -1]

def dominant_eigenvalue_and_vector(matrix, tol=1e-9, max_iterations=1000):
    n = matrix.shape[0]
    vector = np.ones(n)  # Initial guess for eigenvector
    eigenvalue = 0

    for _ in range(max_iterations):
        next_vector = np.dot(matrix, vector)
        next_eigenvalue = np.max(np.abs(next_vector))
        next_vector = next_vector / next_eigenvalue

        if np.linalg.norm(next_vector - vector) < tol:
            eigenvalue = next_eigenvalue
            break

        vector = next_vector

    return eigenvalue, vector

# Coefficient matrix A
A = np.array([
    [3, 5, 5],
    [5, 5, 3],
    [2, 3, 2]
], dtype=float)

# Right-hand side vector b
b = np.array([16, 9, 1, 5], dtype=float)

# Solve using Gauss-Jordan elimination
solution = gauss_jordan_elimination(A, b)

# Find dominant eigenvalue and eigenvector
eigenvalue, eigenvector = dominant_eigenvalue_and_vector(A)

# Display the results
print("Solution:", solution)
print("Dominant Eigenvalue:", eigenvalue)
print("Corresponding Eigenvector:", eigenvector)
