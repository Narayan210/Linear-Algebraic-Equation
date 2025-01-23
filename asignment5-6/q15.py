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

# Coefficient matrix A
A = np.array([
    [3, 5, -3, 1],
    [2, 1, 1, 4],
    [3, -4, 0, -1],
    [2, 1, -3, 9]
], dtype=float)

# Right-hand side vector b
b = np.array([16, 9, 1, 5], dtype=float)

# Solve using Gauss-Jordan elimination
solution = gauss_jordan_elimination(A, b)

# Display the solution
print("Solution:", solution)
