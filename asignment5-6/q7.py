import numpy as np

def gauss_jordan(A, b):
    # Augment the matrix A with b
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))
    rows, cols = augmented_matrix.shape

    for i in range(rows):
        # Make the diagonal element 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Make the elements above and below the pivot 0
        for j in range(rows):
            if i != j:
                augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[j, i] * augmented_matrix[i]

    # Extract the solution
    return augmented_matrix[:, -1]

# Coefficient matrix A and vector b
A = np.array([
    [0, 1, 2, 3],
    [7, 6, 5, 4],
    [8, 9, 0, 1],
    [2, 5, 4, 3]
], dtype=float)
b = np.array([9, 33, 27, 23], dtype=float)

# Solve the system
solution = gauss_jordan(A, b)
print("Solution:", solution)
