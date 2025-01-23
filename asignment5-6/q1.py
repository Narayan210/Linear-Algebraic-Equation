import numpy as np

def gauss_jordan(a, b):
    """
    Perform Gauss-Jordan elimination to solve the system of linear equations Ax = b.

    Parameters:
        a (numpy.ndarray): Coefficient matrix A.
        b (numpy.ndarray): Constant matrix b.

    Returns:
        numpy.ndarray: Solution vector x.
    """
    n = len(b)
    aug_matrix = np.hstack([a, b.reshape(-1, 1)])  # Augmented matrix

    for i in range(n):
        # Make the diagonal element 1
        aug_matrix[i] = aug_matrix[i] / aug_matrix[i, i]

        # Make all other elements in the column 0
        for j in range(n):
            if i != j:
                aug_matrix[j] = aug_matrix[j] - aug_matrix[j, i] * aug_matrix[i]

    return aug_matrix[:, -1]  # Solution vector x

# Define the coefficient matrix and the constants
A = np.array([
    [6, -2, 1],
    [1,  2, -5],
    [-2, 7,  2]
], dtype=float)

b = np.array([11, -1, 5], dtype=float)

# Solve the system
solution = gauss_jordan(A, b)

print("Solution:", solution)
