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

def inverse_via_gauss_jordan(matrix):
    """
    Find the inverse of a matrix using Gauss-Jordan elimination.

    Parameters:
        matrix (numpy.ndarray): The input square matrix.

    Returns:
        numpy.ndarray: Inverse of the matrix.
    """
    n = matrix.shape[0]
    aug_matrix = np.hstack([matrix, np.eye(n)])  # Augmented matrix with identity matrix

    for i in range(n):
        # Make the diagonal element 1
        aug_matrix[i] = aug_matrix[i] / aug_matrix[i, i]

        # Make all other elements in the column 0
        for j in range(n):
            if i != j:
                aug_matrix[j] = aug_matrix[j] - aug_matrix[j, i] * aug_matrix[i]

    return aug_matrix[:, n:]  # Extract the inverse matrix

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

# Define the matrix for eigenvalue calculation
matrix = np.array([
    [1, 2, 0],
    [2, 1, 0],
    [0, 0, -1]
], dtype=float)

# Find the largest eigenvalue and corresponding eigenvector
def largest_eigenvalue_and_vector(matrix):
    """
    Find the largest eigenvalue and the corresponding eigenvector of a matrix.

    Parameters:
        matrix (numpy.ndarray): The input square matrix.

    Returns:
        tuple: Largest eigenvalue and corresponding eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    max_index = np.argmax(eigenvalues)
    return eigenvalues[max_index], eigenvectors[:, max_index]

eigenvalue, eigenvector = largest_eigenvalue_and_vector(matrix)

print("Largest Eigenvalue:", eigenvalue)
print("Corresponding Eigenvector:", eigenvector)

# Solve another system using Gauss-Jordan method
A_new = np.array([
    [2, -1, 4],
    [2,  3, -2],
    [3,  2, -4]
], dtype=float)

b_new = np.array([15, 4, -4], dtype=float)

solution_new = gauss_jordan(A_new, b_new)

print("Solution to the new system:", solution_new)

# Find the inverse of a matrix using Gauss-Jordan method
matrix_to_invert = np.array([
    [2, -1, 4],
    [2,  3, -2],
    [3,  2, -4]
], dtype=float)

inverse_matrix = inverse_via_gauss_jordan(matrix_to_invert)

print("Inverse of the matrix:")
print(inverse_matrix)
