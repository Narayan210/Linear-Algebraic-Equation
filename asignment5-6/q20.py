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

def lu_factorization(A, b):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.sum(L[i, :i] * U[:i, j])
        for j in range(i, n):
            if i == j:
                L[i, i] = 1
            else:
                L[j, i] = (A[j, i] - np.sum(L[j, :i] * U[:i, i])) / U[i, i]

    # Forward substitution to solve Ly = b
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - np.sum(L[i, :i] * y[:i])

    # Backward substitution to solve Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.sum(U[i, i + 1:] * x[i + 1:])) / U[i, i]

    return x

def gauss_elimination_partial_pivoting(A, b):
    n = len(b)
    A = A.astype(float)
    b = b.astype(float)

    for i in range(n):
        # Partial pivoting
        max_row = i + np.argmax(np.abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x

# Coefficient matrix A
A = np.array([
    [-1, 2, 3, -1],
    [2, 4, 1, 2],
    [-3, 8, 4, -1],
    [1, 4, 72, -2]
], dtype=float)

# Right-hand side vector b
b = np.array([3, -1, 6, -4], dtype=float)

# Solve using Gauss Elimination with Partial Pivoting
solution_gepp = gauss_elimination_partial_pivoting(A, b)

# Display the results
print("Solution using Gauss Elimination with Partial Pivoting:", solution_gepp)
