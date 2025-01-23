import numpy as np

def gauss_elimination(A, b):
    n = len(b)
    # Augment the matrix A with b
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        # Partial pivoting: find the row with the largest pivot and swap
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        # Eliminate entries below the pivot
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i] / augmented_matrix[i, i]
            augmented_matrix[j, i:] -= factor * augmented_matrix[i, i:]

    # Back substitution to solve for x
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (augmented_matrix[i, -1] - np.dot(augmented_matrix[i, i + 1:n], x[i + 1:])) / augmented_matrix[i, i]

    return x

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

def doolittle_lu(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        # Upper triangular matrix U
        for j in range(i, n):
            U[i, j] = A[i, j] - sum(L[i, k] * U[k, j] for k in range(i))

        # Lower triangular matrix L
        for j in range(i, n):
            if i == j:
                L[i, i] = 1  # Diagonal as 1
            else:
                L[j, i] = (A[j, i] - sum(L[j, k] * U[k, i] for k in range(i))) / U[i, i]

    return L, U

def lu_solve(L, U, b):
    # Forward substitution to solve Ly = b
    n = L.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    # Back substitution to solve Ux = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]

    return x

def gauss_seidel(A, b, tol=1e-10, max_iterations=100):
    n = len(b)
    x = np.zeros(n)  # Initial guess

    for k in range(max_iterations):
        x_new = np.copy(x)

        for i in range(n):
            sum1 = sum(A[i, j] * x_new[j] for j in range(i))
            sum2 = sum(A[i, j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new

        x = x_new

    raise ValueError("Gauss-Seidel method did not converge")

# Coefficient matrix A and vector b for Gauss-Seidel method
A = np.array([
    [1, 2, 3],
    [6, -4, 5],
    [5, 2, 2]
], dtype=float)
b = np.array([4, 10, 25], dtype=float)

# Solve using Gauss-Seidel method
solution_seidel = gauss_seidel(A, b)
print("Solution using Gauss-Seidel method:", solution_seidel)

# Explanation of matrix factorization:
# Matrix factorization involves breaking down a matrix into simpler matrices (e.g., L and U)
# to simplify solving linear systems, computing determinants, or performing matrix inversion.
