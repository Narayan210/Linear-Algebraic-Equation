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

# Coefficient matrix A and vector b for the Gauss-Jordan method
A = np.array([
    [3, 5, -3, 1],
    [2, 1, 1, 4],
    [3, -4, 0, -1],
    [2, 1, -3, 9]
], dtype=float)
b = np.array([16, 9, 1, 5], dtype=float)

# Solve using Gauss-Jordan method
solution_jordan = gauss_jordan(A, b)
print("Solution using Gauss-Jordan method:", solution_jordan)

# Solve using Gauss Elimination method
solution_elimination = gauss_elimination(A, b)
print("Solution using Gauss Elimination method:", solution_elimination)
