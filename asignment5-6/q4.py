import numpy as np

def gauss_seidel(A, b, x0, tol, max_iterations):
    n = len(b)
    x = x0.copy()

    for iteration in range(max_iterations):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Check for convergence
        if np.linalg.norm(np.array(x_new) - np.array(x), ord=np.inf) < tol:
            return x_new, iteration + 1

        x = x_new

    raise ValueError("Gauss-Seidel method did not converge within the maximum number of iterations.")

# Coefficient matrix and right-hand side vector
A = np.array([
    [1, 2, 3],
    [6, 4, 5],
    [5, 2, 3]
], dtype=float)
b = np.array([4, 16, 12], dtype=float)

# Initial guess, tolerance, and maximum iterations
x0 = np.zeros(len(b))
tol = 1e-6
max_iterations = 100

# Solve the system
try:
    solution, iterations = gauss_seidel(A, b, x0, tol, max_iterations)
    print("Solution:", solution)
    print("Iterations:", iterations)
except ValueError as e:
    print(e)
