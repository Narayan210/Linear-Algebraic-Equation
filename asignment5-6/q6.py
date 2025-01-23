import numpy as np
from scipy.linalg import lu

def lu_solve(A, b):
    # Perform LU decomposition
    P, L, U = lu(A)

    # Forward substitution to solve Ly = Pb
    Pb = np.dot(P, b)
    y = np.zeros_like(b, dtype=float)
    for i in range(len(b)):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Back substitution to solve Ux = y
    x = np.zeros_like(b, dtype=float)
    for i in range(len(b) - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Coefficient matrix A and vector b
A = np.array([
    [1, 2],
    [3, 22]
], dtype=float)
b = np.array([4, 12], dtype=float)

# Solve the system
solution = lu_solve(A, b)
print("Solution:", solution)
