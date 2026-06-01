import numpy as np

def matrix_sum(A, B):
    """(a) Matrix addition A + B"""
    return np.add(A, B)

def matrix_determinant(M):
    """(b) Determinant of a square matrix"""
    return np.linalg.det(M)

def matrix_transpose(A):
    """(c) Transpose of A"""
    return A.T.copy()

def is_symmetric(A, tol=1e-10):
    """(d) True if A == A^T"""
    return np.allclose(A, A.T, atol=tol)

def matrix_multiply(A, B):
    """(e) Matrix multiplication A @ B"""
    if A.shape[1] != B.shape[0]:
        raise ValueError(f"Incompatible shapes: {A.shape} vs {B.shape}")
    return A @ B

def matrix_power(A, n):
    """
    (f) Raise a square matrix A to an integer power n.
    n=0 returns identity, n=-1 returns inverse, n>0 repeats multiplication.
    Example: A^3 = A @ A @ A
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    return np.linalg.matrix_power(A, n)


def matrix_inverse(A):
    """
    (g) Compute the inverse A^-1 of a square, non-singular matrix.
    Raises an error if the matrix is singular (det = 0).
    Property: A @ A^-1 = A^-1 @ A = I
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Matrix must be square, got shape {A.shape}")
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("Matrix is singular (det ≈ 0) — inverse does not exist")
    return np.linalg.inv(A)


def matrix_rank(A):
    """
    (h) Return the rank of matrix A.
    Rank = number of linearly independent rows (or columns).
    For an (m x n) matrix: rank <= min(m, n).
    """
    return np.linalg.matrix_rank(A)


def matrix_trace(A):
    """
    (i) Return the trace of a square matrix A.
    Trace = sum of all diagonal elements = sum of eigenvalues.
    Example: trace([[1,2],[3,4]]) = 1 + 4 = 5
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Trace requires a square matrix, got {A.shape}")
    return np.trace(A)


def is_identity(A, tol=1e-10):
    """
    (j) Return True if A is an identity matrix.
    Checks: diagonal entries = 1, off-diagonal entries = 0.
    Example: np.eye(3) -> True
    """
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A, np.eye(A.shape[0]), atol=tol)


def is_orthogonal(A, tol=1e-10):
    """
    (k) Return True if A is an orthogonal matrix.
    Condition: A @ A^T = I  (columns are orthonormal unit vectors).
    Property: A^-1 = A^T, det(A) = ±1.
    Example: rotation matrices are orthogonal.
    """
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A @ A.T, np.eye(A.shape[0]), atol=tol)


def is_diagonal(A, tol=1e-10):
    """
    (l) Return True if A is a diagonal matrix.
    All off-diagonal entries must be zero (or within tol).
    Example: [[3,0],[0,7]] -> True
    """
    if A.shape[0] != A.shape[1]:
        return False
    return np.allclose(A - np.diag(np.diag(A)), 0, atol=tol)


def matrix_norm(A, ord=None):
    """
    (m) Compute the norm (size/magnitude) of matrix A.
    ord=None  -> Frobenius norm: sqrt(sum of all squared entries)
    ord=2     -> Spectral norm: largest singular value
    ord=1     -> Max absolute column sum
    ord=np.inf-> Max absolute row sum
    """
    return np.linalg.norm(A, ord=ord)


def solve_linear_system(A, b):
    """
    (n) Solve the linear system A @ x = b for unknown vector x.
    Uses LU decomposition (more stable than x = A^-1 @ b).
    Raises an error if A is singular (no unique solution).
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"Coefficient matrix must be square, got {A.shape}")
    if abs(np.linalg.det(A)) < 1e-12:
        raise ValueError("Matrix is singular — no unique solution exists")
    return np.linalg.solve(A, b)


def matrix_kronecker(A, B):
    """
    (o) Compute the Kronecker product A ⊗ B.
    Each element of A is multiplied by the entire matrix B,
    producing a block matrix of shape (m*p, n*q) if A is (m,n) and B is (p,q).
    Used in quantum mechanics and signal processing.
    """
    return np.kron(A, B)

