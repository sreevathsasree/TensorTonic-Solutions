import numpy as np

def matrix_transpose(A):
    """
    Manually transpose a 2D NumPy array.
    Returns a new array of shape (M, N).
    """
    A = np.array(A)

    N, M = A.shape   # Original shape (rows, cols)

    # Create empty matrix of swapped shape
    AT = np.zeros((M, N), dtype=A.dtype)

    # Swap indices
    for i in range(N):
        for j in range(M):
            AT[j, i] = A[i, j]

    return AT