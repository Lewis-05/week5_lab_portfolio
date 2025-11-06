import numpy as np
from scipy.sparse import diags


def generate_safe_system(n):
    """
    Generate a linear system A x = b where A is strictly diagonally dominant,
    ensuring LU factorization without pivoting will work.

    Parameters:
        n (int): Size of the system (n x n)

    Returns:
        A (ndarray): n x n strictly diagonally dominant matrix
        b (ndarray): RHS vector
        x_true (ndarray): The true solution vector
    """

    k = [np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)]
    offset = [-1, 0, 1]
    A = diags(k, offset).toarray()

    # Solution is always all ones
    x_true = np.ones((n, 1))

    # Compute b = A @ x_true
    b = A @ x_true

    return A, b, x_true


def lu_factorisation(A):
    """
    Compute the LU factorisation of a square matrix A.

    The function decomposes a square matrix ``A`` into the product of a lower
    triangular matrix ``L`` and an upper triangular matrix ``U`` such that:

    .. math::
        A = L U

    where ``L`` has unit diagonal elements and ``U`` is upper triangular.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the square matrix to
        factorise.

    Returns
    -------
    L : numpy.ndarray
        A lower triangular matrix with shape ``(n, n)`` and unit diagonal.
    U : numpy.ndarray
        An upper triangular matrix with shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A is not square {A.shape=}")

    # construct arrays of zeros
    #L, U = np.zeros_like(A, dtype=float), np.zeros_like(A, dtype=float)
    L = np.zeros_like(A, dtype=float)
    U = A.astype(float).copy() # researched a better way to intialise the matrices with copy()

    # ...
    #L[0, 0] = 1
    #U[0, 0] = A[0, 0] # lecture notes prove this
    
    # make sure in the lower triangle matrix that the main diagonal is 1
    for j in range(0, n):
        for i in range(0, n+1):
            if i == j:
                L[i,i] = 1

    '''
    # all first row of A is equal to first row of U - shown by lecture notes
    for i in range(1, n):
        U[0, i] = A[0, i]
    '''
    
    # for loops for the process
    for i in range(n):
        # The element for the diagonal point above what is being eliminated
        temp = U[i, i]

        # Iterate over the rows j below the temp row i
        for j in range(i + 1, n):
            # Calculate the multiplier (l_ji), which is the elimination factor that would zero 
            # out the element underneath the current 
            factor = U[j, i] / temp 
            # Store the multiplier in the L matrix as this holds all factors for the corresponding 
            # elements to make A from L @ U
            L[j, i] = factor
            
            # iterate through the elements (k) in the current row
            for k in range(i, n):
                # row j is modified by subtracting a multiple of the row above the gaussian elimination
                # do this by sub(factor * row i) 
                U[j, k] = U[j, k] - factor * U[i, k] # U[i, k] is the element above U[j, k]

    
    return L, U

# tests 
A = np.array([[2,3,4],
              [5,6,7],
              [1,2,3]])

print(A)
print("\n L and U ")
L, U = lu_factorisation(A)
print(L)
print(U)
