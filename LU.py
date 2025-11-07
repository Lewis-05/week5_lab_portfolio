import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import time

def system_size(A, b):
    """
    Validate the dimensions of a linear system and return its size.

    This function checks whether the given coefficient matrix `A` is square
    and whether its dimensions are compatible with the right-hand side vector
    `b`. If the dimensions are valid, it returns the size of the system.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D array of shape ``(n, n)`` representing the coefficient matrix of
        the linear system.
    b : numpy.ndarray
        A array of shape ``(n, o)`` representing the right-hand side vector.

    Returns
    -------
    int
        The size of the system, i.e., the number of variables `n`.

    Raises
    ------
    ValueError
        If `A` is not square or if the size of `b` does not match the number of
        rows in `A`.
    """

    # Validate that A is a 2D square matrix
    if A.ndim != 2:
        raise ValueError(f"Matrix A must be 2D, but got {A.ndim}D array")

    n, m = A.shape
    if n != m:
        raise ValueError(f"Matrix A must be square, but got A.shape={A.shape}")

    if b.shape[0] != n:
        raise ValueError(
            f"System shapes are not compatible: A.shape={A.shape}, "
            f"b.shape={b.shape}"
        )

    return n


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


def determinant(A):
    n = A.shape[0]
    L, U = lu_factorisation(A)

    det_L = 1.0
    det_U = 1.0

    for i in range(n):
        det_L *= L[i, i]
        det_U *= U[i, i]

    return det_L * det_U


def row_add(A, b, p, k, q):
    """
    Perform an in-place row addition operation on a linear system.

    This function applies the elementary row operation:

    ``row_p ← row_p + k * row_q``

    where `row_p` and `row_q` are rows in the coefficient matrix `A` and the
    right-hand side vector `b`. It updates the entries of `A` and `b`
    **in place**, directly modifying the original data.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the linear system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector of the system.
    p : int
        The index of the row to be updated (destination row). Must satisfy
        ``0 <= p < n``.
    k : float
        The scalar multiplier applied to `row_q` before adding it to `row_p`.
    q : int
        The index of the source row to be scaled and added. Must satisfy
        ``0 <= q < n``.
    """
    n = system_size(A, b)

    # Perform the row operation
    for j in range(n):
        A[p, j] = A[p, j] + k * A[q, j]

    # Update the corresponding value in b
    b[p, 0] = b[p, 0] + k * b[q, 0]


def gaussian_elimination(A, b, verbose=False):
    """
    Perform Gaussian elimination to reduce a linear system to upper triangular
    form.

    This function performs **forward elimination** to transform the coefficient
    matrix `A` into an upper triangular matrix, while applying the same
    operations to the right-hand side vector `b`. This is the first step in
    solving a linear system of equations of the form ``Ax = b`` using Gaussian
    elimination.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the coefficient matrix
        of the system.
    b : numpy.ndarray
        A 2D NumPy array of shape ``(n, 1)`` representing the right-hand side
        vector.
    verbose : bool, optional
        If ``True``, prints detailed information about each elimination step,
        including the row operations performed and the intermediate forms of
        `A` and `b`. Default is ``False``.

    Returns
    -------
    None
        This function modifies `A` and `b` **in place** and does not return
        anything.
    """
    # find shape of system
    n = system_size(A, b)

    # perform forwards elimination
    for i in range(n - 1):
        # eliminate column i
        if verbose:
            print(f"eliminating column {i}")
        for j in range(i + 1, n):
            # row j
            factor = A[j, i] / A[i, i]
            if verbose:
                print(f"  row {j} |-> row {j} - {factor} * row {i}")
            row_add(A, b, j, -factor, i)

        if verbose:
            print()
            print("new system")
            print(A)
            print(b)
            print()


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


def forward_substitution(A, b):
    """
    Solve a lower triangular system of linear equations using forward
    substitution.

    This function solves the system of equations:

    .. math::
        A x = b

    where `A` is a **lower triangular matrix** (all elements above the main
    diagonal are zero). The solution vector `x` is computed sequentially by
    solving each equation starting from the first row.

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` representing the lower triangular
        coefficient matrix of the system.
    b : numpy.ndarray
        A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
        ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape ``(n,)`` containing the solution vector.
    """
    """
    solves the system of linear equationa Ax = b assuming that A is lower
    triangular. returns the solution x
    """
    # get size of system
    n = system_size(A, b)

    # check is lower triangular
    if not np.allclose(A, np.tril(A)):
        raise ValueError("Matrix A is not lower triangular")

    # create solution variable
    x = np.empty_like(b)

    # perform forwards solve
    for i in range(n):
        partial_sum = 0.0
        for j in range(0, i):
            partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x


def backward_substitution(A, b):
    '''
    Solve an upper triangular system of linear equations using backward
    substitution.

    This function solves the system of equations:

    .. math::
    A x = b

    where `A` is an **upper triangular matrix** (all elements below the main
    diagonal are zero). The solution vector `x` is computed starting from the
    last equation and proceeding backward.

    Parameters
    ----------
    A : numpy.ndarray
    A 2D NumPy array of shape ``(n, n)`` representing the upper triangular
    coefficient matrix of the system.
    b : numpy.ndarray
    A 1D NumPy array of shape ``(n,)`` or a 2D NumPy array of shape
    ``(n, 1)`` representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
    A NumPy array of shape ``(n,)`` containing the solution vector.
    '''
    # get size of system
    n = system_size(A, b)

    # check is upper triangular
    assert np.allclose(A, np.triu(A))

    # create solution variable
    x = np.empty_like(b)

    # perform backwards solve
    for i in range(n - 1, -1, -1): # iterate over rows backwards
        partial_sum = 0.0
    for j in range(i + 1, n):
        partial_sum += A[i, j] * x[j]
        x[i] = 1.0 / A[i, i] * (b[i] - partial_sum)

    return x


def lu_solver(A, b):
    """
    Solve a linear system A x = b using LU factorisation and substitution.

    The system is solved in three main steps:
    1. Factorisation: A = L U (using lu_factorisation)
    2. Forward Solve: L y = b for the intermediate vector y (using forward_substitution)
    3. Backward Solve: U x = y for the solution vector x (using backward_substitution)

    Parameters
    ----------
    A : numpy.ndarray
        A 2D NumPy array of shape (n, n) representing the coefficient matrix.
    b : numpy.ndarray
        A 2D NumPy array of shape (n, 1) or 1D array of shape (n,) 
        representing the right-hand side vector.

    Returns
    -------
    x : numpy.ndarray
        A NumPy array of shape (n, 1) or (n,) containing the solution vector.
    """
    # 1. Factorisation: A = L U
    L, U = lu_factorisation(A.copy()) # Use a copy to avoid modifying the original A

    # 2. Forward Solve: L y = b
    # b needs to be reshaped to (n,) for the substitution functions if it is (n, 1)
    b_flat = b.flatten()
    y = forward_substitution(L, b_flat)

    # 3. Backward Solve: U x = y
    x = backward_substitution(U, y)

    # Return x in the same shape as b was (if b was 2D, return 2D)
    if b.ndim == 2 and b.shape[1] == 1:
        return x.reshape(-1, 1)
    return x


A_large, b_large, x_large = generate_safe_system(100)
print(A_large)
print(determinant(A_large))

'''


for n in sizes:
    # generate a random system of linear equations of size n
    A, b, x = generate_safe_system(n)

    # do the solve
'''
sizes = [2**j for j in range(1, 6)]
n_trials = 5 # Number of trials for averaging time
lu_factor_times = []
lu_forward_times = []
lu_backward_times = []
ge_elimination_times = []

print(f"{'Size (N)':<10} | {'GE Time (s)':<15} | {'LU Factor (s)':<15} | {'LU Fwd Solve (s)':<15} | {'LU Bwd Solve (s)':<15}")
print("-" * 75)

for n in sizes:
    A_orig, b_orig, x_true = generate_safe_system(n)
    
    # --- Gaussian Elimination Time ---
    total_ge_time = 0.0
    for _ in range(n_trials):
        A_ge, b_ge = A_orig.copy(), b_orig.copy()
        start = time.perf_counter()
        gaussian_elimination(A_ge, b_ge)
        # The backward solve is not included here, as it's separate in the lecture notes
        # but the forward elimination step (which reduces the system to upper triangular form)
        # is the O(N^3) part, equivalent to LU factorisation.
        end = time.perf_counter()
        total_ge_time += (end - start)
        
    avg_ge_elim_time = total_ge_time / n_trials

    # --- LU Factorisation Time ---
    total_lu_factor_time = 0.0
    for _ in range(n_trials):
        start = time.perf_counter()
        L, U = lu_factorisation(A_orig.copy())
        end = time.perf_counter()
        total_lu_factor_time += (end - start)
        
    avg_lu_factor_time = total_lu_factor_time / n_trials
    
    # --- LU Solve Time (Forward and Backward) ---
    b_flat = b_orig.flatten()
    
    # Forward Solve (L y = b)
    total_lu_forward_time = 0.0
    for _ in range(n_trials):
        start = time.perf_counter()
        y = forward_substitution(L, b_flat)
        end = time.perf_counter()
        total_lu_forward_time += (end - start)
        
    avg_lu_forward_time = total_lu_forward_time / n_trials
    
    # Backward Solve (U x = y)
    total_lu_backward_time = 0.0
    for _ in range(n_trials):
        start = time.perf_counter()
        x = backward_substitution(U, y)
        end = time.perf_counter()
        total_lu_backward_time += (end - start)
        
    avg_lu_backward_time = total_lu_backward_time / n_trials

    lu_factor_times.append(avg_lu_factor_time)
    lu_forward_times.append(avg_lu_forward_time)
    lu_backward_times.append(avg_lu_backward_time)
    ge_elimination_times.append(avg_ge_elim_time)

    print(f"{n:<10} | {avg_ge_elim_time:<15.6f} | {avg_lu_factor_time:<15.6f} | {avg_lu_forward_time:<15.6f} | {avg_lu_backward_time:<15.6f}")


# ----------------- Plotting -----------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

# --- Plot 1: O(N^3) Operations ---
ax1.plot(sizes, ge_elimination_times, 'o-', color='red', label='Gaussian Elimination (Elimination only)')
ax1.plot(sizes, lu_factor_times, 's-', color='blue', label='LU Factorisation ($A=LU$)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Comparison of $O(N^3)$ Operations (Log-Log Scale)')
ax1.set_xlabel('System Size (N)')
ax1.set_ylabel('Average Execution Time (s)')
ax1.grid(True, which="both", ls="--", linewidth=0.5)
ax1.legend()

plt.tight_layout()
plt.savefig('performance_comparison.png')
