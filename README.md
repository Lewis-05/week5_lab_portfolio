# week5_lab_portfolio
Portfolio exercise You should submit:
Your code for lu_factorisation
The determinant of A_large from Exercise 1.3.
The plot of run times for LU factorisation (your code) and Gaussian elimination (from the notes)

Implement the algorithm for LU factorisation as described in the notes. You should implement a Python function which accepts an nxn matrix A represented as a numpy array and returns one lower triangular matrix L and one upper triangular matrix U with A=LU.


# tests 
'''
A = np.array([[2,3,4],
              [5,6,7],
              [1,2,3]])

print(A)
print("\n L and U ")
L, U = lu_factorisation(A)
print(L)
print(U)
'''