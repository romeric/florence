# This program generates Jacobi polynomials of any order used for generating interpolation functions
# Based on the book "George Em Karniadakis & Spencer J. Sherwin (1999) - Spectral hp-element methods for CFD" Appendix A
# from WrittenJacobiPolynomials import WrittenJacobiPolynomials
# from WrittenJacobiPolynomials_8 import WrittenJacobiPolynomials_8
# import JacobiPolynomials_Cy

# from numba import jit
# from numba.decorators import jit, autojit
# import numpy as np 

# @jit
def JacobiPolynomials(n,xi,a=0,b=0):
    # Input arguments:
    # n - polynomial degree
    # xi - evalution point
    # a,b - alpha and beta parameters for Jacobi Polynmials
            # a=b=0 for Legendre polynomials
            # a=b=-0.5 for Chebychev polynomials

    # Written Jacobi is not a good idea at least for Python (Numpy/Scipy)
    # P = []
    # # if n < 17:
    # if n < 50:
    #   # P = WrittenJacobiPolynomials(n,xi,a,b)
    #   P = JacobiPolynomials_Cy.JacobiPolynomials(n,xi,a,b)
    # else:

    # The first two polynomials
    # P = np.zeros((n+1,1))
    P=[0]*(n+1)  # List seems much faster than np.array here

    P[0] = 1.0
    if n>0:
        P[1] = 0.5*((a-b)+(a+b+2)*xi)

    if n>1:
        for p in range(1,n):
            # Evaluate coefficients
            a1n = 2*(p+1)*(p+a+b+1)*(2*p+a+b)
            a2n = (2*p+a+b+1)*(a**2-b**2)
            a3n = (2*p+a+b)*(2*p+a+b+1)*(2*p+a+b+2)
            a4n = 2*(p+a)*(p+b)*(2*p+a+b+2)
            # print p
            P[p+1] = ((a2n+a3n*xi)*P[p]-a4n*P[p-1])/a1n

    return P

# @jit
def DiffJacobiPolynomials(n,xi,a=0,b=0,opt=0):
    # opt is for Gauss-Lobatto integration purpose only
    # Compute derivatives
    # dP = np.zeros((n+1,1))
    dP=[0]*(n+1)    # List seems much faster than np.array here 

    if opt==1:
        P = JacobiPolynomials(n,xi,a+1,b+1)
    else:
        P = JacobiPolynomials(n,xi,a,b)

    for p in range(1,n+1):
        dP[p] = 0.5*(a+b+p+1)*P[p-1]
        
    return dP

