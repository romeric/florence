from libc.stdlib cimport malloc, free 
from cython cimport double, sizeof, boundscheck, wraparound

cdef extern from "jacobi.c":
	void jacobi(const unsigned short n, const double xi,
			const double a, const double b, double *P)
	void diffjacobi(const unsigned short n, const double xi,
			const double a, const double b, const unsigned short opt, double *dP)



@boundscheck(False)
@wraparound(False)
def JacobiPolynomials(const unsigned short n, double xi, double a=0., double b=0.):
	cdef:
		int i
		double *P = <double*>malloc( (n+1)*sizeof(double))
	jacobi(n,xi,a,b,P);
	P_py=[0]*(n+1)
	for i in range(n+1):
		P_py[i] = P[i]

	free(P)
	return P_py

@boundscheck(False)
@wraparound(False)
def DiffJacobiPolynomials(const int n,double xi,double a=0.,double b=0.,int opt=0):
	cdef:
		int i
		double *dP = <double*>malloc( (n+1)*sizeof(double))
	
	diffjacobi(n,xi,a,b,opt,dP);
	dP_py=[0]*(n+1)
	for i in range(n+1):
		dP_py[i] = dP[i]
	
	free(dP);
	return dP_py
	
