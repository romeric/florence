def JacobiPolynomials(int n,double xi,double a=0.,double b=0.):
	
	cdef double a1n, a2n, a3n, a4n
	cdef int p
	P = [0]*(n+1)
	
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
			P[p+1] = ((a2n+a3n*xi)*P[p]-a4n*P[p-1])/a1n

	return P


def DiffJacobiPolynomials(int n,double xi,double a=0.,double b=0,int opt=0):
	# opt is for Gauss-Lobatto integration purpose only
	# Compute derivatives
	# dP = np.zeros((n+1,1))
	dP=[0]*(n+1) 	# List seems much faster than np.array here 
	cdef int p

	if opt==1:
		P = JacobiPolynomials(n,xi,a+1,b+1)
	else:
		P = JacobiPolynomials(n,xi,a,b)

	for p in range(1,n+1):
		dP[p] = 0.5*(a+b+p+1)*P[p-1]
		
	return dP

