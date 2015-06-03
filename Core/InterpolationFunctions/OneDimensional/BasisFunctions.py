import numpy as np
import scipy as sp
import scipy.linalg as la
from Core.NumericalIntegration import GaussLobattoQuadrature


def Lagrange(C,xi):
	n = C+2
	nsize = n-1
	ndiv = 2.0/nsize
	eps = 1.0*np.zeros(n)
	eps[0]=-1.; eps[n-1]=1.

	for i in range(0,nsize):
		eps[i+1] = eps[i]+ndiv

	A = 1.0*np.zeros((n,n))
	A[:,0] = np.ones(n)

	for i in range(1,n):
		for j in range(0,n):
			A[j,i] = pow(eps[j],i)


	N = 1.0*np.zeros(n); dN=1.0*np.zeros(n)

	for ishape in range(0,n):
		RHS = 1.0*np.zeros(n)
		RHS[ishape] = 1.

		# Solve linear system (dense LU)
		coeff = sp.linalg.solve(A,RHS)
		# Build shape functions 
		for incr in range(0,n):
			N[ishape] = N[ishape]+coeff[incr]*pow(xi,incr)

		# Build derivate of shape functions
		for incr in range(0,n-1):
			dN[ishape] = dN[ishape]+(incr+1)*coeff[incr+1]*pow(xi,incr)


	return (N,dN,eps) 


def Legendre(C,xi):
	# For Linear Basis Generating Legendre Polynomials is Not Required
	if C==0:
		N = np.array([1.0/2*(1-xi), 1.0/2*(1+xi)])
		dN = np.array([-1.0/2, 1./2])

	# For Higher Order 
	elif C>0:
		# The First Two Legendre Polynomials 
		p0 = 1.0; p1 = xi
		# Derivatives of The First Two Legendre Polynomials 
		dp0 = 0.0; dp1 = 1.0
		# Allocate Size and Dimensions
		ndim = C+2
		P = np.zeros((ndim+1,1)); dP = np.zeros((ndim+1,1))
		N = np.zeros((ndim+1,1)); dN = np.zeros((ndim+1,1))
		P[0] = p0; P[1] = p1
		dP[0] = dp0; dP[1] = dp1
		# Generate Legendre Polynomials
		for i in range(2,ndim+1):
			P[i]  = ((2.0*i-1)*xi*P[i-1] - (i-1)*P[i-2])/(i)
			dP[i]  = ((2.0*i-1)*xi*dP[i-1] + (2.0*i-1)*P[i-1] - (i-1)*dP[i-2])/(i)

		# From Legendre Polynomials Generate FE Basis Functions 
		for i in range(3,ndim+2):
			# N[i-1] =  (P[i-1]-P[i-3])/np.sqrt(2*(2*i-3))
			# dN[i-1] =  (dP[i-1]-dP[i-3])/np.sqrt(2*(2*i-3))
			# Ledger's Normalisation 
			N[i-1] =  (P[i-1]-P[i-3])/((2.0*i-3.))
			dN[i-1] =  (dP[i-1]-dP[i-3])/((2.0*i-3.))


		# Put the hat functions at exterior nodes  
		N = np.append([np.append([1.0/2.0*(1.0-xi)],N[2:-1])],[1.0/2*(1.0+xi)])
		dN = np.append([np.append([-0.5],dN[2:-1])],[0.5])


	return (N,dN)




def LagrangeGaussLobatto(C,xi):
	n = C+2
	nsize = n-1
	ndiv = 2.0/nsize

	eps = GaussLobattoQuadrature(n)[0]

	A = 1.0*np.zeros((n,n))
	A[:,0] = np.ones(n)

	for i in range(1,n):
		for j in range(0,n):
			A[j,i] = pow(eps[j],i)


	N = 1.0*np.zeros(n); dN=1.0*np.zeros(n)

	for ishape in range(0,n):
		RHS = 1.0*np.zeros(n)
		RHS[ishape] = 1.

		# Solve linear system (dense LU)
		coeff = sp.linalg.solve(A,RHS)
		# Build shape functions 
		for incr in range(0,n):
			N[ishape] = N[ishape]+coeff[incr]*pow(xi,incr)

		# Build derivate of shape functions
		for incr in range(0,n-1):
			dN[ishape] = dN[ishape]+(incr+1)*coeff[incr+1]*pow(xi,incr)


	return (N,dN,eps) 



# import matplotlib.pyplot as plt
# C=5 
# n = np.linspace(-1,1,100)
# N = np.zeros((n.shape[0],C+2))
# for m in range(C+2):
# 	for i in range(0,n.shape[0]):
# 		N[i,:] = BasisFunctions().LagrangeGaussLobattoShapeFunctions(C,n[i])[0]
# 		# N[i,:] = BasisFunctions().LagrangeShapeFunctions(C,n[i])[0]
# 	plt.plot(n,N[:,m])
# plt.show()
