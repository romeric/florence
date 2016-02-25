import numpy as np
import numpy.linalg as la 
import scipy.linalg as sla 

def LG_NewtonRaphson(PermittivityW, ElectricFieldx):	
	# Given electric field and permittivity, computes electric displacement

	ndim = ElectricFieldx.shape[0]
	if np.allclose(la.norm(ElectricFieldx),0):
		# BE WARNED THAT THIS MAY NOT ALWAYS BE THE CASE
		D = np.zeros((ndim,1))
	else:
		# Newton-Raphson scheme to find electric displacement from the free energy
		tolerance = 1e-13
		D = np.zeros((ndim,1))
		# D = np.copy(ElectricFieldx)
		Residual = -ElectricFieldx
		ResidualNorm = []

		while np.abs(la.norm(Residual)/la.norm(ElectricFieldx)) > tolerance:

			# Update the hessian - depending on the model, extra arguments needs to be passed
			# PermittivityW = (1.0/varepsilon_1)*d2

			deltaD = sla.solve(PermittivityW,-Residual)
			# Update electric displacement
			D += deltaD
			# Find residual (first term is equivalent to internal traction)
			Residual = np.dot(PermittivityW,D) - ElectricFieldx
			# Save internal tractions
			ResidualNorm = np.append(ResidualNorm,np.abs(la.norm(Residual)/la.norm(ElectricFieldx)))

		# print ResidualNorm

	return D

