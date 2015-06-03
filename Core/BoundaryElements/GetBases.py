import numpy as np 
from Core.InterpolationFunctions.OneDimensional import LagrangeGaussLobatto

def GetBases(C,z):
	# Get basis at all integration points - every column corresponds to a Gauss point 
	Basis = np.zeros((C+2,z.shape[0])); dBasis = np.copy(Basis)
	for i in range(0,z.shape[0]):
		# Basis[0:,i], dBasis[0:,i], _ = Lagrange(C,z[i])
		Basis[0:,i], dBasis[0:,i], _ = LagrangeGaussLobatto(C,z[i])

	return Basis, dBasis