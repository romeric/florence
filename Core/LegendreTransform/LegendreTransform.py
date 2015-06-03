import numpy as np
from Core.Supplementary.Tensors.Tensors import *

def FreeEnergy2Enthalpy(W_Permittivity,W_CoupledTensor,W_Elasticity,opt=0):
	# Converts the directional derivatives of the free energy of a system to its electric enthalpy.
	# opt=0, operates in Voigt form inputs
	# opt=1, operates in indicial form inputs
	# NOTE THAT THE COUPLED TENSOR SHOULD BE SYMMETRIC WITH RESPECT TO THE LAST TWO INDICES

	# Irrespective of the input option (opt), the output is always in Voigt form

	# Permittivity is the same in Voigt and index formats
	Inverse = np.linalg.inv(W_Permittivity)
	H_Permittivity = -Inverse

	if opt==0:
	
		# H_CoupledTensor = np.dot(Inverse,W_CoupledTensor)
		# H_Elasticity = W_Elasticity - np.dot(W_CoupledTensor.T,H_CoupledTensor)

		H_CoupledTensor = np.dot(Inverse,W_CoupledTensor.T)
		H_Elasticity = W_Elasticity - np.dot(W_CoupledTensor,H_CoupledTensor)
		
		H_CoupledTensor = H_CoupledTensor.T

	elif opt==1:
		# Using Einstein summation (using numpy einsum call)
		d = np.einsum
		# Computing directional derivatives of the enthalpy
		W_CoupledTensor_Ts = d('kij',W_CoupledTensor) 
		H_CoupledTensor = d('ij,jkl',Inverse,W_CoupledTensor_Ts)
		H_CoupledTensor_Ts = d('kij',H_CoupledTensor)
		H_CoupledTensor = d('ij,jkl',Inverse,W_CoupledTensor) #
		# H_Elasticity = W_Elasticity - Voigt( d('ijk,klm',W_CoupledTensor,H_CoupledTensor) )
		# H_Elasticity = W_Elasticity - Voigt( d('ijk,klm',W_CoupledTensor_Ts,H_CoupledTensor) )

		# H_Elasticity = W_Elasticity - Voigt( d('kij,klm',W_CoupledTensor,H_CoupledTensor_Ts) )

		H_Elasticity = W_Elasticity - Voigt( d('kij,klm',W_CoupledTensor_Ts,H_CoupledTensor) )  #



		H_CoupledTensor = Voigt(H_CoupledTensor,1)



	return H_Permittivity, H_CoupledTensor, H_Elasticity