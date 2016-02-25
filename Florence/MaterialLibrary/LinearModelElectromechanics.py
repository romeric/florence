import numpy as np
from Florence.Tensor import trace

#####################################################################################################
								# Isotropic Linear Model
#####################################################################################################


class LinearModelElectromechanics(object):
	"""docstring for LinearModelElectromechanics"""
	def __init__(self, ndim):
		super(LinearModelElectromechanics, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim+1
		self.modelname = 'LinearModelElectromechanics'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum
		I = StrainTensors['I']

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# Fourth order elasticity tensor
		C_Voigt = Voigt(			
			lamb*d('ij,kl',I,I)+mu*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1
			)
		
		e_voigt = Voigt( np.zeros((ndim,ndim,ndim)),1)
			
		# Dielectric Tensor (Permittivity - 2nd order)
		# Permittivity = np.zeros((ndim,ndim))
		Permittivity = MaterialArgs.eps_1* np.eye(ndim,ndim)

		# Build the Hessian
		factor = -1.
		H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
		H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
		H_Voigt = np.concatenate((H1,H2),axis=0)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):


		strain = StrainTensors['strain'][gcounter]
		I = StrainTensors['I']

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 2*mu*strain + lamb*np.trace(strain)*I 
		# USE FASTER TRACE FUNCTION
		return 2*mu*strain + lamb*trace(strain)*I  
		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
