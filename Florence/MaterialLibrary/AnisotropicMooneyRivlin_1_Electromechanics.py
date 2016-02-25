import numpy as np
from Florence.Tensor import trace
#####################################################################################################
								# Isotropic AnisotropicMooneyRivlin_1_Electromechanics Model
#####################################################################################################


class AnisotropicMooneyRivlin_1_Electromechanics(object):
	"""docstring for AnisotropicMooneyRivlin_1_Electromechanics"""
	def __init__(self, ndim):
		super(AnisotropicMooneyRivlin_1_Electromechanics, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim+1
		self.modelname = 'AnisotropicMooneyRivlin_1_Electromechanics'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		# H_ = StrainTensors.H
		# G = np.dot(H_.T,H_)
		# g = np.dot(H_,H_.T)

		# Update Lame constants
		mu2 = mu - lamb*(J-1.0)
		lamb2 = lamb*(2.0*J-1.0) - mu

		C_Voigt = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)

		
		# Coupled Tensor (e - 3rd order)
		e_voigt = Voigt( np.zeros((ndim,ndim,ndim)),1)
			
		# Dielectric Tensor (Permittivity - 2nd order)
		Permittivity = np.zeros((ndim,ndim))
		# Permittivity = MaterialArgs.eps_1* np.eye(ndim,ndim)

		# Build the Hessian
		factor = -1.
		H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
		H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
		H_Voigt = np.concatenate((H1,H2),axis=0)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I 


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
