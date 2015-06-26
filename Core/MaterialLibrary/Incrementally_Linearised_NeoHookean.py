import numpy as np
# from Core.Supplementary.Tensors.Tensors import *
from Core.Supplementary.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic Linear Model
#####################################################################################################


class Incrementally_Linearised_NeoHookean(object):
	"""docstring for Incrementally_Linearised_NeoHookean"""
	def __init__(self, ndim):
		super(Incrementally_Linearised_NeoHookean, self).__init__()
		self.ndim = ndim
	def Get(self):
		# self.nvar = self.ndim+1
		self.nvar = self.ndim
		self.modelname = 'Incrementally_Linearised_NeoHookean'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		H_Voigt_k = MaterialArgs.H_Voigt[:,:,elem,gcounter]

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		# UPDATE MATERIAL CONSTANTS
		# mu2 = mu - lamb*(J-1.0)
		# lamb2 = lamb*(2.0*J-1.0) - mu

		# Jk = J 
		Jk = 1
		mu2 = Jk*(mu - lamb*(J-1.0))
		lamb2 = Jk*(lamb*(2.0*J-1.0) - mu)

		# print lamb2, lamb

		# 4TH ORDER ELASTICITY TENSOR
		# H_Voigt = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
		MaterialArgs.H_Voigt[:,:,elem,gcounter] = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)

		MaterialArgs.H_VoigtSize = H_Voigt_k.shape[0]

		# MaterialArgs.H_Voigt[:,:,elem,gcounter] = H_Voigt

		# return H_Voigt
		return H_Voigt_k



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		Sigma_k = MaterialArgs.Sigma[:,:,elem,gcounter]
		H_Voigt_k = MaterialArgs.H_Voigt[:,:,elem,gcounter]


		strain = StrainTensors['strain'][gcounter]
		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 2*mu*strain + lamb*np.trace(strain)*I 
		# USE FASTER TRACE FUNCTION
		# return 2*mu*strain + lamb*trace(strain)*I  

		# Jk=J
		Jk=1
		mu2 = Jk*(mu - lamb*(J-1.0))
		lamb2 = Jk*(lamb*(2.0*J-1.0) - mu)
		# COMPUTE THE NEW STRESS AND STORE
		MaterialArgs.Sigma[:,:,elem,gcounter]  = Jk*(1.0*mu/J*b+(lamb*(J-1.0)-mu)*I)


		# STORE THIS VALUE 
		# MaterialArgs.Sigma[:,:,elem,gcounter] = Jk_sigma_k
		# return Jk_sigma_k + lamb2*trace(strain)*I + 2*mu2*strain

		# COMPUTE INCREMENTALLY LINEARISED STRESS BASED ON STRESS_K AND RETURN 
		return IncrementallyLinearisedStress(Sigma_k,H_Voigt_k,I,strain,StrainTensors['Gradu'][gcounter]), Sigma_k

		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
