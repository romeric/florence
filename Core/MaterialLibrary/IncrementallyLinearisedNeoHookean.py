import numpy as np
from Core.Supplementary.Tensors import *


#####################################################################################################
								# Isotropic Linearised Model
#####################################################################################################


class IncrementallyLinearisedNeoHookean(object):
	"""docstring for IncrementallyLinearisedNeoHookean"""
	def __init__(self, ndim):
		super(IncrementallyLinearisedNeoHookean, self).__init__()
		self.ndim = ndim
	def Get(self):
		# self.nvar = self.ndim+1
		self.nvar = self.ndim
		self.modelname = 'IncrementallyLinearisedNeoHookean'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# H_Voigt_k = MaterialArgs.H_Voigt[:,:,elem,gcounter]
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])
		# J_k = np.copy(MaterialArgs.J[elem,counter])

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		# UPDATE MATERIAL CONSTANTS
		# mu2 = mu - lamb*(J-1.0)
		# lamb2 = lamb*(2.0*J-1.0) - mu

		Jk = J 
		# Jk = 1
		mu2 = Jk*(mu - lamb*(J-1.0))
		lamb2 = Jk*(lamb*(2.0*J-1.0) - mu)
		# lamb2 = Jk*(lamb*(2.0*J-1.0) )
		# mu2 = mu/detF- lamb*(detF-1.0)
		# lamb2 = lamb*(2*detF-1.0) 

		# print lamb2, lamb

		# 4TH ORDER ELASTICITY TENSOR
		# H_Voigt = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
		# MaterialArgs.H_Voigt[:,:,elem,gcounter] = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1) ##
		MaterialArgs.H_Voigt[:,:,elem,gcounter] = lamb2*MaterialArgs.IijIkl+mu2*MaterialArgs.IikIjl

		# MaterialArgs.J[elem,gcounter] = J

		MaterialArgs.H_VoigtSize = H_Voigt_k.shape[0]

		# MaterialArgs.H_Voigt[:,:,elem,gcounter] = H_Voigt
		# print J
		# print H_Voigt_k - MaterialArgs.H_Voigt[:,:,elem,gcounter]

		# return H_Voigt
		return H_Voigt_k



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		# Sigma_k = MaterialArgs.Sigma[:,:,elem,gcounter]
		# H_Voigt_k = MaterialArgs.H_Voigt[:,:,elem,gcounter]

		Sigma_k = np.copy(MaterialArgs.Sigma[:,:,elem,gcounter])
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])		


		strain = StrainTensors['strain'][gcounter]
		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 2*mu*strain + lamb*np.trace(strain)*I 
		# USE FASTER TRACE FUNCTION
		# return 2*mu*strain + lamb*trace(strain)*I  

		Jk=J
		# Jk=1
		# mu2 = Jk*(mu - lamb*(J-1.0))
		# lamb2 = Jk*(lamb*(2.0*J-1.0) - mu)
		# COMPUTE THE NEW STRESS AND STORE
		MaterialArgs.Sigma[:,:,elem,gcounter]  = Jk*(1.0*mu/J*b+(lamb*(J-1.0)-mu)*I)
		# MaterialArgs.Sigma[:,:,elem,gcounter]  = Jk*(1.0*mu/J*b+(lamb*(J-1.0)-mu/J)*I)


		# STORE THIS VALUE 
		# MaterialArgs.Sigma[:,:,elem,gcounter] = Jk_sigma_k
		# return Jk_sigma_k + lamb2*trace(strain)*I + 2*mu2*strain

		# COMPUTE INCREMENTALLY LINEARISED STRESS BASED ON STRESS_K AND RETURN 
		return IncrementallyLinearisedStress(Sigma_k,H_Voigt_k,I,strain,StrainTensors['Gradu'][gcounter]), Sigma_k

		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
