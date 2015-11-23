import numpy as np
from Core.Supplementary.Tensors import *

#####################################################################################################
							# INCREMENTALLY LINEARISED ISOTROPIC MODEL
#####################################################################################################


class IncrementallyLinearisedNeoHookean(object):
	"""This is the incrementally linearised version of the neo-Hookean 
		material model:

			W(C) = lambda*(C:I)-mu*lnJ+lamba/2*(J-1)**2

		For incrementally linearised models, stress and Hessian have to
		be evaluated at each step. The Hessian at the current step (k+1)
		is the Hessian at the previous step (k) and the stress at the 
		current step is give by:

			sigma_k+1 = sigma_k (I+strain) + Gradu : Hessian_k

		"""

	def __init__(self, ndim):
		super(IncrementallyLinearisedNeoHookean, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim

	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# GET THE JACOBIAN AND HESSIAN FROM THE PREVIOUS STEP - NOTE THAT A COPY HAS TO BE MADE
		# H_Voigt_k = MaterialArgs.H_Voigt[:,:,elem,gcounter]
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])
		J_k = np.copy(MaterialArgs.J[elem,gcounter])

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		# COMPUTE UPDATED MATERIAL PROPERTIES
		mu2 = mu/J_k - lamb*(J_k-1.0)
		lamb2 = lamb*(2*J_k-1.0) 

		I = StrainTensors['I']
		

		# 4TH ORDER ELASTICITY TENSOR
		MaterialArgs.H_Voigt[:,:,elem,gcounter] = lamb2*MaterialArgs.IijIkl+mu2*MaterialArgs.IikIjl
		# STORE THE JACOBIAN FOR THE CURRENT STEP
		MaterialArgs.J[elem,gcounter] = StrainTensors['J'][gcounter]
		# STORE SIZE OF HESSIAN - NEEDED ONLY ONCE
		MaterialArgs.H_VoigtSize = H_Voigt_k.shape[0] 

		# THE HESSIAN IN THE CURRENT STEP IS THE HESSIAN FROM THE PREVIOUS STEP
		return H_Voigt_k



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		# GET STRESSES, HESSIANS AND JACOBIANS FROM THE PREVIOUS STEP - NOTE THAT A COPY HAS TO BE MADE
		Sigma_k = np.copy(MaterialArgs.Sigma[:,:,elem,gcounter])
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])
		J_k = np.copy(MaterialArgs.J[elem,gcounter])		

		strain = StrainTensors['strain'][gcounter]
		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# COMPUTE THE STRESS AT THE CURRENT STEP AND STORE
		MaterialArgs.Sigma[:,:,elem,gcounter]  = 1.0*mu/J_k*b+(lamb*(J_k-1.0)-mu/J_k)*I

		# COMPUTE INCREMENTALLY LINEARISED STRESS BASED ON STRESS_K AND RETURN
		return IncrementallyLinearisedStress(Sigma_k,H_Voigt_k,I,strain,StrainTensors['Gradu'][gcounter]), Sigma_k
		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
