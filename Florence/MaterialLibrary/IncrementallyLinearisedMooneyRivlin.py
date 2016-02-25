import numpy as np
from numpy import einsum
from Florence.Tensor import trace

#####################################################################################################
							# INCREMENTALLY LINEARISED ISOTROPIC MODEL
#####################################################################################################


class IncrementallyLinearisedMooneyRivlin(object):
	"""This is the incrementally linearised version of the Mooney-Rivlin
		material model:

			W = alpha*C:I+beta*G:I+lambda/2*(J-1)**2-4*beta*J-2*alpha*lnJ - (3*alpha-beta)

		For incrementally linearised models, stress and Hessian have to
		be evaluated at each step. The Hessian at the current step (k+1)
		is the Hessian at the previous step (k) and the stress at the 
		current step is give by:

			sigma_k+1 = sigma_k (I+strain) + Gradu : Hessian_k

		"""

	def __init__(self, ndim):
		super(IncrementallyLinearisedMooneyRivlin, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# GET THE JACOBIAN AND HESSIAN FROM THE PREVIOUS STEP - NOTE THAT A COPY HAS TO BE MADE
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])
		

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		b = StrainTensors['b'][gcounter]
		J = StrainTensors['J'][gcounter]

		alpha = mu/4.0
		beta = mu/4.0

		# HESSIAN AT THE CURRENT STEP - ALL WITH NEWLY EVALUTED KINEMATIC MEASURES
		H_Voigt = 2.0*beta/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
			(lamb*(2.0*J-1.0) -4.0*beta)*einsum('ij,kl',I,I) - \
			(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )

		# 4TH ORDER ELASTICITY TENSOR
		MaterialArgs.H_Voigt[:,:,elem,gcounter] = Voigt(H_Voigt,1) 

		# STORE SIZE OF HESSIAN - NEEDED ONLY ONCE
		MaterialArgs.H_VoigtSize = H_Voigt_k.shape[0] 


		# THE HESSIAN IN THE CURRENT STEP IS THE HESSIAN FROM THE PREVIOUS STEP
		return H_Voigt_k



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		# GET STRESSES, HESSIANS AND JACOBIANS FROM THE PREVIOUS STEP - NOTE THAT A COPY HAS TO BE MADE
		Sigma_k = np.copy(MaterialArgs.Sigma[:,:,elem,gcounter])
		H_Voigt_k = np.copy(MaterialArgs.H_Voigt[:,:,elem,gcounter])
		strain = StrainTensors['strain'][gcounter]

		I = StrainTensors['I']
		b = StrainTensors['b'][gcounter]
		J = StrainTensors['J'][gcounter]

		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		alpha = mu/4.0
		beta = mu/4.0

		MaterialArgs.Sigma[:,:,elem,gcounter] = 2.0*alpha/J*b+2.0*beta/J*(trace(b)*b - np.dot(b,b)) + \
												(lamb*(J-1.0)-4.0*beta-2.0*alpha/J)*I 


		# COMPUTE INCREMENTALLY LINEARISED STRESS BASED ON STRESS_K AND RETURN 
		return IncrementallyLinearisedStress(Sigma_k,H_Voigt_k,I,strain,StrainTensors['Gradu'][gcounter]), Sigma_k

		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
