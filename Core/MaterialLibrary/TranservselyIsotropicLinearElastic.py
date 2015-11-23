import numpy as np
from Core.Supplementary.Tensors import *
from numpy import einsum

#####################################################################################################
								# Anisotropic MooneyRivlin Model
#####################################################################################################


class TranservselyIsotropicLinearElastic(object):
	"""A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
		The energy is given by:

			W(C) =  gamma * ( alpha*(C:I) + beta*(G:I) ) + 
					eta*(1-alpha)*( (N C N)**2 + N G N) - ut*J + lambda/2*(J-1)**2

			ut = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta  # for the stress to be 
				zero at the origin

		the parameter "gamma" controls the amount of anisotropy and the vector N(ndim,1) is 
		the direction of anisotropy

	"""

	def __init__(self, ndim, gamma=0.5):
		super(TranservselyIsotropicLinearElastic, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim
		self.gamma = gamma


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		N = np.array([-1.,0.]).reshape(2,1)

		gamma = self.gamma
		alpha = mu/4./gamma 
		beta  = mu/4./gamma
		eta   = mu/3.
		ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
		lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta

		H_Voigt = 2.*gamma*beta* ( 2.0*einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
				(ut - lamb ) *einsum('ij,kl',I,I) + \
				ut * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) 


		for m in range(2,4):
			H_Voigt += self.TransverseHessianNCN(StrainTensors,m,eta,gamma,N,elem,gcounter)
		for n in range(1,3):
			H_Voigt += self.TransverseHessianNGN(StrainTensors,n,eta,gamma,N,elem,gcounter)	

		H_Voigt = Voigt(H_Voigt ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt


	def TransverseHessianNCN(self,StrainTensors,m,eta,gamma,N,elem,gcounter):

		I = StrainTensors['I']
		N = N[:,0]

		H_VoigtNCN = 4.*(1-gamma)*eta *(m-1)*einsum('i,j,k,l',N,N,N,N) 

		return H_VoigtNCN

	def TransverseHessianNGN(self,StrainTensors,n,eta,gamma,N,elem,gcounter):

		I = StrainTensors['I']
		N = N[:,0]

		H_VoigtNGN = 4.*(1-gamma)*eta * ( n* einsum('ij,kl',I,I) - \
				0.5 * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
				n * ( einsum('ij,k,l',I,N,N) + einsum('i,j,kl',N,N,I) ) + \
				(n-1.)* einsum('i,j,k,l',N,N,N,N) ) + \
				2.*(1-gamma)*eta * ( einsum('il,j,k',I,N,N) + einsum('jl,i,k',I,N,N) + \
				einsum('ik,j,l',I,N,N) + einsum('jk,i,l',I,N,N) )

		return H_VoigtNGN






	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		N = np.array([-1.,0.]).reshape(2,1)
		# N = np.array([0.,0.]).reshape(2,1)

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		gamma = self.gamma
		alpha = mu/4./gamma 
		beta  = mu/4./gamma
		eta   = mu/3.
		ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
		lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


		stress = (2.*gamma*alpha + 4.*gamma*beta - ut ) * I

		for m in range(2,4):
			stress += self.CauchyStressNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		for n in range(1,3):
			stress += self.CauchyStressNGN(StrainTensors,n,eta,gamma,innerHN,outerHN,elem,gcounter)

		# print stress
		return stress


	def CauchyStressNCN(self,StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter):

		I = StrainTensors['I']

		return 2.*(1.- gamma)*eta*np.dot(N,N.T)

	def CauchyStressNGN(self,StrainTensors,n,eta,gamma,innerHN,outerHN,elem,gcounter):

		I = StrainTensors['I']

		return 2.*(1.- gamma)*eta*(I - np.dot(N,N.T))


	
