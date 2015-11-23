import numpy as np
from Core.Supplementary.Tensors import *
from numpy import einsum

#####################################################################################################
								# Anisotropic MooneyRivlin Model
#####################################################################################################


class TranservselyIsotropicHyperelastic(object):
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
		super(TranservselyIsotropicHyperelastic, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim
		self.gamma = gamma


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		N = np.array([-1.,0.]).reshape(2,1)
		FN = np.dot(F,N)[:,0]
		HN = np.dot(H,N)[:,0]
		innerFN = einsum('i,i',FN,FN)
		innerHN = einsum('i,i',HN,HN)
		outerHN = einsum('i,j',HN,HN)

		gamma = self.gamma
		alpha = mu/4./gamma 
		beta  = mu/4./gamma
		eta   = mu/3.
		ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
		lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta

		H_Voigt = 2.*gamma*beta/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
				(ut - lamb*(2.*J-1.) ) *einsum('ij,kl',I,I) + \
				(ut - lamb*(J-1.) ) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) 

		# n_1 = 1 
		# n_2 = 2
		# m_1 = 2
		# m_2 = 2
		# H_VoigtNCN_1 = self.TransverseHessianNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		# H_VoigtNGN_1 = self.TransverseHessianNGN(StrainTensors,n,eta,gamma,HN,innerHN,elem,gcounter)
		# H_VoigtNCN_1 = self.TransverseHessianNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		# H_VoigtNGN_1 = self.TransverseHessianNGN(StrainTensors,n,eta,gamma,HN,innerHN,elem,gcounter)
		# H_Voigt += H_VoigtNCN_1
		# H_Voigt += H_VoigtNGN_1

		for m in range(2,4):
			H_Voigt += self.TransverseHessianNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		for n in range(1,3):
			H_Voigt += self.TransverseHessianNGN(StrainTensors,n,eta,gamma,HN,innerHN,elem,gcounter)	

		H_Voigt = Voigt(H_Voigt ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt


	def TransverseHessianNCN(self,StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		H_VoigtNCN = 4.*(1-gamma)*eta/J *(m-1)*(innerFN)**(m-2)*einsum('i,j,k,l',FN,FN,FN,FN) 

		return H_VoigtNCN

	def TransverseHessianNGN(self,StrainTensors,n,eta,gamma,HN,innerHN,elem,gcounter):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		H_VoigtNGN = 4.*(1-gamma)*eta/J * ( n*(innerHN)**n * einsum('ij,kl',I,I) - \
				0.5*(innerHN)**n * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
				n*(innerHN)**(n-1)* ( einsum('ij,k,l',I,HN,HN) + einsum('i,j,kl',HN,HN,I) ) + \
				(n-1.)*(innerHN)**(n-2)* einsum('i,j,k,l',HN,HN,HN,HN) ) + \
				2.*(1-gamma)*eta/J *(innerHN)**(n-1)* ( einsum('il,j,k',I,HN,HN) + einsum('jl,i,k',I,HN,HN) + \
				einsum('ik,j,l',I,HN,HN) + einsum('jk,i,l',I,HN,HN) )

		return H_VoigtNGN






	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		N = np.array([-1.,0.]).reshape(2,1)
		# N = np.array([0.,0.]).reshape(2,1)
		FN = np.dot(F,N)
		HN = np.dot(H,N)[:,0]
		innerFN = np.dot(FN.T,FN)[0][0]
		innerHN = einsum('i,i',HN,HN)
		outerHN = einsum('i,j',HN,HN)

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		gamma = self.gamma
		alpha = mu/4./gamma 
		beta  = mu/4./gamma
		eta   = mu/3.
		ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
		lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


		if self.ndim == 3:
			trb = trace(b)
		elif self.ndim == 2:
			trb = trace(b) + 1


		stress = 2.*gamma*alpha/J*b + 2.*gamma*beta/J*(trb*b - np.dot(b,b)) - ut*I + lamb*(J-1.)*I

		# n=1
		# m=2
		# stressNCN_1 = self.CauchyStressNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		# stressNGN_1 = self.CauchyStressNGN(StrainTensors,n,eta,gamma,innerHN,outerHN,elem,gcounter)
		# stress += stressNCN_1 
		# stress += stressNGN_1

		for m in range(2,4):
			stress += self.CauchyStressNCN(StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter)
		for n in range(1,3):
			stress += self.CauchyStressNGN(StrainTensors,n,eta,gamma,innerHN,outerHN,elem,gcounter)

		# print stress
		return stress


	def CauchyStressNCN(self,StrainTensors,m,eta,gamma,FN,innerFN,elem,gcounter):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		return 2.*(1.- gamma)*eta/J*(innerFN)**(m-1)*np.dot(FN,FN.T)

	def CauchyStressNGN(self,StrainTensors,n,eta,gamma,innerHN,outerHN,elem,gcounter):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]

		return 2.*(1.- gamma)*eta/J*(innerHN)**(n-1)*(innerHN*I - outerHN)


	
