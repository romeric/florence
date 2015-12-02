import numpy as np
from numpy import einsum
from Core.Supplementary.Tensors import *
from math import sqrt


#####################################################################################################
								# NEARLY INCOMPRESSIBLE MOONEY-RIVLIN
#####################################################################################################


class NearlyIncompressibleMooneyRivlin(object):
	"""	A nearly incompressible Mooney-Rivlin material model whose energy functional is given by:
		
			W(C,G,J**2) = alpha*J**(-2/3)*(C:I) + beta*J**(-2)*(G:I)**(3/2) + kappa/2*(J-1)**2

		Note that this energy is decomposed into deviatoric and volumetric components such that
		C:I and (G:I)**(3/2) contain only deviatoric contribution and the volumetric contribution
		is taken care of by the bulk modulus (kappa) term (J-1)**2 

		"""

	def __init__(self, ndim, MaterialArgs=None):
		super(NearlyIncompressibleMooneyRivlin, self).__init__()
		
		self.ndim = ndim
		self.nvar = self.ndim

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		self.gamma=1
		self.alpha = self.gamma*mu/2.
		self.beta = (mu - 2.*self.alpha)/3./sqrt(3.)
		# kappa = lamb+2.0*mu/3.0 
		self.kappa = lamb+4.0/3.0*self.alpha+2.0*sqrt(3.0)*self.beta # or

	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		alpha = self.alpha
		beta = self.beta
		kappa = self.kappa
		

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		# b=np.dot(F,F.T)
		H = J*np.linalg.inv(F).T
		g = np.dot(H,H.T)

		


		if self.ndim == 2:
			trb = trace(b)+1
			trg = trace(g)+J**2
		elif self.ndim == 3:
			trb = trace(b)
			trg = trace(g)


		# H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
		# 			4.*alpha/9.*J**(-5/3.)*trb*einsum('ij,kl',I,I) + \
		# 			2/3.*alpha*J**(-5/3.)*trb*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) + \
		# 	beta*J**(-3)*trg**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
		# 	3.*beta*J**(-3)*trg**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
		# 	6.*beta*J**(-3)*trg**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
		# 	3.*beta*J**(-3)*trg**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
		# 	kappa*(2.0*J-1)*einsum('ij,kl',I,I) - kappa*(J-1)*(einsum('ik,jl',I,I)+einsum('il,jk',I,I))			# #


		# WITH PRE-COMPUTED IDENTITY TENSORS
		H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
					4.*alpha/9.*J**(-5/3.)*trb*MaterialArgs.Iijkl + \
					2/3.*alpha*J**(-5/3.)*trb*MaterialArgs.Iikjl + \
			beta*J**(-3)*trg**(3./2.)*( MaterialArgs.Iijkl - MaterialArgs.Iikjl ) - \
			3.*beta*J**(-3)*trg**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
			6.*beta*J**(-3)*trg**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
			3.*beta*J**(-3)*trg**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
			kappa*(2.0*J-1)*MaterialArgs.Iijkl - kappa*(J-1)*MaterialArgs.Iikjl	


		H_Voigt = Voigt( H_Voigt ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		alpha = self.alpha
		beta = self.beta
		kappa = self.kappa

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		g = np.dot(H,H.T)
		bcross = trace(b)*b-np.dot(b,b)
		# b=np.dot(F,F.T)



		# stress = 2.*alpha*J**(-5/3.)*b - 2./3.*alpha*J**(-5/3.)*trace(b)*I + \
		# 		beta*J**(-3)*trace(g)**(3./2.)*I - 3*beta*J**(-3)*trace(g)**(1./2.)*g + \
		# 		+(kappa*(J-1.0))*I #####

		if self.ndim == 2:
			trb = trace(b)+1
			trg = trace(g)+J**2
		elif self.ndim == 3:
			trb = trace(b)
			trg = trace(g)

		stress = 2.*alpha*J**(-5/3.)*b - 2./3.*alpha*J**(-5/3.)*(trb)*I + \
				beta*J**(-3)*(trg)**(3./2.)*I - 3*beta*J**(-3)*(trg)**(1./2.)*g + \
				+(kappa*(J-1.0))*I 

		return stress

	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
