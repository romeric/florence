import numpy as np
from Core.Supplementary.Tensors import *


#####################################################################################################
								# NEARLY INCOMPRESSIBLE MOONEY-RIVLIN
#####################################################################################################


class NearlyIncompressibleMooneyRivlin(object):
	"""	A nearly incompressible neo-Hookean material model whose energy functional is given by:
		
			W(C,G,J**2) = alpha*J**(-2/3)*(C:I) + beta*J**(-2)*(G:I)**(3/2) + kappa/2*(J-1)**2

		Note that this energy is decomposed into deviatoric and volumetric components such that
		C:I and (G:I)**(3/2) contain only deviatoric contribution and the volumetric contribution
		is taken care of by the bulk modulus (kappa) term (J-1)**2 

		"""

	def __init__(self, ndim):
		super(NearlyIncompressibleMooneyRivlin, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim

	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		einsum = np.einsum
		sqrt = np.sqrt

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		# b=np.dot(F,F.T)
		H = J*np.linalg.inv(F).T
		g = np.dot(H,H.T)

		# Update Lame constants
		gamma=1
		alpha = gamma*mu/2.
		beta = (mu - 2.*alpha)/3./sqrt(3.)
		# kappa = lamb+2.0*mu/3.0 
		kappa = lamb+4.0/3.0*alpha+2.0*sqrt(3.0)*beta # or

		# bcross = trace(b)*b-np.dot(b,b)
		# gcross = trace(g)*I-g

		# H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
		# 			4.*alpha/9.*J**(-5/3.)*trace(b)*einsum('ij,kl',I,I) + \
		# 			2/3.*alpha*J**(-5/3.)*trace(b)*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) + \
		# 	beta*J**(-3)*trace(g)**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
		# 	6.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
		# 	kappa*(2.0*J-1)*einsum('ij,kl',I,I) - kappa*(J-1)*(einsum('ik,jl',I,I)+einsum('il,jk',I,I))			# ####

		# H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
		# 			4.*alpha/9.*J**(-5/3.)*trace(b)*einsum('ij,kl',I,I) + \
		# 			2/3.*alpha*J**(-5/3.)*trace(b)*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) + \
		# 	beta*J**(-3)*trace(g)**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',g,I) + einsum('il,jk',I,g) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
		# 	kappa*(2.0*J-1)*einsum('ij,kl',I,I) - kappa*(J-1)*(einsum('ik,jl',I,I)+einsum('il,jk',I,I))			# #

		if self.ndim == 2:
			trb = trace(b)+1
			trg = trace(g)+J**2
		elif self.ndim == 3:
			trb = trace(b)
			trg = trace(g)

		H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
					4.*alpha/9.*J**(-5/3.)*trb*einsum('ij,kl',I,I) + \
					2/3.*alpha*J**(-5/3.)*trb*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) + \
			beta*J**(-3)*trg**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
			3.*beta*J**(-3)*trg**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
			6.*beta*J**(-3)*trg**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
			3.*beta*J**(-3)*trg**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
			kappa*(2.0*J-1)*einsum('ij,kl',I,I) - kappa*(J-1)*(einsum('ik,jl',I,I)+einsum('il,jk',I,I))			# #


		H_Voigt = Voigt( H_Voigt ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		sqrt = np.sqrt

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		g = np.dot(H,H.T)
		bcross = trace(b)*b-np.dot(b,b)
		# b=np.dot(F,F.T)

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		# kappa = lamb+2.0*mu/3.0
		alpha = mu/2.
		beta = (mu - 2.*alpha)/3./sqrt(3.)
		kappa = lamb+4.0/3.0*alpha+2.0*sqrt(3.0)*beta


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

		# print stress	

		return stress

	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
