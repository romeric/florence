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

	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'NearlyIncompressibleMooneyRivlin'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		einsum = np.einsum
		sqrt = np.sqrt

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		# b=np.dot(F,F.T)
		H = J*np.linalg.inv(F).T
		g = np.dot(H,H.T)
		# H_ = StrainTensors.H
		# G = np.dot(H_.T,H_)
		# g = np.dot(H_,H_.T)
		# if StrainTensors['F'].shape[1] != 2:
			# print 'wow'
		# print StrainTensors['F'][0].shape

		# Update Lame constants
		kappa = lamb+2.0*mu/3.0
		alpha = mu
		beta = mu

		bcross = trace(b)*b-np.dot(b,b)
		gcross = trace(g)*I-g
		# print sqrt(5)

		# H_Voigt =  -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
		# 			4.*alpha/9.*J**(-5/3.)*trace(b)*einsum('ij,kl',I,I) + \
		# 			2/3.*alpha*J**(-5/3.)*trace(b)*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) )

		# H_Voigt =  -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
		# 	4.*alpha/9.*J**(-5/3.)*trace(b)*einsum('ij,kl',I,I) + \
		# 	2/3.*alpha*J**(-5/3.)*trace(b)*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) - \
		# 	6.*beta*J**(-3)*sqrt(trace(g))*einsum('ij,kl',bcross,I) + \
		# 	3.*beta*J**(-3)*(trace(g))**(-0.5)*einsum('ij,kl',bcross,gcross) + \
		# 	6*beta*J**(-3)*sqrt(trace(g))*einsum('ij,kl',b,b) - \
		# 	3*beta*J**(-3)*sqrt(trace(g))*( einsum('ik,jl',b,b) + einsum('il,jk',b,b) ) + \
		# 	4*beta*J**(-3)*(trace(g))**(1.5)*einsum('ij,kl',I,I) - \
		# 	6*beta*J**(-3)*sqrt(trace(g))*einsum('ij,kl',I,gcross) + \
		# 	2*beta*J**(-3)*(trace(g))**(1.5) *( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )  

		# H_Voigt =  -6.*beta*J**(-3.)*sqrt(trace(g))*einsum('ij,kl',bcross,I) + \
		# 	3.*beta*J**(-3.)*(trace(g))**(-0.5)*einsum('ij,kl',bcross,gcross) + \
		# 	6*beta*J**(-3.)*sqrt(trace(g))*einsum('ij,kl',b,b) - \
		# 	3.*beta*J**(-3.)*sqrt(trace(g))*( einsum('ik,jl',b,b) + einsum('il,jk',b,b) ) + \
		# 	4.*beta*J**(-3.)*(trace(g))**(1.5)*einsum('ij,kl',I,I) - \
		# 	6.*beta*J**(-3.)*sqrt(trace(g))*einsum('ij,kl',I,gcross) + \
		# 	2.*beta*J**(-3.)*(trace(g))**(1.5) *( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )  	

		# H_Voigt =  3.*beta*J**(-3.)*(trace(g))**(-0.5)*einsum('ij,kl',bcross,gcross) + \
		# 	6.*beta*J**(-3.)*sqrt(trace(g))*einsum('ij,kl',b,b) - \
		# 	3.*beta*J**(-1.)*sqrt(trace(g))*( einsum('ik,jl',b,b) + einsum('il,jk',b,b) )

		# print trace(g), g[0,0]+g[1,1] 


		# H_Voigt =  beta*J**(-3)*trace(g)**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',g,I) + einsum('il,jk',I,g) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(-1./2.)*( einsum('ij,kl',g,g) ) 	

		# H_Voigt =  beta*J**(-3)*trace(g)**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
		# 	3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
		# 	6.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
		# 	3.*beta*J**(-3)*trace(g)**(-1./2.)*( einsum('ij,kl',g,g) ) 				# #

		H_Voigt = -4/3.*alpha*J**(-5/3.)*( einsum('ij,kl',b,I) + einsum('ij,kl',I,b) ) + \
					4.*alpha/9.*J**(-5/3.)*trace(b)*einsum('ij,kl',I,I) + \
					2/3.*alpha*J**(-5/3.)*trace(b)*( einsum('il,jk',I,I) + einsum('ik,jl',I,I) ) + \
			beta*J**(-3)*trace(g)**(3./2.)* ( einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) - \
			3.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ij,kl',I,g) + einsum('ij,kl',g,I) ) + \
			6.*beta*J**(-3)*trace(g)**(1./2.)*( einsum('ik,jl',I,g) + einsum('il,jk',g,I) ) + \
			3.*beta*J**(-3)*trace(g)**(-1./2.)*( einsum('ij,kl',g,g) ) 	+ \
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
		kappa = lamb+2.0*mu/3.0
		alpha = mu
		beta = mu

		# return 2.*alpha*J**(-5/3.)*b - 2./3.*alpha*J**(-5/3.)*trace(b)*I

		# return 2.*alpha*J**(-5/3.)*b - 2./3.*alpha*J**(-5/3.)*trace(b)*I + \
		# 	   3*beta*J**(-3)*sqrt(trace(g))*bcross-2*beta*J**(-3)*(trace(g))**(1.5)*I

		# return 3.*beta*J**(-3.)*sqrt(trace(g))*bcross-2*beta*J**(-3.)*(trace(g))**(1.5)*I

		# return -6.*sqrt(3.)*I+3.*beta*sqrt(trace(g))*bcross	


		# return beta*J**(-3)*trace(g)**(3./2.)*I - 3*beta*J**(-3)*trace(g)**(1./2.)*g	
		return 2.*alpha*J**(-5/3.)*b - 2./3.*alpha*J**(-5/3.)*trace(b)*I + \
				beta*J**(-3)*trace(g)**(3./2.)*I - 3*beta*J**(-3)*trace(g)**(1./2.)*g + \
				+(kappa*(J-1.0))*I 	

	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
