import numpy as np
from Core.Supplementary.Tensors import *
from numpy import einsum

#####################################################################################################
								# Anisotropic MooneyRivlin Model
#####################################################################################################


class BonetTranservselyIsotropicHyperElastic(object):
	"""A compressible transervely isotropic model based on Bonet 1998.
		Material model is not polyconvex
	"""

	def __init__(self, ndim, MaterialArgs=None):
		super(BonetTranservselyIsotropicHyperElastic, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Get material constants (5 in this case)
		E = MaterialArgs.E
		E_A = MaterialArgs.E_A
		G_A = MaterialArgs.G_A
		v = MaterialArgs.nu
		mu = MaterialArgs.mu

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		# N = np.array([-1.,0.]).reshape(2,1)
		N = MaterialArgs.AnisotropicOrientations[elem,:,None]
		FN = np.dot(F,N)[:,0]

		E = MaterialArgs.E
		E_A = MaterialArgs.E_A
		G_A = MaterialArgs.G_A
		v = MaterialArgs.nu
		mu = MaterialArgs.mu

		alpha = E/8.0/(1.+v)
		beta = E/8.0/(1.+v)
		eta_1 = 4.*alpha - G_A
		lamb = - (3*E)/(2*(v + 1)) - (E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A))
		eta_2 = E/(4*(v + 1)) - (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) + \
					(E*(- E*v**2 + E_A))/(4*(v + 1)*(2*E*v**2 + E_A*v - E_A))
		gamma = (E_A**2*(v - 1))/(8*(2*E*v**2 + E_A*v - E_A)) - G_A/2 + \
				 	(E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) - \
				 	(E*(- E*v**2 + E_A))/(8*(v + 1)*(2*E*v**2 + E_A*v - E_A))

		ut = 2*alpha + 4*beta


		H_Voigt = 2.*beta/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
				lamb*(2.*J-1.) *einsum('ij,kl',I,I) + \
				(ut/J - lamb*(J-1.) ) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
				4.*eta_2/J*( einsum('ij,k,l',b,FN,FN) + einsum('i,j,kl',FN,FN,b)  ) + \
				8.*gamma/J*( einsum('i,j,k,l',FN,FN,FN,FN) ) - \
				eta_1/J*( einsum('jk,i,l',b,FN,FN) + einsum('ik,j,l',b,FN,FN)  + 
						einsum('jl,i,k',b,FN,FN) + einsum('il,j,k',b,FN,FN) )


		H_Voigt = Voigt(H_Voigt ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		# N = np.array([-1.,0.]).reshape(2,1)
		N = MaterialArgs.AnisotropicOrientations[elem,:,None]
		FN = np.dot(F,N)
		# HN = np.dot(H,N)[:,0]
		innerFN = np.dot(FN.T,FN)[0][0]
		outerFN = np.dot(FN,FN.T)
		# innerFN = einsum('i,i',FN,FN)
		# outerFN = einsum('i,j',FN,FN)
		bFN = np.dot(b,FN)
		# print FN.shape, innerFN.shape, outerFN.shape

		E = MaterialArgs.E
		E_A = MaterialArgs.E_A
		G_A = MaterialArgs.G_A
		v = MaterialArgs.nu
		mu = MaterialArgs.mu

		alpha = E/8.0/(1.+v)
		beta = E/8.0/(1.+v)
		eta_1 = 4.*alpha - G_A
		lamb = - (3*E)/(2*(v + 1)) - (E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A))
		eta_2 = E/(4*(v + 1)) - (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) + \
					(E*(- E*v**2 + E_A))/(4*(v + 1)*(2*E*v**2 + E_A*v - E_A))
		gamma = (E_A**2*(v - 1))/(8*(2*E*v**2 + E_A*v - E_A)) - G_A/2 + \
				 	(E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) - \
				 	(E*(- E*v**2 + E_A))/(8*(v + 1)*(2*E*v**2 + E_A*v - E_A))

		ut = 2*alpha + 4*beta



		if self.ndim == 3:
			trb = trace(b)
		elif self.ndim == 2:
			trb = trace(b) + 1


		stress = 2.*alpha/J*b + 2.*beta/J*(trb*b - np.dot(b,b)) - ut/J*I + lamb*(J-1.)*I + \
			2.*eta_1/J*outerFN + 2.*eta_2/J*(innerFN-1.)*b + 2.*eta_2/J*(trb-3.)*outerFN + \
			4.*gamma/J*(innerFN-1.)*outerFN - eta_1/J *(np.dot(bFN,FN.T)+np.dot(FN,bFN.T))



		return stress




	
