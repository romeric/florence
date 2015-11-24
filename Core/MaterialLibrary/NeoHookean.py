import numpy as np
from Core.Supplementary.Tensors import *

#####################################################################################################
										# NeoHookean Material Model 1
#####################################################################################################


class NeoHookean(object):
	"""docstring for NeoHookean"""

	def __init__(self, ndim):
		super(NeoHookean, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		detF = StrainTensors.J

		mu1 = 1.0*(mu-lamb*np.log(detF))/detF 
		lamb1 = 1.0*lamb/detF 

		delta = np.eye(ndim,ndim)
		# Hessian is the fourth order elasticity tensor in this case
		C = np.zeros((ndim,ndim,ndim,ndim))
		for i in range(0,ndim):
			for j in range(0,ndim):
				for k in range(0,ndim):
					for l in range(0,ndim):
						# C[i,j,k,l] += lamb1*delta[i,j]*delta[k,l]+2.0*mu1*delta[i,k]*delta[j,l]
						C[i,j,k,l] += lamb1*delta[i,j]*delta[k,l]+mu1*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k])


		C_Voigt = Voigt(C,1)


		MaterialArgs.H_VoigtSize = C_Voigt.shape[0]
		
		return C_Voigt


	def CauchyStress(self,MaterialArgs,StrainTensors):

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		return 1.0*((mu/J)*(b-I)+(lamb/J)*np.log(J)*I)

