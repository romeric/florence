import numpy as np
from Core.Supplementary.Tensors.Tensors import *

#####################################################################################################
										# NeoHookean Material Model 2
#####################################################################################################


class NeoHookean_2(object):
	"""docstring for NeoHookean"""
	def __init__(self, ndim):
		super(NeoHookean_2, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'NeoHookean_2'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		I = StrainTensors['I']
		detF = StrainTensors['J'][gcounter]

		mu2 = mu/detF- lamb*(detF-1.0)
		lamb2 = lamb*(2*detF-1.0) 


		# d = np.einsum
		# C = lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I) + d('il,jk',I,I))
		# C_Voigt = Voigt( C ,1)
		C_Voigt = lamb2*MaterialArgs.IijIkl+mu2*MaterialArgs.IikIjl

		MaterialArgs.H_VoigtSize = C_Voigt.shape[0]

		return C_Voigt

	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		return 1.0*mu/J*b + (lamb*(J-1.0)-mu/J)*I


