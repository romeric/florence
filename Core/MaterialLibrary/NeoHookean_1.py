import numpy as np
from Core.Supplementary.Tensors.Tensors import *

#####################################################################################################
										# NeoHookean Material Model 2
#####################################################################################################


class NeoHookean_1(object):
	"""NeoHookean model with the following energy

		W(C) = u/2*C:I -u*J + lambda *(J-1)**2

		"""
	def __init__(self, ndim, MaterialArgs=None):
		super(NeoHookean_1, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		I = StrainTensors['I']
		detF = StrainTensors['J'][gcounter]

		mu2 = mu - lamb*(detF-1.0)
		lamb2 = lamb*(2*detF-1.0) - mu


		C_Voigt = lamb2*MaterialArgs.vIijIkl+mu2*MaterialArgs.vIikIjl

		MaterialArgs.H_VoigtSize = C_Voigt.shape[0]

		return C_Voigt


	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		return (lamb*(J-1.0)-mu)*I+1.0*mu/J*b


