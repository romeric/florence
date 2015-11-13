import numpy as np
from Core.Supplementary.Tensors import *

#####################################################################################################
							# INCREMENTAL LINEAR ELASTIC ISOTROPIC MODEL
#####################################################################################################


class IncrementalLinearElastic(object):
	"""This is the linear elastic model with zero stresses and constant Hessian
		but the geometry is updated incrementally i.e. at every x=x_k
		"""

	def __init__(self, ndim):
		super(IncrementalLinearElastic, self).__init__()
		self.ndim = ndim

	def Get(self):
		# self.nvar = self.ndim+1
		self.nvar = self.ndim
		self.modelname = 'IncrementalLinearElastic'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
		# GET MATERIAL CONSTANTS
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb		

		MaterialArgs.H_VoigtSize = MaterialArgs.H_Voigt.shape[0]
		# RETURN THE 4TH ORDER ELASTICITY TENSOR (VOIGT FORM)
		# return lamb*MaterialArgs.IijIkl+mu*MaterialArgs.IikIjl
		return MaterialArgs.H_Voigt


	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		# RETURN STRESSES
		return np.zeros((2,2)),np.zeros((2,2))


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
