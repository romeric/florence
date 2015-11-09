import numpy as np
# from Core.Supplementary.Tensors.Tensors import *
from Core.Supplementary.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

#####################################################################################################
								# Isotropic Linear Model
#####################################################################################################


class LinearModel(object):
	"""docstring for LinearModel"""

	def __init__(self, ndim):
		super(LinearModel, self).__init__()
		self.ndim = ndim
		
	def Get(self):
		# self.nvar = self.ndim+1
		self.nvar = self.ndim
		self.modelname = 'LinearModel'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		#---------------------------------------------------------------------------------------------------------#
		# GET MATERIAL CONSTANTS
		# mu = MaterialArgs.mu
		# lamb = MaterialArgs.lamb

		# FOURTH ORDER ELASTICITY TENSOR
		# USING EINSUM
		# d = np.einsum
		# I = StrainTensors['I']
		# H_Voigt = Voigt(lamb*d('ij,kl',I,I)+mu*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
		# MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		# return H_Voigt
		#---------------------------------------------------------------------------------------------------------#
		
		MaterialArgs.H_VoigtSize = MaterialArgs.H_Voigt.shape[0]
		return MaterialArgs.H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):


		strain = StrainTensors['strain'][gcounter]
		I = StrainTensors['I']

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 2*mu*strain + lamb*np.trace(strain)*I 
		# USE FASTER TRACE FUNCTION
		return 2*mu*strain + lamb*trace(strain)*I  
		

	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
