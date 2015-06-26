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

		# WHEN AN IDENTITY TENSOR IS USED TO DESCRIBE THE HESSIAN OF A MATERIAL MODEL, WE DON'T TO CALL EINSUM
		# 2D
		# H_Voigt = lamb*np.array([[1.,1,0],[1,1.,0],[0,0,0.]]) + mu*np.array([[2.,0,0],[0,2.,0],[0,0,1.]]) 
		# 3D
		# block_1 = np.zeros((6,6),dtype=np.float64); block_1[:2,:2] = np.ones((3,3))
		# block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
		# H_Voigt = lamb*block_1 + mu*block_2
		
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
