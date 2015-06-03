import numpy as np
# from Core.Supplementary.Tensors.Tensors import *
from Core.Supplementary.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic Linear Model
#####################################################################################################


class Incrementally_Linearised_NeoHookean(object):
	"""docstring for Incrementally_Linearised_NeoHookean"""
	def __init__(self, ndim):
		super(Incrementally_Linearised_NeoHookean, self).__init__()
		self.ndim = ndim
	def Get(self):
		# self.nvar = self.ndim+1
		self.nvar = self.ndim
		self.modelname = 'Incrementally_Linearised_NeoHookean'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum
		I = StrainTensors.I

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# Fourth order elasticity tensor
		H_Voigt = Voigt(lamb*d('ij,kl',I,I)+mu*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx):


		strain = StrainTensors.strain
		I = StrainTensors.I

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 2*mu*strain + lamb*np.trace(strain)*I 
		# USE FASTER TRACE FUNCTION
		return 2*mu*strain + lamb*trace(strain)*I  
		


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):

		ndim = StrainTensors.I.shape[0]
		
		return np.zeros((ndim,1))
