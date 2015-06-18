import numpy as np
from Core.Supplementary.Tensors.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic AnisotropicMooneyRivlin_1 Model
#####################################################################################################


class AnisotropicMooneyRivlin_1(object):
	"""docstring for AnisotropicMooneyRivlin_1"""
	def __init__(self, ndim):
		super(AnisotropicMooneyRivlin_1, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'AnisotropicMooneyRivlin_1'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors.I
		J = StrainTensors.J
		b = StrainTensors.b
		# H_ = StrainTensors.H
		# G = np.dot(H_.T,H_)
		# g = np.dot(H_,H_.T)

		# Update Lame constants
		mu2 = mu - lamb*(J-1.0)
		lamb2 = lamb*(2.0*J-1.0) - mu

		H_Voigt = Voigt( lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I 


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors.I.shape[0]
		return np.zeros((ndim,1))
