import numpy as np
from Core.Supplementary.Tensors.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# NEARLY INCOMPRESSIBLE NEOHOOKEAN
								# W = mu/2*C:I + k/2*(J-1)**2								
#####################################################################################################


class NearlyIncompressibleNeoHookean(object):
	"""	A nearly incompressible neo-Hookean material model whose energy functional is given by:
		W = mu/2*C:I + k/2*(J-1)**2
		"""
	def __init__(self, ndim):
		super(NearlyIncompressibleNeoHookean, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'NearlyIncompressibleNeoHookean'
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
		H_ = StrainTensors.H
		G = np.dot(H_.T,H_)
		g = np.dot(H_,H_.T)

		# Update Lame constants
		kappa = lamb+2.0*mu/3.0

		H_Voigt = Voigt( kappa*(2.0*J-1)*d('ij,kl',I,I)-kappa*(J-1)*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		kappa = lamb+2.0*mu/3.0

		return 1.0*mu/J*b+(kappa*(J-1.0))*I 


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors.I.shape[0]
		return np.zeros((ndim,1))
