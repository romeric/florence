import numpy as np
# from Core.Supplementary.Tensors.Tensors import *
from Core.Supplementary.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic MooneyRivlin Model
#####################################################################################################


class MooneyRivlin(object):
	"""	Polyconvex compressible MooneyRivlin material model based on the energy:
		W = alpha*C:I+beta*G:I+lambda/2*(J-1)**2-4*beta*J-2*lnJ - (3*alpha-beta)
		where at the origin (alpha + beta) = mu/2"""
	def __init__(self, ndim):
		super(MooneyRivlin, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'MooneyRivlin'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		alpha = mu/4.0
		beta = mu/4.0

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		# H_ = StrainTensors.H
		# G = np.dot(H_.T,H_)
		# g = np.dot(H_,H_.T)


		H_Voigt = Voigt( 4.0*beta/J*d('ij,kl',b,b) - 2.0*beta/J*( d('ik,jl',b,b) + d('il,jk',b,b) ) +\
			(lamb+4.0*beta+4.0*alpha/J)*d('ij,kl',I,I) + 2.0*(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*d('ij,kl',I,I) -\
			1.0*(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1) 
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		
		alpha = mu/4.0
		beta = mu/4.0

		return 2.0*alpha/J*b+2.0*beta/J*(trace(b)*b - np.dot(b,b)) + (lamb*(J-1.0)-4.0*beta-2.0*alpha/J)*I 


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))
