import numpy as np
from Core.Supplementary.Tensors import *
from numpy import einsum

#####################################################################################################
								# Anisotropic MooneyRivlin Model
#####################################################################################################


class AnisotropicMooneyRivlin(object):
	"""A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
		The energy is given by:

			W(C) =  alpha*(u1/2*(C:I) +u2/2*(G:I)) + 
					u3/2(1-alpha)*(N C N + N G N) - ut lnJ + lambda/2*(J-1)**2

			ut = alpha*u1+2*alpha*u2+u3*(1-alpha) # for the stress to be zero at the origin

		the parameter "alpha" controls the amount of anisotropy and the vector N(ndim,1) is 
		the direction of anisotropy

	"""

	def __init__(self, ndim):
		super(AnisotropicMooneyRivlin, self).__init__()
		self.ndim = ndim

	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'AnisotropicMooneyRivlin'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['F'][gcounter]
		H = J*np.linalg.inv(F).T
		# g = np.dot(H,H.T)
		N = np.array([-1.,0.]).reshape(2,1)
		# N = np.array([0.,0.]).reshape(2,1)
		V = np.dot(H,N)
		innerVV = np.dot(V.T,V)[0][0]
		outerVV = np.dot(V,V.T)
		V = V[:,0]

		# mu2 = mu - lamb*(J-1.0)
		# lamb2 = lamb*(2.0*J-1.0) - mu

		# FIX ALPHA
		# alpha = 0.5
		alpha = 0.5

		u1=mu/2. 
		u2=mu/2.
		u3=mu/2.
		ut = alpha*(u1+2.0*u2)+(1.-alpha)*u3

		

		H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
			2.0*(1.-alpha)*u3/J*(innerVV*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerVV) - einsum('ij,kl',outerVV,I) - \
			0.5*innerVV*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + einsum('ik,j,l',I,V,V) + einsum('i,k,jl',V,V,I) ) + \
			ut/J*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + lamb*(2.0*J-1.0)*einsum('ij,kl',I,I) - \
			lamb*(J-1.)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) )

		# H_Voigt = u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
		# 	ut/J*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + lamb*(2.0*J-1.0)*einsum('ij,kl',I,I) - \
		# 	lamb*(J-1.)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) )

		H_Voigt = Voigt( H_Voigt ,1)
		# print H_Voigt
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		I = StrainTensors['I']
		J = StrainTensors['J'][gcounter]
		b = StrainTensors['b'][gcounter]
		F = StrainTensors['b'][gcounter]
		H = J*np.linalg.inv(F).T
		N = np.array([-1.,0.]).reshape(2,1)
		# N = np.array([0.,0.]).reshape(2,1)
		V = np.dot(H,N)
		innerVV = np.dot(V.T,V)
		outerVV = np.dot(V,V.T)
		FN = np.dot(F,N)

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		# FIX ALPHA
		alpha = 1.0

		u1=mu/2. 
		u2=mu/2.
		u3=mu/2.
		ut = alpha*(u1+2.0*u2)+(1.-alpha)*u3

		stress = alpha*u1/J*b + alpha*u2/J*(trace(b)*b - np.dot(b,b)) + \
				u3/J*(1.-alpha)*np.dot(FN.T,FN) + u3/J*(1.-alpha)*(innerVV*I-outerVV) - \
				ut/J*I + lamb*(J-1)*I

		# stress = u1/J*b + u2/J*(trace(b)*b - np.dot(b,b)) - ut/J*I + lamb*(J-1)*I

		# print stress
		return stress

	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		ndim = StrainTensors['I'].shape[0]
		return np.zeros((ndim,1))


# import numpy as np
# from numpy import einsum
# from Core.Supplementary.Tensors import *


# #####################################################################################################
# 								# Isotropic AnisotropicMooneyRivlin Model
# #####################################################################################################


# class AnisotropicMooneyRivlin(object):
# 	"""	Polyconvex compressible MooneyRivlin material model based on the energy:

# 			W = alpha*C:I+beta*G:I+lambda/2*(J-1)**2-4*beta*J-2*alpha*lnJ - (3*alpha-beta)

# 		where at the origin (alpha + beta) = mu/2
# 		"""

# 	def __init__(self, ndim):
# 		super(AnisotropicMooneyRivlin, self).__init__()
# 		self.ndim = ndim

# 	def Get(self):
# 		self.nvar = self.ndim
# 		self.modelname = 'AnisotropicMooneyRivlin'
# 		return self.nvar, self.modelname

# 	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):


# 		# GET MATERIAL CONSTANTS 
# 		mu = MaterialArgs.mu
# 		lamb = MaterialArgs.lamb



# 		I = StrainTensors['I']
# 		J = StrainTensors['J'][gcounter]
# 		b = StrainTensors['b'][gcounter]
# 		F = StrainTensors['F'][gcounter]
# 		H = J*np.linalg.inv(F).T
# 		N = np.array([-1.,0.]).reshape(2,1)
# 		V = np.dot(H,N)
# 		innerVV = np.dot(V.T,V)[0][0]
# 		outerVV = np.dot(V,V.T)
# 		V = V[:,0]

# 		# gamma= 0.0
# 		# u3=beta/5.


# 		# H_Voigt = 2.0*beta/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
# 		# 	(lamb*(2.0*J-1.0) -4.0*beta)*einsum('ij,kl',I,I) - \
# 		# 	(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
# 		# 	4.0*(1.-gamma)*u3/J*(innerVV*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerVV) - einsum('ij,kl',outerVV,I) - \
# 		# 	0.5*innerVV*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + einsum('ik,j,l',I,V,V) + einsum('i,k,jl',V,V,I) )
# 		# H_Voigt = Voigt(H_Voigt,1) 

# 		u2 = mu
# 		ut = mu - u2/2.
# 		alpha  = 1.0

# 		H_Voigt = alpha*u2/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
# 			ut*einsum('ij,kl',I,I) + ut*( einsum('ik,jl',I,I) - einsum('il,jk',I,I) )
# 		H_Voigt = Voigt(H_Voigt,1) 

# 		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

# 		return H_Voigt



# 	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

# 		I = StrainTensors['I']
# 		J = StrainTensors['J'][gcounter]
# 		b = StrainTensors['b'][gcounter]

# 		mu = MaterialArgs.mu
# 		lamb = MaterialArgs.lamb

# 		F = StrainTensors['F'][gcounter]
# 		H = J*np.linalg.inv(F).T
# 		N = np.array([-1.,0.]).reshape(2,1)
# 		V = np.dot(H,N)
# 		innerVV = np.dot(V.T,V)[0][0]
# 		outerVV = np.dot(V,V.T)
# 		V = V[:,0]
# 		FN = np.dot(F,N)
		
# 		u2 = mu
# 		ut = mu - u2/2.
# 		alpha  = 1.0
		
# 		# stress = 2.0*alpha/J*b+2.0*beta/J*(trace(b)*b - np.dot(b,b)) + (lamb*(J-1.0)-4.0*beta-2.0*alpha/J)*I  + \
# 		# 	2.0*u3/J*(1.-gamma)*np.dot(FN.T,FN) + 2.0*u3/J*(1.-gamma)*(innerVV*I-outerVV) - 2.0*u3/J*(1.-gamma)*I

# 		stress = alpha*u2/J*(trace(b)*b - np.dot(b,b)) -ut * I
			

# 		return stress


# 	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
# 		ndim = StrainTensors['I'].shape[0]
# 		return np.zeros((ndim,1))

