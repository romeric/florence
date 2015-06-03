import numpy as np
from Core.Supplementary.Tensors.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic Steinmann Model
#####################################################################################################


class Steinmann(object):
	"""docstring for Steinmann"""
	def __init__(self, ndim):
		super(Steinmann, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim+1
		self.modelname = 'Steinmann'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		c1 = MaterialArgs.c1
		c2 = MaterialArgs.c2
		varepsilon_1 = MaterialArgs.eps_1

		I = StrainTensors.I
		J = StrainTensors.J
		b = StrainTensors.b

		# Update Lame constants
		mu2 = mu - lamb*(J-1.0)
		lamb2 = lamb*(2.0*J-1.0) - mu

		E = 1.0*ElectricFieldx
		Ex = E.reshape(E.shape[0])
		EE = np.dot(E,E.T)
		be = np.dot(b,ElectricFieldx).reshape(ndim)

		# Fourth order elasticity tensor
		# C_Voigt = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) +\
		# 	varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) -AikBjl(EE,I)-AilBjk(EE,I)-AilBjk(I,EE)-AikBjl(I,EE) ) +\
		# 	varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(AikBjl(I,I) + AilBjk(I,I))-0.5*AijBkl(I,I))
		# C_Voigt=0.5*(C_Voigt+C_Voigt.T)

		C_Voigt = Voigt(			
			lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) +\
			varepsilon_1*(d('ij,kl',I,EE) + d('ij,kl',EE,I) - d('ik,jl',EE,I)-d('il,jk',EE,I)-d('il,jk',I,EE)-d('ik,jl',I,EE) ) +\
			varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(d('ik,jl',I,I) + d('il,jk',I,I))-0.5*d('ij,kl',I,I))
			,1
			)

		
		# Coupled Tensor (e - 3rd order)
		# Note that the actual piezoelectric tensor is symmetric wrt to the last two indices
		# Actual tensor (varepsilon_1 bit) is: e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) 
		# We need to make its Voigt_form symmetric with respect to (j,k) instead of (i,j) 
		# e_voigt = 1.0*varepsilon_1*(AijUk(I,Ex)+AikUj(I,Ex)-UiAjk(Ex,I)).T +\
		# (2.0*c2/J)*(AikUj(b,be)+AijUk(b,be)).T

		e_voigt = Voigt(
			1.0*varepsilon_1*(d('ij,k',I,Ex)+d('ik,j',I,Ex)-d('i,jk',Ex,I)) +\
			(2.0*c2/J)*(d('ik,j',b,be)+d('ij,k',b,be))
			,1
			)
			

		# Dielectric Tensor (Permittivity - 2nd order)
		Permittivity = -varepsilon_1*I +\
		(2.0*c1/J)*b +\
		(2.0*c2/J)*np.dot(b,b)

		# Build the Hessian
		factor = -1.
		H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
		H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
		H_Voigt = np.concatenate((H1,H2),axis=0)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx):

		c2 = MaterialArgs.c2

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I
		E = ElectricFieldx

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		varepsilon_1 = MaterialArgs.eps_1

		be = np.dot(b,ElectricFieldx)

		return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I + varepsilon_1*(np.dot(E,E.T)-0.5*np.dot(E.T,E)[0,0]*I) +\
		(2.0*c2/J)*np.dot(be,be.T)


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		
		c1 = MaterialArgs.c1
		c2 = MaterialArgs.c2
		varepsilon_1 = MaterialArgs.eps_1		

		J = StrainTensors.J
		b = StrainTensors.b 
		bb =  np.dot(b,b)
		
		return varepsilon_1*ElectricFieldx -\
		(2.0*c1/J)*np.dot(b,ElectricFieldx) -\
		(2.0*c2/StrainTensors.J)*np.dot(bb,ElectricFieldx).reshape(StrainTensors.b.shape[0],1)
