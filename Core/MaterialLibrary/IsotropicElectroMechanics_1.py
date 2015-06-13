import numpy as np
from Core.Supplementary.Tensors.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz

#####################################################################################################
								# Isotropic Electromechanical Model 1
#####################################################################################################


class IsotropicElectroMechanics_1(object):
	"""docstring for NeoHookean"""
	def __init__(self, ndim):
		super(IsotropicElectroMechanics_1, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim+1
		self.modelname = 'IsotropicElectroMechanics_1'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		varepsilon_1 = MaterialArgs.eps_1

		detF = StrainTensors.J

		mu2 = mu - lamb*(detF-1.0)
		lamb2 = lamb*(2.0*detF-1.0) - mu

		delta = np.eye(ndim,ndim,dtype=np.float64)
		E = 1.0*ElectricFieldx
		
		Ex = E.reshape(E.shape[0])
		EE = np.dot(E,E.T)

		I = delta
		# C = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) + varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) -2.*AikBjl(EE,I)-2.0*AilBjk(I,EE) ) + varepsilon_1*(np.dot(E.T,E)[0,0])*(AikBjl(I,I)-0.5*AijBkl(I,I))
		C = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) +\
			varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) -AikBjl(EE,I)-AilBjk(EE,I)-AilBjk(I,EE)-AikBjl(I,EE) ) +\
			varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(AikBjl(I,I) + AilBjk(I,I))-0.5*AijBkl(I,I))
		C=0.5*(C+C.T)
		C_Voigt = C
		# print C_Voigt

		# Computing the hessian 
		# Elasticity tensor (C - 4th order tensor) 
		# C[i,j,k,l] += lamb2*delta[i,j]*delta[k,l]+2.0*mu2*(delta[i,k]*delta[j,l]) #

		b = StrainTensors.b
		be = np.dot(b,ElectricFieldx).reshape(ndim,1)
		# Coupled Tensor (e - 3rd order)

		# e[k,i,j] += (-2.0*varepsilon_1/detF)*(be[j]*b[i,k] + be[i]*b[j,k]) #
		# e[i,j,k] += 1.0*varepsilon_1*( E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) ##
		# e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) ##

		# Note that the actual piezoelectric tensor is symmetric wrt to the last two indices
		# Actual tensor is: e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) 
		# We need to make its Voigt_form symmetric with respect to (j,k) instead of (i,j) 
		e_voigt = 1.0*varepsilon_1*(AijUk(I,Ex)+AikUj(I,Ex)-UiAjk(Ex,I)).T
		# print e_voigt.shape


		# Dielectric Tensor (Permittivity - 2nd order)
		Permittivity = -varepsilon_1*delta ##

		# bb =  np.dot(StrainTensors.b,StrainTensors.b) #
		# Permittivity = -(2.0*varepsilon_1/detF)*bb #


		factor = -1.
		H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
		H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
		H_Voigt = np.concatenate((H1,H2),axis=0)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		# return H_Voigt, C, e, Permittivity
		return H_Voigt


	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I
		E = ElectricFieldx

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		varepsilon_1 = MaterialArgs.eps_1

		be = np.dot(b,ElectricFieldx)

		return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I + varepsilon_1*(np.dot(E,E.T)-0.5*np.dot(E.T,E)[0,0]*I) ## 
		# return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I - (2.0*varepsilon_1/J)*np.dot(be,be.T)


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
		
		J = StrainTensors.J
		b = StrainTensors.b 
		varepsilon_1 = MaterialArgs.eps_1

		bb =  np.dot(b,b)
		
		return varepsilon_1*ElectricFieldx ##
		# return (2.0*varepsilon_1/StrainTensors.J)*np.dot(bb,ElectricFieldx).reshape(StrainTensors.b.shape[0],1)
