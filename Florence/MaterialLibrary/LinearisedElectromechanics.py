import numpy as np
import numpy.linalg as la
from Florence.Tensor import trace
from Florence.LegendreTransform import *

#####################################################################################################
								# Isotropic Steinmann Model
#####################################################################################################





class LinearisedElectromechanics(object):
	"""docstring for Steinmann"""
	def __init__(self, ndim):
		super(LinearisedElectromechanics, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim+1
		self.modelname = 'LinearisedElectromechanics'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		# Using Einstein summation (using numpy einsum call)
		d = np.einsum

		# Get material constants (5 in this case)
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		c1 = MaterialArgs.c1
		c2 = MaterialArgs.c2
		varepsilon_1 = MaterialArgs.eps_1

		I = StrainTensors['I']

		# Coupled linearised problem
		strain = StrainTensors['strain'][gcounter]
		tre = np.trace(strain)
		d1 = 2.0*strain - tre*I
		d2 = I+(1.0 - tre)*d1
		
		# D is the new D - Don't get confuesed! Let us compute it
		E = 1.0*ElectricFieldx; 							Ex=E.reshape(ndim)
		D = varepsilon_1*np.dot(np.linalg.inv(d2),E); 		Dx=D.reshape(ndim)

		PermittivityW = (1.0/varepsilon_1)*d2
		# Piezo tensor
		d3 = np.dot(d1,Dx) 
		# e_voigtW = (1.0/varepsilon_1)*(-d('i,jk',d3,I)+(1-tre)*(d('ik,j',I,Dx)+d('ij,k',I,Dx)-d('jk,i',I,Dx)))
		# e_voigtW = Voigt(e_voigtW)
		e_voigtW = Voigt( (1.0/varepsilon_1)*(-d('i,jk',d3,I)+(1-tre)*(d('ik,j',I,Dx)+d('ij,k',I,Dx)-d('jk,i',I,Dx))) )
		
		# Elasticity tensor
		Sm = (1.0/varepsilon_1)*(np.dot(D,D.T)-0.5*np.dot(D.T,D)[0,0]*I)
		C_VoigtW = Voigt(lamb*d('ij,kl',I,I)+mu*(d('ik,jl',I,I)+d('il,jk',I,I)) -\
			d('ij,kl',Sm,I) - d('ij,kl',I,Sm)
			,1)

		

		Permittivity,e_voigt,C_Voigt = FreeEnergy2Enthalpy(PermittivityW,e_voigtW,C_VoigtW,0)



		# Build the Hessian
		factor = -1.
		H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
		H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
		H_Voigt = np.concatenate((H1,H2),axis=0)

		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):


		I = StrainTensors['I']
		strain = StrainTensors['strain'][gcounter]

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		varepsilon_1 = MaterialArgs.eps_1

		strain = StrainTensors.strain
		tre = np.trace(strain)
		d1 = 2.0*strain - tre*I
		d2 = I+(1.0 - tre)*d1
		
		# D is the new D - Don't get confuesed! Let us compute it
		E = 1.0*ElectricFieldx; 							Ex=E.reshape(E.shape[0])
		D = varepsilon_1*np.dot(np.linalg.inv(d2),E); 		Dx=D.reshape(D.shape[0])
		eD = np.dot(strain,D)


		sigma = lamb*tre*I + 2.0*mu*strain +\
		((1.0 - tre)/varepsilon_1)*(np.dot(D,D.T)-0.5*np.dot(D.T,D)[0,0]*I) -\
		(1.0/varepsilon_1)*(np.dot(D.T,eD)[0,0]-0.5*np.dot(D.T,D)[0,0]*tre)*I ##


		return sigma


	def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
		
		I = StrainTensors['I']
		varepsilon_1 = MaterialArgs.eps_1
		strain = StrainTensors['strain'][gcounter]
		tre = np.trace(strain)
		d1 = 2.0*strain - tre*I
		d2 = I+(1.0 - tre)*d1		
		
		E = 1.0*ElectricFieldx; 							Ex=E.reshape(E.shape[0])

		# Direct approach for finding D ---- D is the new D - Don't get confused!
		# D = varepsilon_1*np.dot(np.linalg.inv(d2),E); 		Dx=D.reshape(D.shape[0])

		# General Newton-Raphson approach for finding D ---- the permittivity (hessian/derivative) 
		# in the Newton-Raphson is not getting updated in this case (equivalent to having a constant stiffness matrix) 
		PermittivityW = (1.0/varepsilon_1)*d2
		D = LG_NewtonRaphson(PermittivityW,ElectricFieldx)

		return D
