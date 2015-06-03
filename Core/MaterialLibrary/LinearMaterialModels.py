import numpy as np

# MATERIAL MODELS:
# LinearElastic
# LinearElectroElastic
# LinearElectroMagnetoElastic
# LinearThermoElectroMagnetoElastic

# nvar is the sum of dimensions of vectorial field(s) we are solving for.
# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz


class LinearElastic(object):
	"""docstring for LinearElastic"""
	def Hessian(self,MaterialArgs,ndim):
		MaterialArgs.H = MaterialArgs.Elastic
		nvar = ndim
		modelname = 'LinearElastic' 
		return nvar, modelname

	def ConstitutiveStiffnessIntegrand(self,Domain,B,SpatialGradient,H,nvar):

		# Three Dimensions
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,2] = SpatialGradient[2,:]
			# Mechanical - Shear Terms
			B[1:B.shape[0]:nvar,3] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,3] = SpatialGradient[1,:]

			B[0:B.shape[0]:nvar,4] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[0,:]

			B[0:B.shape[0]:nvar,5] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,5] = SpatialGradient[0,:]

		# Two Dimensions
		elif SpatialGradient.shape[0]==2:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			# Mechanical - Shear Terms
			B[0:B.shape[0]:nvar,2] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,2] = SpatialGradient[0,:]

		return np.dot(np.dot(B,H),B.T)




class LinearElectroElastic(object):
	"""docstring for LinearElectroElastic"""
	def Hessian(self,MaterialArgs,ndim):
		H1 = np.concatenate((MaterialArgs.Elastic,MaterialArgs.Piezoelectric),axis=1)
		H2 = np.concatenate((MaterialArgs.Piezoelectric.T,-MaterialArgs.Permitivitty),axis=1)
		MaterialArgs.H = np.concatenate((H1,H2),axis=0)
		nvar = ndim+1 
		modelname = 'LinearElectroElastic'
		return nvar, modelname


	def ConstitutiveStiffnessIntegrand(self,Domain,B,SpatialGradient,H,nvar):

		# Three Dimensions
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,2] = SpatialGradient[2,:]
			# Mechanical - Shear Terms
			B[1:B.shape[0]:nvar,3] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,3] = SpatialGradient[1,:]

			B[0:B.shape[0]:nvar,4] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[0,:]

			B[0:B.shape[0]:nvar,5] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,5] = SpatialGradient[0,:]

			# Electrostatic 
			B[3:B.shape[0]:nvar,6] = SpatialGradient[0,:]
			B[3:B.shape[0]:nvar,7] = SpatialGradient[1,:]
			B[3:B.shape[0]:nvar,8] = SpatialGradient[2,:]

		# Two Dimensions
		elif SpatialGradient.shape[0]==2:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			# Mechanical - Shear Terms
			B[0:B.shape[0]:nvar,2] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,2] = SpatialGradient[0,:]

			# Electrostatic 
			B[2:B.shape[0]:nvar,3] = SpatialGradient[0,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[1,:]

		return np.dot(np.dot(B,H),B.T)




class LinearElectroMagnetoElastic(object):
	"""docstring for LinearElectroElastic"""
	def Hessian(self,MaterialArgs,ndim):
		H1 = np.concatenate((np.concatenate((MaterialArgs.Elastic,MaterialArgs.Piezoelectric),axis=1),MaterialArgs.Piezomagnetic),axis=1)
		H2 = np.concatenate((np.concatenate((MaterialArgs.Piezoelectric.T,-MaterialArgs.Permitivitty),axis=1),MaterialArgs.Electromagnetic),axis=1)
		H3 = np.concatenate((np.concatenate((MaterialArgs.Piezomagnetic.T,MaterialArgs.Electromagnetic.T),axis=1),-MaterialArgs.Permeability),axis=1)
		MaterialArgs.H = np.concatenate((np.concatenate((H1,H2),axis=0),H3),axis=0)
		nvar = ndim+2 
		modelname = 'LinearElectroMagnetoElastic'
		return nvar, modelname


	def ConstitutiveStiffnessIntegrand(self,Domain,B,SpatialGradient,H,nvar):

		# Three Dimensions
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,2] = SpatialGradient[2,:]
			# Mechanical - Shear Terms
			B[1:B.shape[0]:nvar,3] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,3] = SpatialGradient[1,:]

			B[0:B.shape[0]:nvar,4] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[0,:]

			B[0:B.shape[0]:nvar,5] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,5] = SpatialGradient[0,:]

			# Electrostatic 
			B[3:B.shape[0]:nvar,6] = SpatialGradient[0,:]
			B[3:B.shape[0]:nvar,7] = SpatialGradient[1,:]
			B[3:B.shape[0]:nvar,8] = SpatialGradient[2,:]

			# Magnetostatic 
			B[4:B.shape[0]:nvar,9] = SpatialGradient[0,:]
			B[4:B.shape[0]:nvar,10] = SpatialGradient[1,:]
			B[4:B.shape[0]:nvar,11] = SpatialGradient[2,:]

		# Two Dimensions
		elif SpatialGradient.shape[0]==2:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			# Mechanical - Shear Terms
			B[0:B.shape[0]:nvar,2] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,2] = SpatialGradient[0,:]

			# Electrostatic 
			B[2:B.shape[0]:nvar,3] = SpatialGradient[0,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[1,:]

			# Magnetostatic 
			B[3:B.shape[0]:nvar,5] = SpatialGradient[0,:]
			B[3:B.shape[0]:nvar,6] = SpatialGradient[1,:]


		return np.dot(np.dot(B,H),B.T)




class LinearThermoElectroMagnetoElastic(object):
	"""docstring for LinearElectroElastic"""
	def Hessian(self,MaterialArgs,ndim):
		H1 = np.concatenate((np.concatenate((MaterialArgs.Elastic,MaterialArgs.Piezoelectric),axis=1),MaterialArgs.Piezomagnetic),axis=1)
		H1 = np.concatenate((H1,MaterialArgs.ThermoMechanical),axis=1)
		H2 = np.concatenate((np.concatenate((np.concatenate((MaterialArgs.Piezoelectric.T,-MaterialArgs.Permitivitty),axis=1),MaterialArgs.Electromagnetic),axis=1),MaterialArgs.ThermoElectrical),axis=1)
		H3 = np.concatenate((np.concatenate((np.concatenate((MaterialArgs.Piezomagnetic.T,MaterialArgs.Electromagnetic.T),axis=1),-MaterialArgs.Permeability),axis=1),MaterialArgs.ThermoMagnetical),axis=1)
		H4 = np.concatenate((np.concatenate((MaterialArgs.ThermoMechanical.T,MaterialArgs.ThermoElectrical.T),axis=1),MaterialArgs.ThermoMagnetical.T),axis=1)
		H4 = np.concatenate((H4,-MaterialArgs.SpecificHeat),axis=1)
		MaterialArgs.H = np.concatenate((np.concatenate((np.concatenate((H1,H2),axis=0),H3),axis=0),H4),axis=0)
		nvar = ndim+3 
		modelname = 'LinearThermoElectroMagnetoElastic'
		return nvar, modelname


	def ConstitutiveStiffnessIntegrand(self,Domain,B,SpatialGradient,H,nvar):

		# Three Dimensions
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,2] = SpatialGradient[2,:]
			# Mechanical - Shear Terms
			B[1:B.shape[0]:nvar,3] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,3] = SpatialGradient[1,:]

			B[0:B.shape[0]:nvar,4] = SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[0,:]

			B[0:B.shape[0]:nvar,5] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,5] = SpatialGradient[0,:]

			# Electrostatic 
			B[3:B.shape[0]:nvar,6] = SpatialGradient[0,:]
			B[3:B.shape[0]:nvar,7] = SpatialGradient[1,:]
			B[3:B.shape[0]:nvar,8] = SpatialGradient[2,:]

			# Magnetostatic 
			B[4:B.shape[0]:nvar,9] = SpatialGradient[0,:]
			B[4:B.shape[0]:nvar,10] = SpatialGradient[1,:]
			B[4:B.shape[0]:nvar,11] = SpatialGradient[2,:]
			# Thermal
			B[4:B.shape[0]:nvar,7] = Domain.Bases[:,counter]

		# Two Dimensions
		elif SpatialGradient.shape[0]==2:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			# Mechanical - Shear Terms
			B[0:B.shape[0]:nvar,2] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,2] = SpatialGradient[0,:]

			# Electrostatic 
			B[2:B.shape[0]:nvar,3] = SpatialGradient[0,:]
			B[2:B.shape[0]:nvar,4] = SpatialGradient[1,:]

			# Magnetostatic 
			B[3:B.shape[0]:nvar,5] = SpatialGradient[0,:]
			B[3:B.shape[0]:nvar,6] = SpatialGradient[1,:]
			# Thermal
			B[4:B.shape[0]:nvar,7] = Domain.Bases[:,counter]

		return np.dot(np.dot(B,H),B.T)
		











#####################################################################################################
										# NONLINEAR MODELS
#####################################################################################################


class NeoHookean(object):
	"""docstring for NeoHookean"""
	def __init__(self, ndim):
		super(NeoHookean, self).__init__()
		self.ndim = ndim
	def Get(self):
		self.nvar = self.ndim
		self.modelname = 'NeoHookean'
		return self.nvar, self.modelname

	def Hessian(self,MaterialArgs,ndim,StrainTensors):
		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb
		detF = StrainTensors.J

		# mu1 = 1.0*(mu-lamb*np.log(detF))/detF 
		# lamb1 = 1.0*lamb/detF 

		mu2 = mu - lamb*(detF-1.0)
		lamb2 = lamb*(2*detF-1.0) - mu

		delta = np.eye(ndim,ndim)
		# Hessian is the fourth order elasticity tensor in this case
		C = np.zeros((ndim,ndim,ndim,ndim))
		for i in range(0,ndim):
			for j in range(0,ndim):
				for k in range(0,ndim):
					for l in range(0,ndim):
						# C[i,j,k,l] += lamb1*delta[i,j]*delta[k,l]+2.0*mu1*delta[i,k]*delta[j,l]
						# C[i,j,k,l] += lamb1*delta[i,j]*delta[k,l]+mu1*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k])

						C[i,j,k,l] += lamb2*delta[i,j]*delta[k,l]+mu2*(delta[i,k]*delta[j,l] + delta[i,l]*delta[j,k])

		# Symmetric part
		Csym = np.zeros((ndim,ndim,ndim,ndim))
		for i in range(0,ndim):
			for k in range(0,ndim):
				for j in range(0,ndim):
					for l in range(0,ndim):
						Csym[i,k,j,l] = 1.0/4.0*(C[i,k,j,l]+C[k,i,j,l]+C[i,k,l,j]+C[k,i,l,j])


		# Voigt Notation
		C_Voigt = 0.5*np.array([
			[2*C[0,0,0,0],2*C[0,0,1,1],2*C[0,0,2,2],C[0,0,0,1]+C[0,0,1,0],C[0,0,0,2]+C[0,0,2,0],C[0,0,1,2]+C[0,0,2,1]],
			[0			 ,2*C[1,1,1,1],2*C[1,1,2,2],C[1,1,0,1]+C[1,1,1,0],C[1,1,0,2]+C[1,1,2,0],C[1,1,1,2]+C[1,1,2,1]],
			[0			 ,0			  ,2*C[2,2,2,2],C[2,2,0,1]+C[2,2,1,0],C[2,2,0,2]+C[2,2,2,0],C[2,2,1,2]+C[2,2,2,1]],
			[0			 ,0			  ,0		   ,C[0,1,0,1]+C[0,1,1,0],C[0,1,0,2]+C[0,1,2,0],C[0,1,1,2]+C[0,1,2,1]],
			[0			 ,0			  ,0		   ,0					 ,C[0,2,0,2]+C[0,2,2,0],C[0,2,1,2]+C[0,2,2,1]],
			[0			 ,0			  ,0		   ,0					 ,0					   ,C[1,2,1,2]+C[1,2,2,1]]
			])
		C_Voigt = C_Voigt+C_Voigt.T 
		for i in range(0,C_Voigt.shape[0]):
			C_Voigt[i,i] = C_Voigt[i,i]/2.0


		MaterialArgs.H_VoigtSize = C_Voigt.shape[0]
		
		return C, C_Voigt



	def ConstitutiveStiffnessIntegrand(self,Domain,B,MaterialGradient,nvar,SpatialGradient,ndim,CauchyStressTensor,H,H_Voigt):

		# # Indicial Form
		# BDB = np.zeros((nvar*SpatialGradient.shape[0],nvar*SpatialGradient.shape[0]))
		# t = np.zeros((nvar*SpatialGradient.shape[0],1))

		# for a in range(0,Domain.Bases.shape[0]):
		# 	for b in range(0,Domain.Bases.shape[0]):
		# 		BDB_ab = np.zeros((nvar,nvar))		
		# 		for i in range(0,ndim):
		# 			for j in range(0,ndim):
		# 				for k in range(0,ndim):
		# 					for l in range(0,ndim):			
		# 						BDB_ab[i,j] += SpatialGradient[a,k]*H[i,k,j,l]*SpatialGradient[b,l]
		# 		BDB[nvar*a:nvar*a+nvar,nvar*b:nvar*b+nvar] = BDB_ab  
 	
		# for a in range(0,Domain.Bases.shape[0]):
		# 	for i in range(0,ndim):
		# 		t[ndim*a+i,0] += np.dot(CauchyStressTensor[i,:],SpatialGradient[a,:])


		# Matrix Form
		SpatialGradient = SpatialGradient.T
		factor=1

		# Three Dimensions
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,2] = SpatialGradient[2,:]
			# Mechanical - Shear Terms
			B[1:B.shape[0]:nvar,5] = factor*SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,5] = factor*SpatialGradient[1,:]

			B[0:B.shape[0]:nvar,4] = factor*SpatialGradient[2,:]
			B[2:B.shape[0]:nvar,4] = factor*SpatialGradient[0,:]

			B[0:B.shape[0]:nvar,3] = factor*SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,3] = factor*SpatialGradient[0,:]

			CauchyStressTensor_Voigt = np.array([
				CauchyStressTensor[0,0],CauchyStressTensor[1,1],CauchyStressTensor[2,2],
				CauchyStressTensor[0,1],CauchyStressTensor[0,2],CauchyStressTensor[1,2]
				]).reshape(6,1)
		
		BDB = np.dot(np.dot(B,H_Voigt),B.T)
		t = np.dot(B,CauchyStressTensor_Voigt)

				
		return BDB, t


	def GeometricStiffnessIntegrand(self,SpatialGradient,CauchyStressTensor,nvar,ndim):

		# # Indicial Form
		# BDB = np.zeros((nvar*SpatialGradient.shape[0],nvar*SpatialGradient.shape[0]))
		# I = np.eye(ndim,ndim)
		# for a in range(0,SpatialGradient.shape[0]):
		# 	for b in range(0,SpatialGradient.shape[0]):
		# 		BDB_ab = np.zeros((nvar,nvar))
		# 		for i in range(0,ndim):
		# 			for j in range(0,ndim):
		# 				for k in range(0,ndim):
		# 					for l in range(0,ndim):
		# 						BDB_ab[i,j] += SpatialGradient[a,k]*CauchyStressTensor[k,l]*SpatialGradient[b,l]*I[i,j]
		# 		BDB[nvar*a:nvar*a+nvar,nvar*b:nvar*b+nvar] = BDB_ab  


		# Matrix Form
		B = np.zeros((nvar*SpatialGradient.shape[0],nvar*nvar))
		SpatialGradient = SpatialGradient.T
		if SpatialGradient.shape[0]==3:

			B[0:B.shape[0]:nvar,0] = SpatialGradient[0,:]
			B[0:B.shape[0]:nvar,1] = SpatialGradient[1,:]
			B[0:B.shape[0]:nvar,2] = SpatialGradient[2,:]

			B[1:B.shape[0]:nvar,3] = SpatialGradient[0,:]
			B[1:B.shape[0]:nvar,4] = SpatialGradient[1,:]
			B[1:B.shape[0]:nvar,5] = SpatialGradient[2,:]

			B[2:B.shape[0]:nvar,6] = SpatialGradient[0,:]
			B[2:B.shape[0]:nvar,7] = SpatialGradient[1,:]
			B[2:B.shape[0]:nvar,8] = SpatialGradient[2,:]

		S = np.zeros((3*ndim,3*ndim))
		S[0:ndim,0:ndim] = CauchyStressTensor
		S[ndim:2*ndim,ndim:2*ndim] = CauchyStressTensor
		S[2*ndim:,2*ndim:] = CauchyStressTensor

		BDB = np.dot(np.dot(B,S),B.T)

				
		return BDB

	def CauchyStress(self,MaterialArgs,StrainTensors):

		b = StrainTensors.b 
		J = StrainTensors.J
		I = StrainTensors.I

		mu = MaterialArgs.mu
		lamb = MaterialArgs.lamb

		# return 1.0*((mu/J)*(b-I)+(lamb/J)*np.log(J)*I)

		return mu/J*b+(lamb*(J-1.0)-mu)*I


