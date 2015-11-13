import numpy as np 

#------------------------------------------------------------------Elemental Matrices-------------------------------------------------------------------------#

def ConstitutiveStiffnessIntegrand(self,B,nvar,ndim,AnalysisType,Prestress,SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt):

	# MATRIX FORM
	SpatialGradient = SpatialGradient.T

	# THREE DIMENSIONS
	if SpatialGradient.shape[0]==3:

		B[0::nvar,0] = SpatialGradient[0,:]
		B[1::nvar,1] = SpatialGradient[1,:]
		B[2::nvar,2] = SpatialGradient[2,:]
		# Mechanical - Shear Terms
		B[1::nvar,5] = SpatialGradient[2,:]
		B[2::nvar,5] = SpatialGradient[1,:]

		B[0::nvar,4] = SpatialGradient[2,:]
		B[2::nvar,4] = SpatialGradient[0,:]

		B[0::nvar,3] = SpatialGradient[1,:]
		B[1::nvar,3] = SpatialGradient[0,:]


		if AnalysisType == 'Nonlinear' or Prestress:
			CauchyStressTensor_Voigt = np.array([
				CauchyStressTensor[0,0],CauchyStressTensor[1,1],CauchyStressTensor[2,2],
				CauchyStressTensor[0,1],CauchyStressTensor[0,2],CauchyStressTensor[1,2]
				]).reshape(6,1)

			TotalTraction = CauchyStressTensor_Voigt

	elif SpatialGradient.shape[0]==2:

		B[0::nvar,0] = SpatialGradient[0,:]
		B[1::nvar,1] = SpatialGradient[1,:]
		# Mechanical - Shear Terms
		B[0::nvar,2] = SpatialGradient[1,:]
		B[1::nvar,2] = SpatialGradient[0,:]


		if AnalysisType == 'Nonlinear' or Prestress:
			CauchyStressTensor_Voigt = np.array([
				CauchyStressTensor[0,0],CauchyStressTensor[1,1],
				CauchyStressTensor[0,1]]).reshape(3,1)

			TotalTraction = CauchyStressTensor_Voigt

	BDB = np.dot(np.dot(B,H_Voigt),B.T)
	# BDB = np.dot(np.dot(B,H_Voigt),B.T.copy())
	t=[]
	if AnalysisType == 'Nonlinear' or Prestress:
		t = np.dot(B,TotalTraction)

			
	return BDB, t


def GeometricStiffnessIntegrand(self,SpatialGradient,CauchyStressTensor,nvar,ndim):


	# Matrix Form
	B = np.zeros((nvar*SpatialGradient.shape[0],ndim*ndim))
	# B = np.zeros((nvar*SpatialGradient.shape[0],3*ndim))
	SpatialGradient = SpatialGradient.T
	S = 0
	if SpatialGradient.shape[0]==3:

		B[0::nvar,0] = SpatialGradient[0,:]
		B[0::nvar,1] = SpatialGradient[1,:]
		B[0::nvar,2] = SpatialGradient[2,:]

		B[1::nvar,3] = SpatialGradient[0,:]
		B[1::nvar,4] = SpatialGradient[1,:]
		B[1::nvar,5] = SpatialGradient[2,:]

		B[2::nvar,6] = SpatialGradient[0,:]
		B[2::nvar,7] = SpatialGradient[1,:]
		B[2::nvar,8] = SpatialGradient[2,:]

		S = np.zeros((3*ndim,3*ndim))
		S[0:ndim,0:ndim] = CauchyStressTensor
		S[ndim:2*ndim,ndim:2*ndim] = CauchyStressTensor
		S[2*ndim:,2*ndim:] = CauchyStressTensor

	elif SpatialGradient.shape[0]==2:

		B[0::nvar,0] = SpatialGradient[0,:]
		B[0::nvar,1] = SpatialGradient[1,:]

		B[1::nvar,2] = SpatialGradient[0,:]
		B[1::nvar,3] = SpatialGradient[1,:]

		S = np.zeros((ndim*ndim,ndim*ndim))
		S[0:ndim,0:ndim] = CauchyStressTensor
		S[ndim:2*ndim,ndim:2*ndim] = CauchyStressTensor


	BDB = np.dot(np.dot(B,S),B.T)
			
	return BDB





def MassIntegrand(self,Bases,N,Minimal,MaterialArgs):

	nvar = Minimal.nvar
	ndim = Minimal.ndim

	# We will work in total Lagrangian for mass matrix (no update needed)
	rho = MaterialArgs.rho

	# Three Dimensions
	if ndim==3:
		for ivar in range(0,ndim):
			N[ivar::nvar,ivar] = Bases

	
	rhoNN = rho*np.dot(N,N.T)

			
	return rhoNN