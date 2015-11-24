import numpy as np

def InitiateNonlinearAnalysisData(MainData,mesh):
	
	# INFORMATION REQUIRED FOR NONLINEAR ANALYSIS
	################################################################################
	Tolerance = 1.0e-05
	if MainData.Analysis == 'Static':
		LoadIncrement = 1
	else:
		LoadIncrement = MainData.BoundaryData.nstep
	
	class AssemblyParameters(object):
		"""Information about load increments, iterations and such"""
		ExternalLoadNature = 'Linear'
		LoadIncrements = LoadIncrement
		LoadIncrementNumber = 0
		IterationNumber = 0
		NRTolerance = Tolerance
		GeometryUpdate = 0
		MaxIter = 50
		FailedToConverge = False


	MainData.AssemblyParameters = AssemblyParameters
	###########################################################################

	return np.zeros((mesh.points.shape[0]*MainData.nvar,1)), np.zeros((mesh.points.shape[0]*MainData.nvar,1))