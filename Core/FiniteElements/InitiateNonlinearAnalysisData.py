import numpy as np

def InitiateNonlinearAnalysisData(MainData,nmesh):
	
	# INFORMATION REQUIRED FOR NONLINEAR ANALYSIS
	################################################################################
	Tolerance = 1.0e-05
	if MainData.Analysis == 'Static':
		LoadIncrement = 2
	else:
		LoadIncrement = MainData.BoundaryData.nstep
	# LoadFactor = 1./LoadIncrement
	
	class AssemblyParameters(object):
		"""docstring for AssemblyParameters"""#
		ExternalLoadNature = 'Linear'
		LoadIncrements = LoadIncrement
		LoadIncrementNumber = 0
		IterationNumber = 0
		NRTolerance = Tolerance
		GeometryUpdate = 0
		MaxIter = 50

	MainData.AssemblyParameters = AssemblyParameters
	###########################################################################

	return np.zeros((nmesh.points.shape[0]*MainData.nvar,1)), np.zeros((nmesh.points.shape[0]*MainData.nvar,1))