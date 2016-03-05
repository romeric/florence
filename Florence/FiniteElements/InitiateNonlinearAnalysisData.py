import numpy as np
from warnings import warn

def InitiateNonlinearAnalysisData(MainData,mesh,material):
    
    # INFORMATION REQUIRED FOR NONLINEAR ANALYSIS
    ################################################################################
    Tolerance = 2.0e-07
    if MainData.Analysis == 'Static':
    	Increments = getattr(MainData,"LoadIncrement",None)
    	if Increments is None:
    		LoadIncrement = 1
    	else:
    		LoadIncrement = Increments
    else:
        LoadIncrement = MainData.BoundaryData.nstep


    if material.mtype == "LinearModel" and LoadIncrement > 1:
        warn("LinearModel cannot be solved in multiple increments. Load increment is set back to 1")
        LoadIncrement = 1
    
    class AssemblyParameters(object):
        """Information about load increments, iterations and such"""
        ExternalLoadNature = 'Linear'
        LoadIncrements = LoadIncrement
        # LoadIncrements = MainData.LoadIncrement
        LoadIncrementNumber = 0
        IterationNumber = 0
        NRTolerance = Tolerance
        GeometryUpdate = 0
        MaxIter = 50
        FailedToConverge = False



    MainData.AssemblyParameters = AssemblyParameters
    ###########################################################################

    return np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float32), \
    	np.zeros((mesh.points.shape[0]*MainData.nvar,1),dtype=np.float32)