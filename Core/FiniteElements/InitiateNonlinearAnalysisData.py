import numpy as np
from warnings import warn

def InitiateNonlinearAnalysisData(MainData,mesh):
    
    # INFORMATION REQUIRED FOR NONLINEAR ANALYSIS
    ################################################################################
    Tolerance = 2.0e-04
    if MainData.Analysis == 'Static':
        LoadIncrement = 2
        pass
    else:
        LoadIncrement = MainData.BoundaryData.nstep


    if MainData.MaterialArgs.Type == "LinearModel" and LoadIncrement > 1:
        warn("LinearModel cannot be solved in multiple increments. I am changing LoadIncrement to 1")
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

    return np.zeros((mesh.points.shape[0]*MainData.nvar,1)), np.zeros((mesh.points.shape[0]*MainData.nvar,1))