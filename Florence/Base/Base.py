from __future__ import print_function
import os, sys, imp
from time import time
import multiprocessing as MP
import numpy as np
# try:
#     from mpi4py import MPI
#     hasMPI = True
#     comm = MPI.COMM_WORLD
# except ImportError:
#     hasMPI = False





# FLORENCE BASE CLASS 
class Base(object):
    """FLorence base class. General data such as directories, files, analysis session, etc 
        that needs to be loaded a priori are stored
        
        pwd:                        Florence's top level directory
        session:                    {'FEM','BEM','Coupled'} Session to run
        __NO_DEBUG__:               Enter debug mode of the package (if false). Activates all numerical checks                 
        __VECTORISATION__:          Activate numpy's (einsum) for computing elemental matrices with no loops
        __PARALLEL__:               Activate multiprocessing for either shared or distributed memory or both
        numCPU:                     Number of concurrent cores/hyperthreads for parallelisation
        __MEMORY__:                 {'SHARED','DISTRIBUTED','AUTO','HYBRID'} Option for shared/distributed 
                                    memory parallelisation
        
        C:                          [int] order of basis functions. Note that C=P-1 where P is polynomial degree
        norder:                     [int] number of quadrature points 
        plot:                       [tuple of ints] plot flag for BEM
        nrplot:                     [tuple] plot flag for Newton-Raphson convergence
        write:                      [boolean] flag for writting simulation results in .vtu/.mat/.eps/.dat formats
        
        """

    FloatType=np.float32
    IntType=np.int32

    if sys.version_info.major == 2:
        Range = xrange
    else:
        Range = range 


    pwd = os.path.dirname(os.path.realpath('__file__'))
    session = 'FEM'
    # session = 'BEM'
    # session = 'Coupled'


    __NO_DEBUG__ = False
    __VECTORISATION__ = True
    __PARALLEL__ = False
    nCPU = 1
    __MEMORY__ = 'SHARED'


    C = 0
    norder = 2 
    plot = (0,3)
    nrplot = (0,'last')
    write = 0



    # # PROBLEM SPATIAL DIMENSION- 1D, 2D, 3D
    # ndim = 2
    # nvar = ndim
    # Fields = 'Mechanics'
    # # Fields = 'ElectroMechanics'
    
    # Formulation = 'DisplacementApproach'
    # # Formulation = 'DisplacementElectricPotentialApproach'

    # Analysis = 'Static'
    # # Analysis = 'Dynamic'
    # AnalysisType = 'Linear'
    # # AnalysisType = 'Nonlinear'


    Timer = 0


    # DECIDE WHICH PARALLEL MODEL TO ACTIVATE
    def ParallelModel(self):
        if self.__MEMORY__ == "shared":
            pass
        elif self.__MEMORY__ == "distributed":
            print(comm.rank)

    isScaledJacobianComputed = False
