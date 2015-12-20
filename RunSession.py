#!/usr/bin/env python
from __future__ import print_function, absolute_import
""" THE RunSession ROUTINE RUNS A SPECIFIC SESSION, FOR INSTANCE, A FEM, A BEM OR A COUPLED SESSION.
    THE ENTIRE CODE IS EXECUTED FROM HERE."""


import imp
import os
import sys
import time
from sys import exit
from datetime import datetime
import cProfile
import pdb
import numpy as np
import scipy as sp
import numpy.linalg as la
from numpy.linalg import norm
from datetime import datetime
import multiprocessing as MP
# from mpi4py import MPI
# from numba.decorators import jit
# import cython
# from pympler import tracker, asizeof, summary, muppy
# from memory_profiler import profile

# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode
# SET NUMPY'S LINEWIDTH PRINT OPTION
np.set_printoptions(linewidth=300)

# IMPORT NECESSARY CLASSES FROM BASE
from Base import Base as MainData
# RETAIN CORE AFFINITY
# os.system("taskset -p 0xff %d" % os.getpid())


if __name__ == "__main__":
    
    # START THE ANALYSIS
    print("Initiating the routines... Current time is", datetime.now().time())
    
    MainData.__NO_DEBUG__ = True
    MainData.__VECTORISATION__ = True
    MainData.__PARALLEL__ = True
    MainData.numCPU = MP.cpu_count()
    # MainData.__PARALLEL__ = False
    # nCPU = 8
    __MEMORY__ = 'SHARED'
    # __MEMORY__ = 'DISTRIBUTED'
    
    MainData.C = 3
    MainData.norder = 2
    MainData.plot = (0, 3)
    nrplot = (0, 'last')
    MainData.write = 0

    # RUN THE APPROPRIATE SESSION
    from Main.FiniteElements.MainFEM import main
    # import Main.BoundaryElements.Main_BEM3D as BEM3D
    #-------------------------------------------------------------------------
    # exit(0)
    # FEM SESSION
    if MainData.session == 'FEM':
        t_FEM = time.time()
        # cProfile.run('main(MainData)')
        # pdb.run('main(MainData)')
        main(MainData)
        print("Time taken for the entire analysis was",
              time.time() - t_FEM, "seconds \n")
        # MEMORY USAGE INFORMATION
        # print 'Global sparse matrix needed', MainData.spmat, \
        #   'MB of memory with IJV indices requiring', MainData.ijv, 'MB'
        # print sys.getsizeof(MainData)

    # BEM SESSION
    elif MainData.session == 'BEM':
        for MainData.C in range(0, 1):
            t_BEM = time.time()
            rel_error = Main.BoundaryElements.main(MainData,
                                                   MainData.C, 5, 5, MainData.norder, 0, 0, 1)[0]
            print("Time taken for the entire analysis was",
                  time.time() - t_BEM, "seconds \n")
            # print (rel_error, MainData.C)
    elif MainData.session == 'BEM3D':
        BEM3D.main(MainData, 0, 0, 0)
    #-------------------------------------------------------------------------

    # delete all .pyc files
    # find . -name '*.pyc' -delete

    # import inspect; print(inspect.getsource(numpy.linspace)) # numpy/scipy documenation in a shell
    # import inspect; print(inspect.getsource(np.unique)) # numpy/scipy
    # documenation in a shellnp un

    # print np.__version__
    # import numpy.distutils.system_info as sysinfo
    # sysinfo.get_info('umfpack')
    # sysinfo.get_info('openblas')
    # sp.__config__.show()
    # sysinfo.show_all()

    # sudo update-alternatives --config libblas.so.3
    # ldd /usr/lib/python2.7/dist-packages/numpy/core/_dotblas.so
