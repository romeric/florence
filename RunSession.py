#!/usr/bin/env python
""" THE RunSession ROUTINE RUNS A SPECIFIC SESSION, FOR INSTANCE, A FEM, A BEM OR A COUPLED SESSION.
	THE ENTIRE CODE IS EXECUTED FROM HERE."""


import imp, os, sys, time
import cProfile, pdb
# from pympler import tracker, asizeof, summary, muppy
# from memory_profiler import profile
from datetime import datetime
import numpy as np
import scipy as sp 
import multiprocessing as MP
# from mpi4py import MPI
# from numba.decorators import jit
# import cython


# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode
# import pypar

# START THE ANALYSIS
print "Initiating the routines... Current time is", datetime.now().time()

# FLORENCE BASE CLASS 
class MainData(object):
 	"""FLorence base class. General data such as directories, files, analysis session, etc 
        that needs to be loaded a priori are stored
        
        pwd:                            Florence's top level directory
        session:                        {'FEM','BEM','Coupled'} Session to be run
        __NO_DEBUG__:                   Enter debug mode of the package (if false). Activates all numerical checks                 
        __VECTORISATION__:              Activate numpy's SIMD instructions e.g. (einsum) for computing elemental matrices with no loops
        __PARALLEL__:                   Activate multiprocessing for either or both shared and distributed memory
        numCPU:                         Number of concurrent cores/hyperthreads for parallelisation
        __PARALLEL_MEMORY__:            {'SHARED','DISTRIBUTED','AUTO','BOTH'} Option for shared or distributed memory parallelisation
        
        C:                              [int] order of basis functins. Note that C=P-1 where P is polynomial degree
        norder:                         [int] number of quadrature points 
        plot:                           [tuple of ints] plot flag for BEM
        nrplot:                         [tuple] plot flag for Newton-Raphson convergence
        write:                          [boolean] flag for writting simulation results in .vtu/.mat/.eps/.dat formats
        
        """

 	pwd = os.path.dirname(os.path.realpath('__file__'))
 	session = 'FEM'
 	# session = 'BEM'
 	# session = 'Coupled'


 	__NO_DEBUG__ = True						 
 	__VECTORISATION__ = True				 
 	__PARALLEL__ = True 					
 	numCPU = MP.cpu_count()				
 	# __PARALLEL__ = False
 	# nCPU = 8
 	__MEMORY__ = 'SHARED'			
 	# __MEMORY__ = 'DISTRIBUTED'					

 	C = 1									
 	norder = 2  							
 	plot = (0,3)						
 	nrplot = (0,'last')				
 	write = 0	


# RUN THE APPROPRIATE SESSION
from Main.FiniteElements.MainFEM import main
# import Main.BoundaryElements.Main_BEM3D as BEM3D
# from Core.FiniteElements.ConvergencePlot import ConvergencePlot
#----------------------------------------------------------------------------------------------------------------------

# FEM SESSION
if MainData.session == 'FEM':
	t_FEM = time.time()
	# tr = tracker.SummaryTracker()
	# cProfile.run('main(MainData)')
	# comm = MPI.COMM_WORLD
	# if comm.rank==0:
		# main(MainData)	
	# sp.__config__.show()
	# sp.test()
	# pdb.run('main(MainData)')
	main(MainData)	
	# tr.print_diff()
	# print asizeof.asizeof(MainData)
	print 'Time taken for the entire analysis was', time.time()-t_FEM, 'seconds \n'
	# MEMORY USAGE INFORMATION
	# print 'Global sparse matrix needed', MainData.spmat, 'MB of memory with IJV indices requiring', MainData.ijv, 'MB'
	# print sys.getsizeof(MainData)


# BEM SESSION
elif MainData.session == 'BEM':
	for MainData.C in range(0,1):
		t_BEM = time.time()
		rel_error = Main.BoundaryElements.main(MainData,MainData.C,5,5,MainData.norder,0,0,1)[0]
		print 'Time taken for the entire analysis was', time.time()-t_BEM, 'seconds'
		print rel_error, MainData.C
elif MainData.session == 'BEM3D':
	BEM3D.main(MainData,0,0,0)
#----------------------------------------------------------------------------------------------------------------------


# arc length
# https://sites.google.com/site/joebartok/circleradiusarc

# import inspect; print(inspect.getsource(numpy.linspace)) # numpy/scipy documenation in a shell
# import inspect; print(inspect.getsource(np.unique)) # numpy/scipy documenation in a shellnp un
# print u'\u2713'.encode('utf8')

# print np.version.version
# print np.__version__
# import numpy.distutils.system_info as sysinfo
# sysinfo.get_info('atlas')
# sysinfo.get_info('umfpack')
# sysinfo.get_info('openblas')
# sp.__config__.show()
# sysinfo.show_all()


# sudo update-alternatives --config libblas.so.3
# ldd /usr/lib/python2.7/dist-packages/numpy/core/_dotblas.so


# .gala-notification {
#     border: none;
#     border-radius: 4px;
#     background-color: transparent;
#     background-image: linear-gradient(to bottom,
#                                   alpha (@bg_color, 0.98),
#                                   alpha (@bg_color, 0.98) 80%,
#                                   alpha (shade(@bg_color, 0.94), 0.98)
#                                   );
#     box-shadow: inset 0 0 0 1px alpha (#fff, 0.10),
#                 inset 0 1px 0 0 alpha (#fff, 0.90),
#                 inset 0 -1px 0 0 alpha (#fff, 0.30),
#                 0 0 0 1px alpha (#000, 0.20),
#                 0 3px 6px alpha (#000, 0.16),
#                 0 3px 6px alpha (#000, 0.23);



# http://stackoverflow.com/questions/23872946/force-numpy-ndarray-to-take-ownership-of-its-memory-in-cython/
# http://stackoverflow.com/questions/25830764/numpy-with-atlas-or-openblas
# http://danielnouri.org/notes/2012/12/19/libblas-and-liblapack-issues-and-speed,-with-scipy-and-ubuntu/

#------ useful script for testing numpy's blas linkage--------#
# import numpy
# from numpy.distutils.system_info import get_info
# import sys
# import timeit
 
# print("version: %s" % numpy.__version__)
# print("maxint:  %i\n" % sys.maxint)
 
# info = get_info('blas_opt')
# print('BLAS info:')
# for kk, vv in info.iteritems():
#     print(' * ' + kk + ' ' + str(vv))
 
# setup = "import numpy; x = numpy.random.random((3000, 3000))"
# count = 3
 
# t = timeit.Timer("numpy.dot(x, x.T)", setup=setup)
# print("\ndot: %f sec" % (t.timeit(count) / count))
#------------------------------------------------------------#

# g++ -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce/ -I. -L/usr/local/lib -l:libTKIGES.so.9 -l:libTKXSBase.so.9 -l:libTKBRep.so.9 \
# -l:libTKernel.so.9 -l:libTKTopAlgo.so.9 -l:libTKGeomBase.so.9 -l:libTKMath.so.9 -l:libTKHLR.so.9 -l:libTKG2d.so.9 -l:libTKBool.so.9 -l:libTKG3d.so.9 \
# -l:libTKOffset.so.9 -l:libTKG2d.so.9 -l:libTKXMesh.so.9 -l:libTKGeomAlgo.so.9 -l:libTKShHealing.so.9 -l:libTKFeat.so.9 -l:libTKFillet.so.9 -l:libTKBO.so.9 \
# -l:libTKPrim.so.9 -std=c++11 -Wno-unused main.cpp occ_frontend.cpp py_to_occ_frontend.cpp -o main

# g++ -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce/ -I. -L/usr/local/lib /usr/local/lib/libTKIGES.so.9 /usr/local/lib/libTKXSBase.so.9 \
# /usr/local/lib/libTKBRep.so.9 /usr/local/lib/libTKernel.so.9 /usr/local/lib/libTKTopAlgo.so.9 /usr/local/lib/libTKGeomBase.so.9 /usr/local/lib/libTKMath.so.9 \
# /usr/local/lib/libTKHLR.so.9 /usr/local/lib/libTKG2d.so.9 /usr/local/lib/libTKBool.so.9 /usr/local/lib/libTKG3d.so.9 /usr/local/lib/libTKOffset.so.9 \
# /usr/local/lib/libTKG2d.so.9 /usr/local/lib/libTKXMesh.so.9 /usr/local/lib/libTKGeomAlgo.so.9 /usr/local/lib/libTKShHealing.so.9 /usr/local/lib/libTKFeat.so.9 \
# /usr/local/lib/libTKFillet.so.9 /usr/local/lib/libTKBO.so.9 /usr/local/lib/libTKPrim.so.9 -std=c++11 -Wno-unused \
# main.cpp occ_frontend.cpp py_to_occ_frontend.cpp -o main

# g++ -I/home/roman/Dropbox/eigen-devel/ -I/usr/local/include/oce/ -I. -L/usr/local/lib -lTKIGES -lTKXSBase \
# -lTKBRep -lTKernel -lTKTopAlgo -lTKGeomBase -lTKMath -lTKHLR -lTKG2d -lTKBool \
# -lTKXMesh\
# -lTKFillet\
# -lTKGeomBase\
# -lTKPrim\
# -lTKOffset\
# -lTKHLR\
# -lTKMath\
# -lTKBO\
# -lTKG2d\
# -lTKG3d\
# -lTKShHealing\
# -lTKBRep\
# -lTKBool\
# -lTKBRep\
# -lTKTopAlgo\
# -lTKMesh\
# -lTKFeat\
# -lTKGeomAlgo\
# -std=c++11 -Wno-unused \
# main.cpp occ_frontend.cpp py_to_occ_frontend.cpp -o main
