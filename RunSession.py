#!/usr/bin/env python
""" THE RunSession ROUTINE RUNS A SPECIFIC SESSION, FOR INSTANCE, A FEM, A BEM OR A COUPLED SESSION.
	THE ENTIRE CODE IS EXECUTED FROM HERE."""


import imp, os, sys, time
import cProfile	
# from pympler import tracker, asizeof, summary, muppy
# from memory_profiler import profile
from datetime import datetime
import numpy as np
import scipy as sp 
import multiprocessing as MP
# from mpi4py import MPI
# from numba.decorators import jit
# import cython
import pdb

# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode
# import pypar

# START THE ANALYSIS
print "Initiating the routines... Current time is", datetime.now().time()

# ALLOCATE/BUILD THE GENERAL DATA FOR THE ANALYSIS
class MainData(object):
 	"""General data such as directories, files, analysis session, etc that needs to be loaded a priori"""
 	pwd = os.path.dirname(os.path.realpath('__file__'))
 	session = 'FEM'
 	# session = 'BEM'
 	# session = 'Coupled'


 	__NO_DEBUG__ = True						# ENTER DEBUG MODE OF THE PACKAGE (IF FALSE). ACTIVATES ALL NUMERICAL CHECKS
 	__VECTORISATION__ = True				# ACTIVATE NUMPY'S SIMD INSTRUCTIONS E.G. EINSTEIN SUMMATION (EINSUM) FOR COMPUTING ELEMENTAL MATRICES WITH NO LOOPS.
 	__PARALLEL__ = True 					# ACTIVATE MULTI-PROCESSING 
 	nCPU = MP.cpu_count()					# CPU COUNT FOR MULTI-PROCESSING
 	# __PARALLEL__ = False
 	# nCPU = 8
 	__MEMORY__ = 'SHARED'					# SHARED OR DISTRIBUTED MEMORY FOR PARALLELISATION (FOR DISTRIBUTED MEMORY THE INTERPRETER NEEDS TO BE INVOKED WITH MPI) 
 	# __MEMORY__ = 'DISTRIBUTED'					

 	C = 1									# ORDER OF BASIS FUNCTIONS (NOTE THAT C=P-1, WHERE P IS THE POLYNOMIAL DEGREE)
 	norder = 2  							# ORDER/NO OF QUADRATURE POINTS
 	plot = (0,3)							# PLOT FLAG FOR BEM 
 	nrplot = (0,'last')						# PLOT FLAG FOR NEWTON-RAPHSON CONVERGENCE
 	write = 0								# FLAG FOR WRITING THE RESULTS IN VTK/MAT/EPS/DAT ETC



# import profile_imports
# profile_imports.install()
# import cython
# sys.exit(0)

# import profile_imports_manual
# profile_imports_manual.install()
# from Core.QuadratureRules import GaussQuadrature, QuadraturePointsWeightsTet, QuadraturePointsWeightsTri
# profile_imports_manual.log_stack_info(sys.stderr)
# sys.exit(0)


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