#!/usr/bin/env pypy
""" THE RunSession ROUTINE RUNS A SPECIFIC SESSION, FOR INSTANCE, A FEM, A BEM OR A COUPLED SESSION.
	THE ENTIRE CODE IS EXECUTED FROM HERE."""


import imp, os, sys, time, cProfile	
from datetime import datetime
import numpy as np
# import scipy as sp 
import multiprocessing as MP
# from numba.decorators import jit
# import cython

# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode

# START THE ANALYSIS
print 'Initiating the routines... Current time is', datetime.now().time()

# ALLOCATE/BUILD THE GENERAL DATA FOR THE ANALYSIS
class MainData(object):
 	"""General data such as directories, files, analysis session, etc that needs to be loaded a priori"""
 	pwd = os.path.dirname(os.path.realpath('__file__'))
 	session = 'FEM'
 	# session = 'BEM'
 	# session = 'BEM3D'
 	# session = 'Coupled'

 	C = 2									# ORDER OF BASIS FUNCTIONS (NOTE THAT C=P-1, WHERE P IS THE POLYNOMIAL DEGREE)
 	norder = 2  							# ORDER/NO OF QUADRATURE POINTS
 	plot = (0,3)							# PLOT FLAG FOR BEM 
 	nrplot = (1,'last')						# PLOT FLAG FOR NEWTON-RAPHSON CONVERGENCE
 	write = 0								# FLAG FOR WRITING THE RESULTS IN VTK/MAT/EPS/DAT ETC
 	Parallel = True 						# MULTI-PROCESSING 
 	nCPU = MP.cpu_count()					# CPU COUNT FOR MULTI-PROCESSING
 	# Parallel = False
 	# nCPU = 8





# RUN THE APPROPRIATE SESSION
# import Main
from Main.FiniteElements.MainFEM import main
# import Main.BoundaryElements.Main_BEM3D as BEM3D
# from Core.FiniteElements.ConvergencePlot import ConvergencePlot
#----------------------------------------------------------------------------------------------------------------------

# FEM SESSION
if MainData.session == 'FEM':
	t_FEM = time.time()
	# cProfile.run('main(MainData)')
	main(MainData)	
	print 'Time taken for the entire analysis was', time.time()-t_FEM, 'seconds \n'
	# MEMORY USAGE INFORMATION
	print 'Global sparse matrix needed', MainData.spmat, 'MB of memory with IJV indices requiring', MainData.ijv, 'MB'


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

# "theme": "Elementary Dark.sublime-theme",








