# IMPORT ALL REQUIRED MODULES IN BASE
import os, sys, imp, time
# from time import time
import multiprocessing as MP

# FLORENCE BASE CLASS 
class Base(object):
	"""FLorence base class. General data such as directories, files, analysis session, etc 
		that needs to be loaded a priori are stored
		
		pwd:                      	Florence's top level directory
		session:                    {'FEM','BEM','Coupled'} Session to run
		__NO_DEBUG__:               Enter debug mode of the package (if false). Activates all numerical checks                 
		__VECTORISATION__:          Activate numpy's (einsum) for computing elemental matrices with no loops
		__PARALLEL__:               Activate multiprocessing for either shared or distributed memory or both
		numCPU:                     Number of concurrent cores/hyperthreads for parallelisation
		__MEMORY__:			        {'SHARED','DISTRIBUTED','AUTO','HYBRID'} Option for shared/distributed 
									memory parallelisation
		
		C:                          [int] order of basis functions. Note that C=P-1 where P is polynomial degree
		norder:                     [int] number of quadrature points 
		plot:                       [tuple of ints] plot flag for BEM
		nrplot:                     [tuple] plot flag for Newton-Raphson convergence
		write:                      [boolean] flag for writting simulation results in .vtu/.mat/.eps/.dat formats
		
		"""

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



	# PROBLEM SPATIAL DIMENSION- 1D, 2D, 3D
	ndim = 2
	nvar = ndim
	Fields = 'Mechanics'
	# Fields = 'ElectroMechanics'
	
	Formulation = 'DisplacementApproach'
	# Formulation = 'DisplacementElectricPotentialApproach'

	Analysis = 'Static'
	# Analysis = 'Dynamic'
	AnalysisType = 'Linear'
	# AnalysisType = 'Nonlinear'


	Timer = 0





	# MaterialArgs class
	class MaterialArgs(object):
		"""Container for all material models"""

		E = 0.
		nu = 0.
		lamb = 0.
		mu = 0.
		rho = 0.

		def __init__(self):
			pass

		# def AnisotropicFibreOrientation(self,mesh):
			# pass


	# MeshInfo class 
	class MeshInfo(object):
		"""Container for mesh information"""
		MeshType = 'tri'
		Reader = 'Read'
		ProblemPath = ''

		def __init__(self,path):
			"""Invoke it like so to get path of the current file:
			>>> MainData.MeshInfo(__file__)

			 """
			self.ProblemPath = os.path.dirname(os.path.realpath(__file__))


	# ScaledJacobian
	isScaledJacobianComputed = False
