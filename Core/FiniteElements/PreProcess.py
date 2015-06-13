import os, imp, sys
from time import time
import numpy as np
import scipy as sp 

# pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

import Core.MaterialLibrary as MatLib 
# from Core.MaterialLibrary import *
from Core.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Core.MeshGeneration.ReadSalomeMesh import ReadMesh
from Core.MeshGeneration.HigherOrderMeshing import *
from Core.NumericalIntegration import GaussQuadrature, QuadraturePointsWeightsTet, QuadraturePointsWeightsTri
from Core.FiniteElements.GetBases import *
import Core.Formulations.DisplacementElectricPotentialApproach as DEPB
import Core.Formulations.DisplacementApproach as DB
from Core import Mesh

# import cProfile
from Core.Supplementary.Timing.Timing import timing
@timing
def PreProcess(MainData,Pr,pwd):

	# PARALLEL PROCESSING
	############################################################################
	try:
		# CHECK IF MULTI-PROCESSING IS ACTIVATED
		MainData.Parallel
	except NameError:
		# IF NOT THEN ASSUME SINGLE PROCESS
		MainData.Parallel = False
		MainData.nCPU = 1

	#############################################################################

	# READ MESH-FILE
	############################################################################
	mesh = Mesh()
	mesh.Read(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)

	# sys.exit("STOPPED")
	# GENERATE pMESHES IF REQUIRED
	# if MainData.C==0:
	# 	nmesh = mesh
	# else:
	# 	print 'Generating p = '+str(MainData.C+1)+' mesh based on the linear mesh...'
	# 	t_mesh = time()
	# 	# BUILD A NEW MESH BASED ON THE LINEAR MESH
	# 	if MainData.MeshInfo.MeshType == 'tri':
	# 		# BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TRIANGLES  
	# 		# nmesh = HighOrderMeshTri(MainData.C,mesh,MainData.MeshInfo,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 		nmesh = HighOrderMeshTri_UNSTABLE(MainData.C,mesh,Decimals=10,Parallel=MainData.Parallel,nCPU=MainData.nCPU)

	# 		# nmesh1 = HighOrderMeshTri(MainData.C,mesh,MainData.MeshInfo,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 		# nmesh2 = HighOrderMeshTri_UNSTABLE(MainData.C,mesh,MainData.MeshInfo,Decimals=10,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 	elif MainData.MeshInfo.MeshType == 'tet':
	# 		# BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TETRAHEDRALS
	# 		# nmesh = HighOrderMeshTet(MainData.C,mesh,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 		nmesh = HighOrderMeshTet_UNSTABLE(MainData.C,mesh,Decimals=10,Zerofy=0,Parallel=MainData.Parallel,nCPU=MainData.nCPU)

	# 		# nmesh1 = HighOrderMeshTet(MainData.C,mesh,MainData.MeshInfo,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 		# nmesh2 = HighOrderMeshTet_UNSTABLE(MainData.C,mesh,MainData.MeshInfo,Decimals=10,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# 	elif MainData.MeshInfo.MeshType == 'quad':
	# 		# BUILD A NEW MESH USING THE GAUSS-LOBATTO NODAL POINTS FOR QUADS
	# 		nmesh = HighOrderMeshQuad(MainData.C,mesh,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
		
	# 	print 'Finished generating the high order mesh. Time taken', time()-t_mesh,'sec'
	############################################################################
	# index_sort_x = np.argsort(nmesh.points[:,0])
	# sorted_repoints = nmesh.points[index_sort_x,:]
	# print
	# print sorted_repoints
	# print nmesh.elements
	# print nmesh.points 
	# print np.linalg.norm(nmesh1.points-nmesh2.points)
	# print np.linalg.norm(nmesh1.elements-nmesh2.elements)
	# print np.linalg.norm(nmesh1.edges-nmesh2.edges)
	# print np.linalg.norm(nmesh1.faces-nmesh2.faces)
	# print
	# help(MainData)
	# print '\n',np.max(nmesh.elements)+1, nmesh.points.shape[0]
	# print 'Number of nodes: ', nmesh.points.shape[0]
	# np.savetxt('/home/roman/Dropbox/time.dat',np.array([time()-t_mesh, nmesh.points.shape[0]]))
	

	# Mesh().GetBoundaryEdgesTri(mesh.elements)
	# mesh = Mesh()
	# print dir(mesh1)
	# mesh.Read(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)
	# mesh.edges=[]
	# mesh.faces=[]
	# print dir(mesh1)
	# print mesh.elemtype
	# print mesh.points
	# print mesh.elements
	# mesh.GetBoundaryEdgesTri()
	# print mesh.faces
	# mesh.GetBoundaryEdgesTet()
	# mesh.GetBoundaryFacesTet()
	# print mesh.faces
	# print mesh.edges
	# mesh.WriteVTK()
	# mesh.WriteVTK(fname='/home/roman/Desktop/toba.vtu',pdata=mesh.points[:,1])

	# print mesh.elements
	if MainData.C>0:
		mesh.GetHighOrderMesh(MainData.C,Parallel=MainData.Parallel,nCPU=MainData.nCPU)
	# print mesh.elements

	# sys.exit("STOPPED")



	# STORE PATHS FOR MAIN, CORE & PROBLEM DIRECTORIES
	############################################################################
	class Path(object):
		"""docstring for Path"""
		def __init__(self, arg):
			super(Path, self).__init__()
			self.arg = arg

		TopLevel = pwd
		Main = pwd+'/Main/FiniteElements'
		Problem = os.path.dirname(Pr.__file__)
		Core = pwd+'/Core/FiniteElements'

	MainData.Path = Path


	if os.path.isdir(MainData.Path.Problem+'/Results'):
		print 'Writing results in the problem directory:', MainData.Path.Problem
	else:
		print 'Writing the results in problem directory:', MainData.Path.Problem
		os.mkdir(MainData.Path.Problem+'/Results')


	MainData.Path.ProblemResults = MainData.Path.Problem+'/Results/'
	MainData.Path.ProblemResultsFileNameMATLAB = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(MainData.C)+'.mat'
	# For nonlinear analysis - do not add the extension
	MainData.Path.ProblemResultsFileNameVTK = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(MainData.C)
	# For linear analysis
	# MainData.Path.ProblemResultsFileNameVTK = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(MainData.C)+'.vtu'

	# CONSIDERATION OF MATERAIL MODEL
	MainData.Path.MaterialModel = MainData.MaterialArgs.Type + '_Model/'

	# Analysis specific directories
	if MainData.Analysis == 'Static':
		# MainData.Path.LinearStatic = 'LinearStatic'		# One step/increment
		# MainData.Path.LinearDynamic = 'LinearDynamic'
		MainData.Path.Analysis = 'NonlinearStatic/' 	# Increments
		# Subdirectories
		if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis):
			if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel):
				pass
			else:
				os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel)
		else:
			os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis)
			if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel):
				pass
			else:
				os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel)

	elif MainData.Analysis == 'Dynamic':
		MainData.Path.Analysis = 'NonlinearDynamic/'
		# SUBDIRECTORIES
		if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis):
			if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel):
				pass
			else:
				os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel)
		else:
			os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis)
			if os.path.isdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel):
				pass
			else:
				os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel)
	############################################################################



	# COMPUTING BASES FUNCTIONS AT ALL INTEGRATION POINTS
	############################################################################
	# GET QUADRATURE POINTS AND WEIGHTS
	z=[]; w=[]; 
	QuadratureOpt=1 	# OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS


	if MainData.MeshInfo.MeshType == 'quad' or MainData.MeshInfo.MeshType == 'hex':
		z, w = GaussQuadrature(MainData.C+MainData.norder,-1.,1.)
	elif MainData.MeshInfo.MeshType == 'tet':
		zw = QuadraturePointsWeightsTet.QuadraturePointsWeightsTet(MainData.C+1,QuadratureOpt)
		z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]; #w = np.repeat(w,MainData.ndim) 
	elif MainData.MeshInfo.MeshType == 'tri':
		zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(MainData.C+2,QuadratureOpt) # PUT C+4 OR HIGHER
		z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]

	class Quadrature(object):
		"""docstring for Quadrature"""
		points = z
		weights = w
		Opt = QuadratureOpt

	if MainData.ndim == 3:
		# GET BASES AT ALL INTEGRATION POINTS (VOLUME)
		Domain = GetBases3D(MainData.C,Quadrature,MainData.MeshInfo.MeshType)
		# GET BOUNDARY BASES AT ALL INTEGRATION POINTS (SURFACE)
		# Boundary = GetBasesBoundary(MainData.C,z,MainData.ndim)
	elif MainData.ndim == 2: 
		# Get basis at all integration points (surface)
		Domain = GetBases(MainData.C,Quadrature,MainData.MeshInfo.MeshType)
		# GET BOUNDARY BASES AT ALL INTEGRATION POINTS (lINE)
		# Boundary = GetBasesBoundary(MainData.C,z,MainData.ndim)
	Boundary = []

	############################################################################



	# COMPUTING GRADIENTS AND JACOBIAN A PRIORI FOR ALL INTEGRATION POINTS
	############################################################################
	Domain.Jm = []; Domain.AllGauss=[]
	if MainData.MeshInfo.MeshType == 'hex':
		Domain.Jm = np.zeros((MainData.ndim,Domain.Bases.shape[0],w.shape[0]**MainData.ndim))	
		Domain.AllGauss = np.zeros((w.shape[0]**MainData.ndim,1))	
		counter = 0
		for g1 in range(0,w.shape[0]):
			for g2 in range(0,w.shape[0]): 
				for g3 in range(0,w.shape[0]):
					# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
					Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
					Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
					Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

					Domain.AllGauss[counter,0] = w[g1]*w[g2]*w[g3]

					counter +=1

	elif MainData.MeshInfo.MeshType == 'quad':
		Domain.Jm = np.zeros((MainData.ndim,Domain.Bases.shape[0],w.shape[0]**MainData.ndim))	
		Domain.AllGauss = np.zeros((w.shape[0]**MainData.ndim,1))	
		counter = 0
		for g1 in range(0,w.shape[0]):
			for g2 in range(0,w.shape[0]): 
				# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
				Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
				Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

				Domain.AllGauss[counter,0] = w[g1]*w[g2]
				counter +=1

	elif MainData.MeshInfo.MeshType == 'tet':
		Domain.Jm = np.zeros((MainData.ndim,Domain.Bases.shape[0],w.shape[0]))	
		Domain.AllGauss = np.zeros((w.shape[0],1))	
		for counter in range(0,w.shape[0]):
			# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
			Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
			Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
			Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

			Domain.AllGauss[counter,0] = w[counter]

	elif MainData.MeshInfo.MeshType == 'tri':
		Domain.Jm = [];  Domain.AllGauss = []

		Domain.Jm = np.zeros((MainData.ndim,Domain.Bases.shape[0],w.shape[0]))	
		Domain.AllGauss = np.zeros((w.shape[0],1))	
		for counter in range(0,w.shape[0]):
			# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
			Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
			Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

			Domain.AllGauss[counter,0] = w[counter]

	MainData.Domain = Domain
	MainData.Boundary = Boundary
	############################################################################







	#############################################################################
							# MATERIAL MODEL PRE-PROCESS
	#############################################################################
	#############################################################################

	# STRESS COMPUTATION FLAGS FOR LINEARISED ELASTICITY
	###########################################################################
	MainData.Prestress = 0
	if MainData.MaterialArgs.Type == 'Incrementally_Linearised_NeoHookean':
		# RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE WITHOUT UPDATING THE GEOMETRY BUT WITH GEOMETRIC TERM
		MainData.Prestress = 1
		if MainData.Fields == 'Mechanics':
			Hsize = 6 if MainData.ndim == 3 else 3
		elif MainData.Fields == 'ElectroMechanics':
			Hsize = 9 if MainData.ndim == 3 else 5
		else:
			raise NotImplementedError('H_Voigt size not giveN')

		MainData.MaterialArgs.H_Voigt = np.zeros((Hsize,Hsize,nmesh.nelem,Quadrature.weights.shape[0]),dtype=np.float64)
		MainData.MaterialArgs.Sigma = np.zeros((MainData.ndim,MainData.ndim,nmesh.nelem,Quadrature.weights.shape[0]),dtype=np.float64)


	# CHOOSE AND INITIALISE THE RIGHT MATERIAL MODEL 
	##############################################################################

	# GET THE MEHTOD NAME FOR THE RIGHT MATERIAL MODEL
	MaterialFuncName = getattr(MatLib,MainData.MaterialArgs.Type)
	# INITIATE THE FUNCTIONS FROM THIS MEHTOD
	MainData.nvar, MainData.MaterialModelName = MaterialFuncName(MainData.ndim).Get()
	MainData.Hessian = MaterialFuncName(MainData.ndim).Hessian
	MainData.CauchyStress = MaterialFuncName(MainData.ndim).CauchyStress

	# INITIALISE
	StrainTensors = KinematicMeasures(np.diag(np.ones(MainData.ndim))).Compute(MainData.AnalysisType)
	MaterialFuncName(MainData.ndim).Hessian(MainData.MaterialArgs,MainData.ndim,StrainTensors,elem=0,gcounter=0)

	##############################################################################




	# FORMULATION TYPE FLAGS
	#############################################################################
	if MainData.Formulation == 1:
		if MainData.MaterialArgs.Type == 'IsotropicElectroMechanics_1' or MainData.MaterialArgs.Type == 'Steinmann' or \
		MainData.MaterialArgs.Type == 'AnisotropicMooneyRivlin_1_Electromechanics' or MainData.MaterialArgs.Type == 'LinearisedElectromechanics' or \
		MainData.MaterialArgs.Type == 'LinearModelElectromechanics':
			MainData.ConstitutiveStiffnessIntegrand = DEPB.ConstitutiveStiffnessIntegrand
			MainData.GeometricStiffnessIntegrand = DEPB.GeometricStiffnessIntegrand
			MainData.MassIntegrand =  DEPB.MassIntegrand

		elif MainData.MaterialArgs.Type == 'LinearModel' or MainData.MaterialArgs.Type == 'AnisotropicMooneyRivlin_1' or \
		MainData.MaterialArgs.Type == 'Incrementally_Linearised_NeoHookean':
			MainData.ConstitutiveStiffnessIntegrand = DB.ConstitutiveStiffnessIntegrand
			MainData.GeometricStiffnessIntegrand = DB.GeometricStiffnessIntegrand
			MainData.MassIntegrand =  DB.MassIntegrand




	# GEOMETRY UPDATE FLAGS
	###########################################################################
	if MainData.MaterialArgs.Type == 'LinearisedElectromechanics' or MainData.MaterialArgs.Type == 'LinearModel' or \
	MainData.MaterialArgs.Type == 'LinearModelElectromechanics' or MainData.MaterialArgs.Type == 'Incrementally_Linearised_NeoHookean':
		# RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE WITHOUT UPDATING THE GEOMETRY
		MainData.GeometryUpdate = 0
	else:
		MainData.GeometryUpdate = 1









	# CHOOSING THE SOLVER
	#############################################################################
	# MainData.Solver = 'iterative'
	# MainData.Solver = 'direct'
	class solve(object):
		"""docstring for solve"""
		tol = 1e-07

	if mesh.points.shape[0]*MainData.nvar > 300000:
		solve.type = 'iterative'
		print 'Large system of equations. Switching to iterative solver'
	else:
		solve.type = 'direct'

	MainData.solve = solve 
			

	if mesh.nelem > 100000:
		MainData.AssemblyRoutine = 'Large'
		print 'Large number of elements. Switching to faster assembly routine'
	else:
		MainData.AssemblyRoutine = 'Small'
		# print 'Small number of elements. Sticking to small assembly routine'


	#############################################################################



	# MINIMAL MAINDATA VARIABLES
	############################################################################
	class Minimal(object):
		"""docstring for Minimal"""
		def __init__(self, arg):
			super(Minimal, self).__init__()
			self.arg = arg
		C = MainData.C
		nvar = MainData.nvar
		ndim = MainData.ndim

	MainData.Minimal = Minimal
	#############################################################################



	# DICTIONARY OF SAVED VARIABLES
	#############################################################################
	if MainData.write == 0:
		MainData.MainDict = {'ProblemPath': MainData.Path.ProblemResults, 
		 'MeshPoints':mesh.points, 'MeshElements':mesh.elements, 'MeshFaces':mesh.faces, 'MeshEdges':mesh.edges,
		 'Solution':[], 'DeformationGradient':[], 'CauchyStress':[],
		 'SecondPiolaStress':[], 'ElectricField':[], 'ElectricDisplacement':[]}

	#############################################################################
			
			
	# PLACE IN MAINDATA
	MainData.Quadrature = Quadrature

	return mesh




