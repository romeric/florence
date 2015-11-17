import numpy as np 
import os, imp


def ProblemData(MainData):

	MainData.ndim = 2	
	MainData.Fields = 'Mechanics'	
	MainData.Formulation = 'DisplacementApproach'
	MainData.Analysis = 'Static'
	# MainData.AnalysisType = 'Linear'
	MainData.AnalysisType = 'Nonlinear'

	# MATERIAL INPUT DATA 
	# MainData.MaterialArgs.Type = 'LinearModel'
	# MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
	# MainData.MaterialArgs.Type = 'IncrementallyLinearisedNeoHookean'
	# MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
	# MainData.MaterialArgs.Type = 'NeoHookean_1'
	MainData.MaterialArgs.Type = 'NeoHookean_2'
	# MainData.MaterialArgs.Type = 'MooneyRivlin'
	# MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
	# MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 

	MainData.MaterialArgs.E  = 1.0e5
	MainData.MaterialArgs.nu = 0.35

	# MainData.MaterialArgs.E = MainData.E 
	# MainData.MaterialArgs.nu = MainData.nu
	# print 'Poisson ratio is:', MainData.MaterialArgs.nu


	E = MainData.MaterialArgs.E
	nu = MainData.MaterialArgs.nu
	# GET LAME CONSTANTS
	MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
	MainData.MaterialArgs.mu = E/2./(1+nu)
	# lamb = lamb + mu


	ProblemPath = os.path.dirname(os.path.realpath(__file__))
	MainData.MeshInfo.MeshType = "tri"
	MainData.MeshInfo.Reader = "Read"
	MainData.MeshInfo.FileName = ProblemPath + '/MechanicalComponent2D_192.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/MechanicalComponent2D_664.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_321.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_2672.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_236.dat'
		


	class BoundaryData(object):
		# NURBS/NON-NURBS TYPE BOUNDARY CONDITION
		Type = 'nurbs'
		RequiresCAD = True
		# ProjectionType = 'orthogonal'
		ProjectionType = 'arc_length'
		CurvilinearMeshNodalSpacing = 'fekete'
		# CurvilinearMeshNodalSpacing = 'equal'

		scale = 1.
		condition = 1e10

		IGES_File = ProblemPath + '/mechanical2D.iges'
		# IGES_File = ProblemPath + '/mechanical2d.igs' # non-smooth

		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									
		class NeuArgs(object):
			pass			

		def DirichletCriterion(self,DirichArgs):
			pass


		def NURBSParameterisation(self):
			import scipy.io as spio 
			dummy_nurbs, nurbs={}, {}
			spio.loadmat('/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.mat',mdict=dummy_nurbs)
			nurbs['Pw'] = dummy_nurbs['nurbs']['Pw'][0]
			nurbs['U'] = dummy_nurbs['nurbs']['U'][0]
			nurbs['start'] = dummy_nurbs['nurbs']['iniParam'][0]
			nurbs['end'] = dummy_nurbs['nurbs']['endParam'][0]

			fnurbs=[]
			from Core.Supplementary.Tensors import itemfreq_py
			for i in range(0,nurbs['Pw'].shape[0]):
				degree = np.int64(itemfreq_py(nurbs['U'][i])[-1,-1] - 1)
				fnurbs.append(({'U':nurbs['U'][i],'Pw':nurbs['Pw'][i],
					'start':nurbs['start'][i][0][0],'end':nurbs['end'][i][0][0],'degree':degree}))

			return fnurbs 

		def NURBSCondition(self,x):
			return np.sqrt(x[:,0]**2 + x[:,1]**2) < 0.1
			

		def ProjectionCriteria(self,mesh):
			projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
			num = mesh.edges.shape[1]
			for iedge in range(mesh.edges.shape[0]):
				x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
				y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
				x *= self.scale
				y *= self.scale 
				if np.sqrt(x*x+y*y)< self.condition:
					projection_edges[iedge]=1
			
			return projection_edges


		
		def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
			pass


	class AnalyticalSolution(object):

		class Args(object):
			node = 0
			points = 0

		def Get(self,Args):
			pass

			
	# PLACE THEM ALL INSIDE THE MAIN CLASS
	MainData.BoundaryData = BoundaryData
	MainData.AnalyticalSolution = AnalyticalSolution

