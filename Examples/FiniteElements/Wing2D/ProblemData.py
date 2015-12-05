import numpy as np 
import os, imp


def ProblemData(MainData):

	MainData.ndim = 2	
	MainData.Fields = 'Mechanics'
	# MainData.Fields = 'ElectroMechanics'
	MainData.Formulation = 'DisplacementApproach'
	MainData.Analysis = 'Static'
	# MainData.Analysis = 'Dynamic'
	# MainData.AnalysisType = 'Linear'
	MainData.AnalysisType = 'Nonlinear'

	# MATERIAL INPUT DATA 
	# MainData.MaterialArgs.Type = 'LinearModel'
	# MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
	# MainData.MaterialArgs.Type = 'IncrementallyLinearisedNeoHookean'
	# MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin_1'
	# MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
	# MainData.MaterialArgs.Type = 'NeoHookean_1'
	MainData.MaterialArgs.Type = 'NeoHookean_2'
	# MainData.MaterialArgs.Type = 'MooneyRivlin'
	# MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
	# MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 
	# MainData.MaterialArgs.Type = 'TranservselyIsotropicHyperElastic'

	# E = MainData.E 
	# nu = MainData.nu 

	MainData.MaterialArgs.E  = 1.0e5
	MainData.MaterialArgs.nu = 0.001

	E = MainData.MaterialArgs.E
	nu = MainData.MaterialArgs.nu
	# GET LAME CONSTANTS
	MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
	MainData.MaterialArgs.mu = E/2./(1+nu)

	# MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E
	# E = MainData.MaterialArgs.E
	# nu = MainData.MaterialArgs.nu
	# E_A = MainData.MaterialArgs.E_A
	# MainData.MaterialArgs.G_A = (E*(E_A*nu - E_A + E_A*nu**2 + E*nu**2))/(2*(nu + 1)*(2*E*nu**2 + E_A*nu - E_A))




	ProblemPath = os.path.dirname(os.path.realpath(__file__))
	MainData.MeshInfo.MeshType = "tri"
	# MainData.MeshInfo.Reader = "Read"
	MainData.MeshInfo.Reader = "ReadHighOrderMesh"
	MainData.MeshInfo.Format = "GID"
	
	MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch25.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch50.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch100.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch200.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch400.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch800.dat'
	# MainData.MeshInfo.FileName = ProblemPath + '/sd7003_Stretch1600.dat'

	MainData.MeshInfo.IsHighOrder = True
		


	class BoundaryData(object):
		# NURBS/NON-NURBS TYPE BOUNDARY CONDITION
		Type = 'nurbs'
		RequiresCAD = True
		# ProjectionType = 'orthogonal' # gives worse quality for wing 2D - checked
		ProjectionType = 'arc_length'
		CurvilinearMeshNodalSpacing = 'fekete'

		scale = 1000.
		condition = 1 # this condition is not used

		IGES_File = ProblemPath + '/sd7003.igs'

		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									

		class NeuArgs(object):
			pass
				

		def DirichletCriterion(self,DirichArgs):
			pass


		def NURBSParameterisation(self):
			pass

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
				# if x > -510 and x < 510 and y > -10 and y < 10:
				if x > -630 and x < 830 and y > -100 and y < 300:	
					projection_edges[iedge]=1
			
			return projection_edges


		
		def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
			pass


	class AnalyticalSolution(object):
		class Args(object):
			node = 0
			points = 0




		def Get(self,Args):
			node = Args.node
			ndim = 2
			nvar = 3

			m=2
			if node.size==2:
				x = node[0]
				y = node[1]
				# sol = np.array([ 0.0,x*np.sin(y),0.0])
				sol = np.array([ 0.0,0.1*y**m])
			else:
				x = node[:,0]
				y = node[:,1]
				ux = y*0.0
				# uy = x*np.sin(y)
				uy = 0.1*y**m
				sol = np.zeros((node.shape[0],nvar))
				sol[:,0] = ux
				sol[:,1] = uy

			return sol 

			
	# PLACE THEM ALL INSIDE THE MAIN CLASS
	MainData.BoundaryData = BoundaryData
	MainData.AnalyticalSolution = AnalyticalSolution
