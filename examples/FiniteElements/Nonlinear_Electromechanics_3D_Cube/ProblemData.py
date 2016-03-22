import numpy as np 
import os, imp


def ProblemData(MainData):

	# ndim - Dimension of the problem - 1D, 2D, 3D
	MainData.ndim = 3
	# Type of formulation - Displacement-based/ mixed etc
		# 1 - Displacement approach (for electromechanics, it is displacement-electric potential approach)
		# 2 - x, J approach
	MainData.Formulation = 1 	# Displacement-Potential based formulation
	MainData.Analysis = 'Static'
	# MainData.Analysis = 'Dynamic'

	class MaterialArgs(object):
		"""docstring for MaterialArgs"""
		# Type = 'NeoHookean'
		# Type = 'NeoHookean_1'
		# Type = 'Steinmann'
		Type = 'LinearisedElectromechanics'
		# mu = 1.5
		# lamb = 4.0
		# rho = 1.0
		# eps_1 = 1.0
		# c1 = 1.0
		# c2 = 1.0

		mu = 2.3*10e+04
		lamb = 8.0*10.0e+04
		rho= 7.5*10e-6
		eps_1=1505*10.0e-11
		c1=0.
		c2=0.

		# mu = 23.3*1000   # N/mm^2
		# lamb = 79.4*1000 # N/mm^2
		# eps_1 = 1.5*10e-11  # C/mm^2

	ProblemPath = os.path.dirname(os.path.realpath(__file__))
	class MeshInfo(object):
		MeshType = 'hex'
		Nature = 'straight'
		# FileName = ProblemPath + '/Mesh_Cube_1.dat'					
		FileName = ProblemPath + '/Mesh_Cube_8.dat'					
		# FileName = ProblemPath + '/Mesh_Cube_27.dat'					
		# FileName = ProblemPath + '/Mesh_Cube_125.dat'					
		# FileName = ProblemPath + '/Mesh_Cube_1000.dat'

		# FileName = ProblemPath + '/p2-Mesh_Cube_1.dat'
		# FileName = ProblemPath + '/p2-Mesh_Cube_8.dat'	


		# if MainData.C==0:
		# 	# FileName = ProblemPath + '/Mesh_Cube_1.dat'	
		# 	FileName = ProblemPath + '/Mesh_Cube_8.dat'
		# elif MainData.C==1:
		# 	FileName = ProblemPath + '/p2-Mesh_Cube_1.dat'


	class BoundaryData(object):
		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									

		class NeuArgs(object):
			node1=0
			node2=0
			points=0
			node = 0
			# Applied_at = 'face'
			Applied_at = 'node'
			# The condition upon which Neumann is applied 
			# - tuple (first is direction x=0,y=1,z=2 and second is value of in that direction e.g. x, y or z) 
			cond = np.array([[2,2.]])
			# Loads corresponding to cond
			Loads = np.array([
				[0.,0.,200.e0,-0.1]
				])
			# Number of nodes is necessary
			no_nodes = 0.

		# Dynamic Data
		nstep = 100
		dt = 1./nstep
		drange = np.linspace(0.,60.,nstep)
		Amp = 10000.0
		DynLoad = Amp*np.sin(drange)
				

		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node 
			mesh_points = DirichArgs.points 

			if np.allclose(node[2],0.0):
				b = np.array([0.,0.,0.,0.])
			# elif np.allclose(node[1],2.0):
			# 	b = [[],[],0.1]
			else:
				b = [[],[],[],[]]

				# b = [0.,0.,0.,[]]
				# All mechanical variables fixed
				# b = np.array([[[],0.,0.,0.]]); b = np.fliplr(b); b=b.reshape(4)
				# All electric variables fixed
				# b = np.array([[],[],[],0.])

			return b


		
		def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
			# USING THIS APPROACH YOU EITHER NEED TO APPLY FORCE (N) OR YOU SHOULD KNOW THE VALUE OF AREA (M^2)
			node = NeuArgs.node
			# Area should be specified for as many physical (no meshed faces i.e. not mesh.faces) as Neumann is applied 
			area = 1.0*np.array([4.,4.,100.])

			t=[]
			for i in range(0,len(NeuArgs.cond)):
				no_nodes = 1.0*NeuArgs.no_nodes[i] 
				if Analysis != 'Static':
					if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
						t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
					else:
						t = [[],[],[],[]]

				# Static Analysis 
				if Analysis=='Static':
					if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
						# print node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]
						t = NeuArgs.Loads[i,:]*area[i]/no_nodes
					else:
						t = [[],[],[],[]]

			return t


		# class DynamicData(object):
			# nstep = 100
			# dt = 1./nstep
			# drange = np.linspace(0.,60.,nstep)
			# Amp = 100.0
			# DynLoad = Amp*np.sin(drange)





	class AnalyticalSolution(object):
		class Args(object):
			node = 0
			points = 0

		def Get(self,Args):
			node = Args.node

			uz = 2*node[2]

			return np.array([0,0,uz,0,0,0])

			


	return MainData, MaterialArgs, MeshInfo, BoundaryData, AnalyticalSolution 