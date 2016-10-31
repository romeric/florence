import numpy as np 



def ProblemData(MainData):

	# ndim - Dimension of the problem - 1D, 2D, 3D
	MainData.ndim = 3

	class MaterialArgs(object):
		"""docstring for MaterialArgs"""
		# Type = 'NeoHookean'
		Type = 'NeoHookean_1'
		mu = 1.5
		lamb = 4.0


	class MeshInfo(object):
		MeshType = 'hex'
		Nature = 'straight'
		FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_1.dat'
		# FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_1a.dat'
		# FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_8.dat'
		# FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_27.dat'
		# FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_125.dat'
		# FileName = '/home/roman/Dropbox/Python/Problems/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_1000.dat'



	class BoundaryData(object):
		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									

		class NeuArgs(object):
			node1=0
			node2=0
			points=0
			node = 0
			p_i = 0.5
			p_o = -1.0
			Applied_at = 'face'
				

		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node 
			mesh_points = DirichArgs.points 

			if np.allclose(node[2],0.0):
				b = np.array([0.,0.,0.])
			# elif np.allclose(node[1],2.0):
			# 	b = [[],[],0.1]
			else:
				b = [[],[],[]]

			return b


		
		def NeumannCriterion(self,NeuArgs):
			points = NeuArgs.points
			node1 = points[0,:]
			node2 = points[1,:]
			node3 = points[2,:]
			node4 = points[3,:]

			t = np.zeros((1,6))

			if np.allclose(node1[2],2) and np.allclose(node2[2],2) and np.allclose(node3[2],2) and np.allclose(node4[2],2):
				t = np.array([0.,0.,-2.5e-0]).reshape(1,3)

			return t




	class AnalyticalSolution(object):
		class Args(object):
			node = 0
			points = 0

		def Get(self,Args):
			node = Args.node

			uz = 2*node[2]

			return np.array([0,0,uz,0,0,0])

			


	return MainData, MaterialArgs, MeshInfo, BoundaryData, AnalyticalSolution 