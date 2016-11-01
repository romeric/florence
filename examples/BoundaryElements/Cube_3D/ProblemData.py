import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
from Core.MeshGeneration.MeshGeneration import MeshGeneration


#############################################################################################
def ProblemData(general_data):

	class mesh_info(object):
		kind = 'hex'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/MultiPhysics_3D_Cube/Mesh_Cube_125.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/MultiPhysics_3D_Cube/Mesh_Cube_1000.dat'
		nature = 'straight'


	class BoundaryData(object):
		class DirichArgs(object):
			node1=0
			node2=0
			points=0
			# path_potential='/home/roman/Dropbox/Matlab_Files/potential_arc.txt'
			# path_displacement='/home/roman/Dropbox/Matlab_Files/displacement_arc.txt'
			node = 0
			Applied_at = 'node' 	# This argument explains if the boundary conditions are to be applied at meshed or initial geometry
									# In case of initial geometry this would be edges i.e. Applied_at = 'edge'


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
			# path_potential = DirichArgs.path_potential
			# path_displacement = DirichArgs.path_displacement
			# displacements = np.loadtxt(path_displacement)
			# potentials = np.loadtxt(path_potential)
			# points = DirichArgs.points



			if np.allclose(node[2],0.0):
				b = np.array([0.,0.,0.,0.,0.,0.])
			# elif np.allclose(node[2],2.0):
			# 	b = np.array([0.,0.,0.,0.,10.,0.])
			# else:
			# 	b  = [[0.,0.,0.,0.,[],0.]]
			else:
				b = [[],[],[],0.,0.,0.]



			return b
			###################################################


		
		def NeumannCriterion(self,NeuArgs):
			points = NeuArgs.points
			node1 = points[0,:]
			node2 = points[1,:]
			node3 = points[2,:]
			node4 = points[3,:]

			t = np.zeros((1,6))

			if np.allclose(node1[2],2) and np.allclose(node2[2],2) and np.allclose(node3[2],2) and np.allclose(node4[2],2):
				t = np.array([0.,0.,0.1,0.,0.,0.]).reshape(1,6)

			return t


	return general_data, mesh_info, BoundaryData




