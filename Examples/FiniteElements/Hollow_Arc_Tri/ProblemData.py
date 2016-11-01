import numpy as np 
import os, imp


def ProblemData(MainData):

	# ndim - Dimension of the problem - 1D, 2D, 3D
	MainData.ndim = 2
	# Type of formulation - Displacement-based/ mixed etc
		# 1 - Displacement approach (for electromechanics, it is displacement-electric potential approach)
		# 2 - x, J approach
	MainData.Formulation = 1 	# Displacement-Potential based formulation
	MainData.Analysis = 'Static'
	# MainData.Analysis = 'Dynamic'
	# MainData.AnalysisType = 'Linear'
	MainData.AnalysisType = 'Nonlinear'

	class MaterialArgs(object):
		"""docstring for MaterialArgs"""
		# Type = 'NeoHookean'
		# Type = 'NeoHookean_1'
		# Type = 'LinearModel'
		Type = 'Steinmann'
		# Type = 'LinearisedElectromechanics'
		# mu = 1.5
		# lamb = 4.0
		# rho = 1.0
		# eps_1 = 1.0
		# c1 = 1.0
		# c2 = 1.0

		# mu = 2.3*10e+04
		# lamb = 8.0*10.0e+04
		# rho= 7.5*10e-6
		# eps_1=1505*10.0e-11
		# c1=0.
		# c2=0.

		# mu = 1.
		lamb = 2.
		mu = 0.3571
		lamb = 1.4286
		rho= 7.5*10e-6
		eps_1= 1.0
		c1=0.
		c2=0.

		# mu = 23.3*1000   # N/mm^2
		# lamb = 79.4*1000 # N/mm^2
		# eps_1 = 1.5*10e-11  # C/mm^2

	MainData.MaterialArgs = MaterialArgs

	ProblemPath = os.path.dirname(os.path.realpath(__file__))
	class MeshInfo(object):
		MeshType = 'tri'
		Nature = 'straight'

		# FileName = ProblemPath + '/Mesh_Square_Tri_1.dat'
		# FileName = ProblemPath + '/Mesh_Square_Tri_2.dat'	
		# FileName = ProblemPath + '/Mesh_Square_Tri_16.dat'	
		# FileName = ProblemPath + '/Mesh_Square_Tri_80.dat'
		FileName = ProblemPath + '/Mesh_Square_Tri_212.dat'	
		# FileName = ProblemPath + '/Mesh_Square_Tri_1838.dat'
		# FileName = ProblemPath + '/Mesh_Square_Tri_7748.dat'

		# FileName = ProblemPath + '/Mesh_12.dat'					
		# FileName = ProblemPath + '/Mesh_46.dat'					
		# FileName = ProblemPath + '/Mesh_114.dat'					
		# FileName = ProblemPath + '/Mesh_1096.dat'


		# MeshType = 'quad'
		# FileName = ProblemPath + '/Mesh_Square_Quad_1.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_9.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_4.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_64.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_100.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_400.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_3600.dat'
		# FileName = ProblemPath + '/Mesh_Square_Quad_6400.dat'
		# FileName = ProblemPath + '/Mesh_Hollow_Arc_100.dat'
		# MeshType = 'hex'
		# FileName = ProblemPath + '/Mesh_125.dat'
		# MeshType = 'tet'
		# FileName = ProblemPath + '/Circular_Holes.dat'
		# FileName = ProblemPath + '/Mesh_8th_Sphere.dat'
		# FileName = ProblemPath + '/Mesh_8th_Sphere_514.dat'
		# FileName = ProblemPath + '/Mesh_8th_Sphere_337.dat'
		# FileName = ProblemPath + '/Mesh_8th_Sphere_125.dat'

		# FileName = ProblemPath + '/Mesh_Cube_Tet_12.dat'
		# FileName = ProblemPath + '/Mesh_Cube_Tet_181.dat'
		# FileName = ProblemPath + '/Mesh_Cube_Tet_1473.dat'

		# FileName = ProblemPath + '/Mesh_Plate_12.dat'
		# FileName = ProblemPath + '/Mesh_Plate_97.dat'
		# FileName = ProblemPath + '/Mesh_Plate_363.dat'


	class BoundaryData(object):
		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									

		class NeuArgs(object):
			points=0
			node = 0
			# Applied_at = 'face'
			Applied_at = 'node'
			#--------------------------------------------------------------------------------------------------------------------------#
			# The condition upon which Neumann is applied 
			# - tuple (first is the coordinate direction x=0,y=1,z=2 and second is value of coordinate in that direction e.g. x, y or z) 
			# cond = np.array([[2,10.],[1,2.],[0,2.]])
			cond = np.array([[1,2.]])
			# cond = np.array([[2,0.1]])
			# cond = np.array([[2,2.]])	
			# cond = np.array([[1,2.],[1,0.]])	
			# Loads corresponding to cond
			# Loads = np.array([
			# 	[2000.,0.,0.,0.],
			# 	])
			# Loads = np.array([
			# 	[0.,40000.,0.e0,0.0],
			# 	])
			# 2D
			Loads = np.array([
				[0.2,0.,0.],
				])
			# Number of nodes is necessary
			no_nodes = 0.
			#--------------------------------------------------------------------------------------------------------------------------#


		# Dynamic Data
		nstep = 100
		dt = 1./nstep
		drange = np.linspace(0.,60.,nstep)
		Amp = 10000.0
		DynLoad = Amp*np.sin(drange)
				

		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node 
			mesh_points = DirichArgs.points 

			# # if np.allclose(node[2],0.0):
			# if np.allclose(node[1],0.0):
			# 	# b = np.array([0.,0.,0.,0.])
			# 	# b = np.array([0.,0.,0.2,0.])
			# # elif np.allclose(node[2],2.0):
			# 	# b = np.array([ [],[],0.,[] ])
			# 	# b = [[],[],0.2,[]]

			# 	# 2D
			# 	b = np.array([0.,0.2,0.])
			# else:
			# 	b = [[],[],[],[]]

			# 	# b = [0.,0.,0.,[]]
			# 	# All mechanical variables fixed
			# 	# b = np.array([[[],0.,0.,0.]]); b = np.fliplr(b); b=b.reshape(4)
			# 	# All electric variables fixed
			# 	# b = np.array([[],[],[],0.])

			# 	# b = np.array([[],[],0.2,0.])

			# 	# 2D all elec fixed
			# 	b = np.array([[],[],0.])


			# # ANALYTICAL 2D
			# node = DirichArgs.node 
			# mesh_points = DirichArgs.points 

			# # if np.allclose(node[2],0.0):
			# if np.allclose(node[1],0.0) or np.allclose(node[1],2.0)	or np.allclose(node[0],0.0) or np.allclose(node[0],2.0):
			# 	x = node[0]
			# 	y = node[1]
			# 	b = np.array([ x*np.sin(y), y*np.sin(x), 0.0])
			# 	# b = np.array([0.,0.2,0.])
			# else:
			# 	# 2D all elec fixed
			# 	b = np.array([[],[],0.])


			# # SIMPLE ANALYTICAL 2D - uy
			# node = DirichArgs.node 
			# mesh_points = DirichArgs.points 

			# if np.allclose(node[1],0.0):
			# 	b = np.array([0.0,0.0,0.0])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.0,0.2,0.0])
			# else:
			# 	b = np.array([[],[],0.])
			# 	b[0]=0.0

			# # SIMPLE ANALYTICAL 2D - ux
			# node = DirichArgs.node 
			# mesh_points = DirichArgs.points 

			# if np.allclose(node[1],0.0):
			# 	b = np.array([0.0,0.0,0.0])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.2,0.0,0.0])
			# else:
			# 	b = np.array([[],0.0,0.0])

			#------------------------------------------------------------------
			# # SIMPLE ANALYTICAL 2D - ux
			# mesh_points = DirichArgs.points 

			# if np.allclose(node[0],0.0) or np.allclose(node[0],2.0) or np.allclose(node[1],0.0) or np.allclose(node[1],2.0):
			# 	x=node[0]
			# 	y=node[1]
			# 	# print node 
			# 	# b = np.array([ 0.0,x*np.sin(y),0.0])
			# 	b = np.array([ 0.0,0.1*y**2,0.0 ])
			# 	# if np.allclose(node[0],2.0) and np.allclose(node[1],2.0):
			# 		# print node, b 
			# else:
			# 	# print node 
			# 	b = np.array([[],[],0.0])
			# 	b[0]=0.0
			#------------------------------------------------------------------

			
			# if np.allclose(node[0],0.0) or np.allclose(node[0],2.0) or np.allclose(node[1],0.0) or np.allclose(node[1],2.0):
			# 	x=node[0]
			# 	y=node[1]
			# 	mm=2
			# 	b = np.array([ 0.0,x**mm*y**mm,0.0 ])
			# 	# b = np.array([ 0.0,x*y**mm,0.0 ])
			# else:
			# 	# print node 
			# 	b = np.array([[],[],0.0])
			# 	b[0]=0.0

			# # Poissons
			# if np.allclose(node[0],0.0) or np.allclose(node[0],2.0) or np.allclose(node[1],0.0) or np.allclose(node[1],2.0):
			# 	x=node[0]
			# 	y=node[1]
			# 	mm=2
			# 	b = np.array([ 0.0,0.0,x**mm*y**mm ])
			# 	# b = np.array([ 0.0,x*y**mm,0.0 ])
			# else:
			# 	# print node 
			# 	b = np.array([[],[],[] ])
			# 	b[0]=0.0; b[1]=0.0
			# 	b = [ 0.0,0.0,[] ]
			# 	# print node, b 

			mm=0.05
			if np.allclose(node[1],0.0):
				# print node 
				b=np.array([ 0.0,0.0,0.0])
			elif np.allclose(node[1],2.0):
				# print node 
				b=np.array([ [],mm,0.0])
			else:
				b=np.array([ [],[],0.0 ])

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
						# t = np.array([0.,0.,200000.01e0,0.1e-06])*area/no_nodes
						t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
					else:
						t = [[],[],[],[]]

				# Static Analysis 
				if Analysis=='Static':
					if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
						# print node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]
						# t = np.array([0.,0.,0.e0,-0.2])*area[i]/no_nodes
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

		# def Get(self,Args):
		# 	node = Args.node
		# 	ndim = 2
		# 	nvar = 3

		# 	if node.size==2:
		# 		x = node[0]
		# 		y = node[1]
		# 		sol = np.array([ x*np.sin(y), y*np.sin(x), 0.0])
		# 	else:
		# 		x = node[:,0]
		# 		y = node[:,1]
		# 		ux = x*np.sin(y)
		# 		uy = y*np.sin(x)
		# 		phi = np.zeros(node.shape[0])
		# 		sol = np.zeros((node.shape[0],nvar))
		# 		sol[:,0] = ux
		# 		sol[:,1] = uy
		# 		sol[:,2] = phi

			# u = x*np.sin(y) + y*np.sin(x)


		def Get(self,Args):
			node = Args.node
			ndim = 2
			nvar = 3

			# if node.size==2:
			# 	x = node[0]
			# 	y = node[1]
			# 	sol = np.array([ 0.0, 0.1*y, 0.0])
			# else:
			# 	x = node[:,0]
			# 	y = node[:,1]
			# 	ux = x*0.0
			# 	uy = y*0.1
			# 	phi = np.zeros(node.shape[0])
			# 	sol = np.zeros((node.shape[0],nvar))
			# 	sol[:,0] = ux
			# 	sol[:,1] = uy
			# 	sol[:,2] = phi

			# ux
			# if node.size==2:
			# 	x = node[0]
			# 	y = node[1]
			# 	sol = np.array([ 0.1*y, 0.0*y, 0.0])
			# else:
			# 	x = node[:,0]
			# 	y = node[:,1]
			# 	ux = y*0.1
			# 	uy = y*0.0
			# 	phi = np.zeros(node.shape[0])
			# 	sol = np.zeros((node.shape[0],nvar))
			# 	sol[:,0] = ux
			# 	sol[:,1] = uy
			# 	sol[:,2] = phi
			m=2
			if node.size==2:
				x = node[0]
				y = node[1]
				# sol = np.array([ 0.0,x*np.sin(y),0.0])
				sol = np.array([ 0.0,0.1*y**m,0.0])
			else:
				x = node[:,0]
				y = node[:,1]
				ux = y*0.0
				# uy = x*np.sin(y)
				uy = 0.1*y**m
				phi = np.zeros(node.shape[0])
				sol = np.zeros((node.shape[0],nvar))
				sol[:,0] = ux
				sol[:,1] = uy
				sol[:,2] = phi


			return sol 

			


	return MainData, MeshInfo, BoundaryData, AnalyticalSolution 