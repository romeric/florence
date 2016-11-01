import numpy as np 


def ProblemData(general_data):

	# Degree of continuity 
	# C = 1
	C = general_data.C
	# nvar is the the dimension of vectorial field we are solving for.
		# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
		# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz
	nvar = 3

	# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		ndiv = 50.




	mu = 1.5
	lam = 3


	E = mu*(3.0*lam+2.0*mu)/(mu+lam)
	nu = lam/2.0/(mu+lam)
	
	D = 1e+20*E/(1.0+nu)*(1.0-2.0*nu)*np.array([
    	[1-nu,nu,0], 
    	[nu,1-nu,0],
    	[0,0,(1.0-2.0*nu)/2.0]
    	])


	P = np.zeros((3,2))


	e = np.array([
		[2.0,0.0],
		[0.0,3.0]
		])



	class material_args(object):
		"""docstring for material_args"""
		def __init__(self, arg):
			super(material_args, self).__init__()
			self.arg = arg
		elastic = D 
		piezoelectric = P
		electric = e 			


	class mesh_info(object):
		kind = 'quad'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Square_Piezo/Mesh_Square_9.dat'
		nature = 'straight'



	class BoundaryData(object):
		class DirichArgs(object):
			node1=0
			node2=0
			points=0
			path_potential='/home/roman/Dropbox/Matlab_Files/potential_arc.txt'
			path_displacement='/home/roman/Dropbox/Matlab_Files/displacement_arc.txt'
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
			Applied_at = 'node'
				

		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node 
			mesh_points = DirichArgs.points 
			path_potential = DirichArgs.path_potential
			path_displacement = DirichArgs.path_displacement
			# displacements = np.loadtxt(path_displacement)
			# potentials = np.loadtxt(path_potential)
			# points = DirichArgs.points

			################################################
			# Electro
			# if np.allclose(node[1],0):
			# 	b = np.array([0,0,5.0])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0,0,10])
			# else:
			# 	b=np.array([[[],0,0]]); b = np.fliplr(b);
			# 	b= b.reshape(b.shape[1])

			# Mech ux
			# if np.allclose(node[0],0):
			# 	b = np.array([15.0,0,0.0])
			# elif np.allclose(node[0],2.0):
			# 	b = np.array([20.0,0,0])
			# else:
			# 	b  = np.array([[],0,0])

			# Mech uy
			if np.allclose(node[1],0):
				b = np.array([0.0,1.5,0.0])
			elif np.allclose(node[1],2.0):
				b = np.array([0.0,4.0,0])
			else:
				b  = np.array([[],[],0])
				b[0] = 0.0 

			# # Mech ux and uy
			# if np.allclose(node[0],0):
			# 	b = np.array([-1.0,1.5,0.0])
			# elif np.allclose(node[0],2.0):
			# 	b = np.array([-2.0,3.0,0])
			# else:
			# 	b  = np.array([[],[],0])

			return b
			###################################################


		
		def NeumannCriterion(self,NeuArgs):

			return 0




	class AnalyticalSolution(object):
		class Args(object):
			node1 = 0
			node2 = 0
			node = 0
			points = 0

		def Get(self,Args):
			node = Args.node

			# Electrostatics
			# ux=0; uy=0
			# phi = 2.5*node[1]+5.0

			# Mech Ux
			# uy=0; phi=0; ux = 2.5*node[0]+15.0

			# Mech Uy
			ux=0; phi=0; uy = 0.75*node[1]+1.5


			return np.array([ux,uy,phi])

			


	return C, nvar, geo_args, material_args, mesh_info, BoundaryData, AnalyticalSolution 