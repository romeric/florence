import numpy as np 


def ProblemData(general_data):

	# Degree of continuity 
	# C = 1
	C = general_data.C
	# nvar is the the dimension of vectorial field we are solving for.
		# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
		# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz
	general_data.nvar = 6
	# ndim - Dimension of the problem - 1D, 2D, 3D
	general_data.ndim = 3


	# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		ndiv = 50.




	# mu = 1.5
	# lam = 3
	# E = mu*(3.0*lam+2.0*mu)/(mu+lam)
	# nu = lam/2.0/(mu+lam)
	
	# Elastic
	D = np.diag(np.ones(6))

	# Piezoelectric
	Pe = np.zeros((6,3))

	# Permitivitty
	e = np.diag(np.ones(3))

	# Permeability
	mu_tensor = np.diag(np.ones(3))

	# Piezomagnetic
	Pm = np.zeros((6,3))

	# Electromagnetic
	em = np.zeros((3,3))

	# Thermo-Mechanical - 3x1
	tmech = np.zeros((6,1))
	# Thermo-Electric - 2x1
	telectric = np.zeros((3,1))
	# Thermo-Magnetic - 2x1
	tmagnetic = np.zeros((3,1))
	# Specific Heat - scalar
	spheat = 1.





	class material_args(object):
		"""docstring for material_args"""
		def __init__(self, arg):
			super(material_args, self).__init__()
			self.arg = arg
		Elastic = D 
		Piezoelectric = Pe
		Permitivitty = e 
		Piezomagnetic = Pm
		Electromagnetic = em
		Permeability = mu_tensor
		ThermoMechanical = tmech
		ThermoElectrical = telectric
		ThermoMagnetical = tmagnetic
		SpecificHeat = np.array(spheat).reshape(1,1)

	# Build H matrix
	H1 = np.concatenate((np.concatenate((material_args.Elastic,material_args.Piezoelectric),axis=1),material_args.Piezomagnetic),axis=1)
	H1 = np.concatenate((H1,material_args.ThermoMechanical),axis=1)
	H2 = np.concatenate((np.concatenate((np.concatenate((material_args.Piezoelectric.T,-material_args.Permitivitty),axis=1),material_args.Electromagnetic),axis=1),material_args.ThermoElectrical),axis=1)
	H3 = np.concatenate((np.concatenate((np.concatenate((material_args.Piezomagnetic.T,material_args.Electromagnetic.T),axis=1),-material_args.Permeability),axis=1),material_args.ThermoMagnetical),axis=1)
	H4 = np.concatenate((np.concatenate((material_args.ThermoMechanical.T,material_args.ThermoElectrical.T),axis=1),material_args.ThermoMagnetical.T),axis=1)
	H4 = np.concatenate((H4,-material_args.SpecificHeat),axis=1)
	H = np.concatenate((np.concatenate((np.concatenate((H1,H2),axis=0),H3),axis=0),H4),axis=0)

	# Add to the class
	material_args.H = H


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
				# print points
				# print 
				# t = np.array([1e-01,0.,-1.0,0,0,0]).reshape(1,6)
				t = np.array([0.,0.,0.1,0.,0.,0.]).reshape(1,6)
			# elif np.allclose(node1[1],2) and np.allclose(node3[1],2) and np.allclose(node3[1],2) and np.allclose(node4[1],2):
			# 	t = np.array([0.,1e-02,0,0,0,0]).reshape(1,6)



			# if np.allclose(node1[0],2.) and np.allclose(node1[1],2.) and np.allclose(node1[2],2.):
			# 	t = np.array([2.1,2.1,2.1,0,0,0]).reshape(1,6)
			# elif np.allclose(node2[0],2.) and np.allclose(node2[1],2.) and np.allclose(node2[2],2.):
			# 	t = np.array([2.1,2.1,2.1,0,0,0]).reshape(1,6)
			# elif np.allclose(node2[0],2.) and np.allclose(node2[1],2.) and np.allclose(node2[2],2.):
			# 	t = np.array([2.1,2.1,2.1,0,0,0]).reshape(1,6)
			# elif np.allclose(node2[0],2.) and np.allclose(node2[1],2.) and np.allclose(node2[2],2.):
			# 	t = np.array([2.1,2.1,2.1,0,0,0]).reshape(1,6)

			return t




	class AnalyticalSolution(object):
		class Args(object):
			node = 0
			points = 0

		def Get(self,Args):
			node = Args.node

			# Uz = 2
			uz = 2*node[2]

			return np.array([0,0,uz,0,0,0])

			


	return general_data, geo_args, material_args, mesh_info, BoundaryData, AnalyticalSolution 