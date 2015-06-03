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
	# D = np.diag(np.ones(6))
	D = np.array([
		[116, 77, 78, 0, 0, 0],
		[77, 116, 78, 0, 0, 0],
		[78, 58, 162, 0, 0, 0],
		[0, 0, 0, 89, 0, 0],
		[0, 0, 0, 0, 89, 0],
		[0, 0, 0, 0, 0, 86],
		])*1e09

	# Piezoelectric
	# Pe = np.zeros((6,3))
	Pe = np.array([
		[0,0,-4.4],
		[0,0,-4.4],
		[0,0,18.6],
		[0.0,0,0.],
		[0,11.6,0],
		[11.6,0,0]
		])

	# Permitivitty
	# e = np.diag(np.ones(3))
	e = np.array([
		[11.2,0,0.],
		[0,11.2,0.],
		[0.,0,12.6]
		])*1e-9

	# Permeability
	# mu_tensor = np.diag(np.ones(3))
	mu_tensor = np.array([
		[5.0,0,0.],
		[0,5.0,0.],
		[0.,0,10.]
		])*1e-6

	# Piezomagnetic
	# Pm = np.zeros((6,3))
	Pm = np.array([
		[0,0,5.8],
		[0,0,5.8],
		[0,0,7],
		[0.0,0,0.],
		[0,5.5,0],
		[5.5,0,0]
		])*1e2

	# Electromagnetic
	# em = np.zeros((3,3))
	em = np.array([
		[5.37,0,0.],
		[0,5.37,0.],
		[0.,0,2327.5]
		])*1e-12

	# Thermo-Mechanical - 6x1
	tmech = np.zeros((6,1))
	# tmech = (np.array([1.67, 1.67, 1.96, 0, 0, 0])*1e6).reshape(6,1)
	# Thermo-Electric - 3x1
	telectric = np.zeros((3,1))
	# telectric = (np.array([58.3,58.3,58.3])*1e-5).reshape(3,1)
	# Thermo-Magnetic - 3x1
	tmagnetic = np.zeros((3,1))
	# tmagnetic = (np.array([5.,5.,5.])*1e-2).reshape(3,1)
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
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/MultiPhysics_Fibre_3D/Mesh_Fibre_1000.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/MultiPhysics_Fibre_3D/Mesh_Fibre_3375.dat'
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
			Applied_at = 'node'
				

		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node 
			mesh_points = DirichArgs.points 
			# path_potential = DirichArgs.path_potential
			# path_displacement = DirichArgs.path_displacement
			# displacements = np.loadtxt(path_displacement)
			# potentials = np.loadtxt(path_potential)
			# points = DirichArgs.points



			if np.allclose(node[0],0.0):
				b = np.array([0.,0.,0.,0.,0.,0.])
			elif np.allclose(node[0],0.1):
				b = np.array([0.005,0.01,0.015,0.,10.,0.])
			else:
				b  = [[[],[],[],[],[],0.]]


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

			


	return general_data, geo_args, material_args, mesh_info, BoundaryData, AnalyticalSolution 