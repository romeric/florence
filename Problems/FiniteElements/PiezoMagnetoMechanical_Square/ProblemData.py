import numpy as np 


def ProblemData(general_data):

	# Degree of continuity 
	# C = 1
	C = general_data.C
	# nvar is the the dimension of vectorial field we are solving for.
		# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
		# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz
	general_data.nvar = 5
	general_data.ndim = 2

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
	D = 1e+09*np.array([
    	[116.,77.,0.], 
    	[77.,116,0.],
    	[0.,0.,86.]
    	])

	# Piezoelectric
	Pe = np.zeros((3,2))
	# Pe = np.array([
	# 	[0.,0.],
	# 	[0.,0.],
	# 	[0.,11.6]
	# 	])

	# Permitivitty
	e = 1e-9*np.array([
		[11.2,0.0],
		[0.0,11.2]
		])

	# Permeability
	mu_tensor = 1e-6*np.array([
		[5.0,0.0],
		[0.0,5.0]
		])

	# Piezomagnetic
	Pm = np.zeros((3,2))
	# Pm = 100*np.array([
	# 	[0.,0.],
	# 	[0.,0.],
	# 	[0.,5.5]
	# 	])

	# Electromagnetic
	# em = np.zeros((2,2))
	em = 1e-12*np.array([[5.37,0.],[0,5.37]])

	# Thermo-Mechanical - 3x1
	tmech = np.zeros((3,1))
	# tmech = 1e06*np.array([1.67,1.67,0.]).reshape(3,1)
	# Thermo-Electric - 2x1
	telectric = np.zeros((2,1))
	# telectric = 1e-5*np.array([[58.3,58.3]]).reshape(2,1)
	# Thermo-Magnetic - 2x1
	tmagnetic = np.zeros((2,1))
	# tmagnetic = np.array([[0.005,0.005]]).reshape(2,1)
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
	# H1 = np.concatenate((np.concatenate((np.concatenate((material_args.Elastic,material_args.Piezoelectric),axis=1),material_args.Piezomagnetic),axis=1)),material_args.ThermoMechanical),axis=1)
	H1 = np.concatenate((np.concatenate((material_args.Elastic,material_args.Piezoelectric),axis=1),material_args.Piezomagnetic),axis=1)
	H1 = np.concatenate((H1,material_args.ThermoMechanical),axis=1)
	H2 = np.concatenate((np.concatenate((np.concatenate((material_args.Piezoelectric.T,-material_args.Permitivitty),axis=1),material_args.Electromagnetic),axis=1),material_args.ThermoElectrical),axis=1)
	H3 = np.concatenate((np.concatenate((np.concatenate((material_args.Piezomagnetic.T,material_args.Electromagnetic.T),axis=1),-material_args.Permeability),axis=1),material_args.ThermoMagnetical),axis=1)
	# H4 = np.concatenate((np.concatenate((np.concatenate((material_args.ThermoMechanical.T,-material_args.ThermoElectrical.T),axis=1),-material_args.ThermoMagnetical.T),axis=1),material_args.SpecificHeat),axis=1)
	H4 = np.concatenate((np.concatenate((material_args.ThermoMechanical.T,material_args.ThermoElectrical.T),axis=1),material_args.ThermoMagnetical.T),axis=1)
	H4 = np.concatenate((H4,-material_args.SpecificHeat),axis=1)
	H = np.concatenate((np.concatenate((np.concatenate((H1,H2),axis=0),H3),axis=0),H4),axis=0)

	# Add to the class
	material_args.H = H


	class mesh_info(object):
		kind = 'quad'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/PiezoMagnetoMechanical_Square/Mesh_Square.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/PiezoMagnetoMechanical_Square/Mesh_Square_256.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/PiezoMagnetoMechanical_Square/Mesh_Square_100.dat'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/PiezoMagnetoMechanical_Square/Mesh_Square_625.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/PiezoMagnetoMechanical_Square/Mesh_Square_2500.dat'
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

			# # Mech uy
			# if np.allclose(node[1],0):
			# 	b = np.array([0.,1.5,0.,0.])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.,4.,0.,0.])
			# else:
			# 	b  = np.array([[],[],0.,0.])
			# 	b[0] = 0.0 

			# # Mech ux and uy
			# if np.allclose(node[0],0):
			# 	b = np.array([-1.0,1.5,0.0])
			# elif np.allclose(node[0],2.0):
			# 	b = np.array([-2.0,3.0,0])
			# else:
			# 	b  = np.array([[],[],0])

			# Magneto
			# if np.allclose(node[1],0):
			# 	b = np.array([0.,0.,0.,1.,0.])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.,0.,0.,4.,0.])
			# else:
			# 	# b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	b  = np.array([[[],[],0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1]); b[4]=0

			# return b



			# Magneto-Thermal
			# if np.allclose(node[1],0):
			# 	b = np.array([0.,0.,0.,0.,2.])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.,0.,0.,0.,10.])
			# else:
			# 	b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])

			# return b

			# # Magneto-Thermal
			# if np.allclose(node[1],0):
			# 	b = np.array([0.,0.,0.,1.,1.])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.,0.,0.,4.,10.])
			# else:
			# 	# b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	b  = np.array([[[],[],0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])#; b[4]=0

			# return b

			# THIS for square
			# if np.allclose(node[1],0) or np.allclose(node[0],0.0) or np.allclose(node[0],2.0):
			# 	b = np.array([0.,0.,0.,0.,0.])
			# elif np.allclose(node[1],2.0):
			# 	b = np.array([0.,0.,0.,10.,0.])
			# else:
			# 	# b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	b  = np.array([[[],[],[],0,0]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	# b=[]


			# return b


			# if np.allclose(node[1],0) or np.allclose(node[0],0.0) or np.allclose(node[0],3.5):
			# 	b = np.array([0.,0.,0.,0.,0.])
			# elif np.allclose(node[1],1.2):
			# 	b = np.array([0.,0.,0.,10.,0.])
			# else:
			# 	# b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	# b  = np.array([[[],[],[],0,[]]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	# b  = np.array([[0,0,0,[],0]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
			# 	# b=[]
			# 	b  = [[0.,0.,0.,[],0.]]


			if np.allclose(node[1],0.0) or np.allclose(node[0],0.0) or np.allclose(node[0],3.5*0.0001):
				b = np.array([0.,0.,0.,0.,0.])
			elif np.allclose(node[1],1.2*0.0001):
				b = np.array([0.,0.,0.,10.,0.])
			else:
				# b  = np.array([[[],0.,0.,0.,0.]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
				# b  = np.array([[[],[],[],0,[]]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
				# b  = np.array([[0,0,0,[],0]]); b = np.fliplr(b); b= b.reshape(b.shape[1])
				# b=[]
				b  = [[0.,0.,0.,[],0.]]


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