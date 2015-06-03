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

	# Material Properties - Plane strain 
	# Elastic constitutive model
	D = 1e10*np.array([
	[11.7,8.41,0],
	[8.41,12.6,0],
	[0,0,2.3]
	])
	# Piezoelectric tensor
	P = np.array([
		[23.3,0],
		[-6.5,0],
		[0,0]
		])
	# Electrostatic tensor
	e = 1e-08*np.array([
		[1.505,0],
		[0,1.505]
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
		info = 'quad'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Problem_CooksLinear_Piezoelectric/Mesh_3.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Problem_CooksLinear_Piezoelectric/Mesh_1Elem.dat'

		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_400.dat'



	class BoundaryData(object):
		class DirichArgs(object):
			node1=0
			node2=0
			points=0

		def DirichletCriterion(self,DirichArgs=0):
			node_1 = DirichArgs.node1 
			node_2 = DirichArgs.node2 
			if np.allclose(node_1[0],0) and np.allclose(node_2[0],0):
				b = np.array([0,0,0])
			else:
				b = []
			return b
		
		def NeumannCriterion(self,node_1,node_2):
			if np.allclose(node_1[1],1) and np.allclose(node_2[1],1):
				p=np.array([0,2e04,0])
			else:
				p=np.zeros((3))

			return p

			

			


	return C, nvar, geo_args, material_args, mesh_info, BoundaryData 