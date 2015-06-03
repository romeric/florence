import numpy as np 


def ProblemData(general_data):

	# Degree of continuity 
	# C = 1
	C = general_data.C
	# nvar is the the dimension of vectorial field we are solving for.
		# for instance in continuum 2d problems nvar is 2 since we solve for ux and uy
		# for 3d beam problems nvar is 6 since solve for ux, uy, uz, tx, ty and tz
	nvar = 2

	# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		ndiv = 50.

	# Material Properties
	E = 3.*1e7
	nu = 0.3
	# Plane strain
	# D = E/(1+nu)/(1-2*nu)*np.array([
	# 	[1-nu,nu,0],
	# 	[nu,1-nu,0],
	# 	[0,0,(1-2*nu)/2.0]
	# 	])
	
	# Plane stress 
	D = E/(1-nu**2)*np.array([
	[1.0,nu,0],
	[nu,1.0,0],
	[0,0,(1.-nu)/2.0]
	])


	class mesh_info(object):
		info = 'quad'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Problem_CooksLinear/Mesh_2.dat'



	# Boundary conditions - applied at mesh edges
	dir_boundary_data = np.array([
		[0.,0.,0.],
		[2.,0.,-1.]
		])

	neu_boundary_data = np.zeros((1,3))

	class boundary_data(object):
		Dirichlet = dir_boundary_data
		Neumann = neu_boundary_data
			





	return C, nvar, geo_args, D, mesh_info, boundary_data 