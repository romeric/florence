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

	# Material Properties - Plane strain
	# Elastic constitutive model

	mu = 1.5
	lam = 3.0


	E = mu*(3.0*lam+2.0*mu)/(mu+lam)
	nu = lam/2.0/(mu+lam)

	D = E/(1.0+nu)*(1.0-2.0*nu)*np.array([
		[1-nu,nu,0],
		[nu,1-nu,0],
		[0,0,(1.0-2.0*nu)/2.0]
		])

	Pe = np.zeros((3,2))

	e = np.array([
		[2.0,0.0],
		[0.0,2.0]
		])

	# Permeability
	mu_tensor = np.array([
		[5.0,0.0],
		[0.0,5.0]
		])

	# Piezomagnetic
	Pm = np.zeros((3,2))
	# Electromagnetic Tensor
	em = np.zeros((2,2))
	# Thermo-Mechanical - 3x1
	tmech = np.zeros((3,1))
	# Thermo-Electric - 2x1
	telectric = np.zeros((2,1))
	# Thermo-Magnetic - 2x1
	tmagnetic = np.zeros((2,1))
	# Specific Heat - scalar
	spheat = 1.


	class material_args(object):
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
	# print H[:,:-1]







	# Geometry Correction for curved meshes
	class GeometryRepresentation(object):
		"""Correct geometry using blending functions, Parameterise the geometry"""

		X1=0; X2=0; X3=0; X4=0
		Y1=0; Y2=0; Y3=0; Y4=0
		eta = 0
		zeta = 0
		xycoord = np.zeros((4,2))
		nlayer=0

		def Blending(self,elem):
			# IMPORTANT
			# The blending is not only dependent on the geometry but the way it has been meshed.
			# Check the meshing closely before assigning blending functions.

			# We aim to modify not only the surface for this problem but the entire geometry
			# since quad mesh is uniform in the hollow arc this is feasible.
			r = np.sqrt(self.xycoord[:,0]**2+self.xycoord[:,1]**2)
			ro = np.max(r); ri = np.min(r)

			# Blending
			for i in range(0,self.nlayer):
				if elem%self.nlayer==i:
					xedge = (ro*np.cos(np.pi/(self.nlayer*4.0)*(self.eta+2*i+1)) - (1+self.eta)/2.0*self.X4-(1-self.eta)/2.0*self.X1)*(1-self.zeta)/2.0
					xedge += (ri*np.cos(np.pi/(self.nlayer*4.0)*(self.eta+2*i+1)) - (1+self.eta)/2.0*self.X3-(1-self.eta)/2.0*self.X2)*(1+self.zeta)/2.0
					yedge = (ro*np.sin(np.pi/(self.nlayer*4.0)*(self.eta+2*i+1)) - (1+self.eta)/2.0*self.Y4-(1-self.eta)/2.0*self.Y1)*(1-self.zeta)/2.0
					yedge += (ri*np.sin(np.pi/(self.nlayer*4.0)*(self.eta+2*i+1)) - (1+self.eta)/2.0*self.Y3-(1-self.eta)/2.0*self.Y2)*(1+self.zeta)/2.0

			return xedge, yedge



	class mesh_info(object):
		kind = 'quad'
		mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_25.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_100.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_400.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_900.dat'
		# mesh_filename = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Benchmark/Mesh_Hollow_Arc_1600.dat'

		nature = 'curved'
		# nature = 'straight'
		GeoCorrection = GeometryRepresentation




	class BoundaryData(object):
		class DirichArgs(object):
			node1=0
			node2=0
			points=0
			path_potential = '/home/roman/Dropbox/Python/potentials_arc_allmesh_electro_nmesh.txt'
			path_displacement = '/home/roman/Dropbox/Python/displacements_arc_allmesh_mech_nmesh.txt'

			node = 0
			Applied_at = 'node' 	# This argument explains if the boundary conditions are to be applied at meshed or initial geometry
									# In case of initial geometry this would be edges i.e. Applied_at = 'edge'

		class NeuArgs(object):
			node1=np.zeros(2)
			node2=np.zeros(2)
			points=0
			node = 0
			p_i = 0.5
			p_o = 1.0
			Applied_at = 'node'



		def DirichletCriterion(self,DirichArgs):
			node = DirichArgs.node
			mesh_points = DirichArgs.points
			path_potential = DirichArgs.path_potential
			path_displacement = DirichArgs.path_displacement
			displacements = np.loadtxt(path_displacement)
			potentials = np.loadtxt(path_potential)
			points = DirichArgs.points



			#########################################################
			# # Mechanical
			# if np.allclose(np.sqrt(node[0]**2+node[1]**2),1.0):
			# # if np.allclose(np.sqrt(node[0]**2+node[1]**2),1.0,atol=1e-2):
			# 	theta = np.arctan(node[1]/node[0])

			# 	aa=0.5
			# 	b = np.array([aa*np.cos(theta),aa*np.sin(theta),0])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2.0):
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2.0,atol=1e-2):
			# 	theta = np.arctan(node[1]/node[0])
			# 	bb=3.0
			# 	b = np.array([bb*np.cos(theta),bb*np.sin(theta),0])
			# elif np.allclose(node[0],0):
			# 	x=np.where(points[:,1]==node[1])[0]
			# 	if x.shape[0]!=0:
			# 		theta = np.arctan(node[1]/node[0])
			# 		ux = displacements[x[0]]*np.cos(theta)
			# 		uy = displacements[x[0]]*np.sin(theta)
			# 		b = np.array([ux,uy,0])

			# elif np.allclose(node[1],0):
			# 	x=np.where(points[:,0]==node[0])[0]
			# 	if x.shape[0]!=0:
			# 		theta = np.arctan(node[1]/node[0])
			# 		ux = displacements[x[0]]*np.cos(theta)
			# 		uy = displacements[x[0]]*np.sin(theta)
			# 		b = np.array([ux,uy,0])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1.0,atol=1e-2) or np.allclose(np.sqrt(node[0]**2+node[1]**2),2.0,atol=1e-2):
			# 	x=np.where(points[:,0]==node[0])[0]
			# 	if x.shape[0]!=0:
			# 		theta = np.arctan(node[1]/node[0])
			# 		ux = displacements[x[0]]*np.cos(theta)
			# 		uy = displacements[x[0]]*np.sin(theta)
			# 		b = np.array([ux,uy,0])
			# 		# print np.sqrt(node[0]**2+node[1]**2), np.sqrt(ux**2+uy**2)


			# else:
			# 	# b=[]
			# 	b = np.array([[],[],0])
			# 	# print b
			# 	# print np.sqrt(node[0]**2+node[1]**2)


			# return b


			#######################################################################


			# # Electric
			# if np.allclose(np.sqrt(node[0]**2+node[1]**2),1.0):
			# 	theta = np.arctan(node[1]/node[0])

			# 	b = np.array([0,0,5])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2.0):
			# 	theta = np.arctan(node[1]/node[0])
			# 	bb=3.0
			# 	b = np.array([bb*np.cos(theta),bb*np.sin(theta),0])
			# elif np.allclose(node[0],0):
			# 	x=np.where(points[:,1]==node[1])[0]
			# 	if x.shape[0]!=0:
			# 		b = np.array([0,0,potentials[x[0]]])

			# elif np.allclose(node[1],0):
			# 	x=np.where(points[:,0]==node[0])[0]
			# 	if x.shape[0]!=0:
			# 		b = np.array([0,0,potentials[x[0]]])

			# else:
			# 	# b=[]
			# 	# b = np.array([0,0,[]])
			# 	b = np.array([[[],0,0]]); 	b = np.fliplr(b); b = b.reshape(b.shape[1])
			# 	# print b


			# return b



			# #########################################################
			# # Coupled
			# if np.allclose(node[0],0):
				# x=np.where(points[:,1]==node[1])[0]
				# if x.shape[0]!=0:

				# 	theta = np.arctan(node[1]/node[0])

				# 	ux = displacements[x[0]]*np.cos(theta)    	# Note that at this edge it gives ux=0 which should be the case
				# 	uy = displacements[x[0]]*np.sin(theta)
				# 	phi = potentials[x[0]]
				# 	# ux=0; uy=0;

				# 	b = np.array([ux,uy,phi])

			# elif np.allclose(node[1],0):
				# x=np.where(points[:,0]==node[0])[0]
				# if x.shape[0]!=0:

				# 	theta = np.arctan(node[1]/node[0])

				# 	ux = displacements[x[0]]*np.cos(theta)
				# 	uy = displacements[x[0]]*np.sin(theta) 		# Note that at this edge it gives uy=0 which should be the case
				# 	phi = potentials[x[0]]
				# 	# ux=0; uy=0;

				# 	b = np.array([ux,uy,phi])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1):
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1,atol=1e-1):
			# 	theta = np.arctan(node[1]/node[0])
			# 	aa=0.5; mm=5.
			# 	b = np.array([aa*np.cos(theta),aa*np.sin(theta),mm])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2):
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2,atol=1e-1):
			# 	theta = np.arctan(node[1]/node[0])
			# 	bb=3.0; nn=10.
			# 	b = np.array([bb*np.cos(theta),bb*np.sin(theta),nn])

			# # C==1
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1.97537):
			# # 	theta = np.arctan(node[1]/node[0])
			# # 	bb=3.0; nn=10.
			# # 	b = np.array([bb*np.cos(theta),bb*np.sin(theta),nn])
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),0.987688):
			# # 	theta = np.arctan(node[1]/node[0])
			# # 	aa=0.5; mm=5.
			# # 	b = np.array([aa*np.cos(theta),aa*np.sin(theta),mm])
			# # C==2
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1.98032):
			# # 	theta = np.arctan(node[1]/node[0])
			# # 	bb=3.0; nn=10.
			# # 	b = np.array([bb*np.cos(theta),bb*np.sin(theta),nn])
			# # elif np.allclose(np.sqrt(node[0]**2+node[1]**2),0.99016):
			# # 	theta = np.arctan(node[1]/node[0])
			# # 	aa=0.5; mm=5.
			# # 	b = np.array([aa*np.cos(theta),aa*np.sin(theta),mm])


			# else:
			# 	b=[]
			# 	# print b
			# 	# print np.sqrt(node[0]**2+node[1]**2)


			# return b
			#########################################################



			# #########################################################
			# # Coupled - analytical solution not applied
			if np.allclose(np.sqrt(node[0]**2+node[1]**2),1.0):
				theta = np.arctan(node[1]/node[0])

				aa=0.5; mm=5.
				b = np.array([aa*np.cos(theta),aa*np.sin(theta),mm,0.,0.])

			elif np.allclose(np.sqrt(node[0]**2+node[1]**2),2.0):
				theta = np.arctan(node[1]/node[0])
				bb=3.0; nn=10.
				b = np.array([bb*np.cos(theta),bb*np.sin(theta),nn,0.,0.])
				# print np.sqrt(node[0]**2+node[1]**2)
			# X=0
			elif np.allclose(node[0],0):
				# b = [0,[],0,0,0]
				b = np.array([[[],[],0]]); 	b = np.fliplr(b); b = b.reshape(b.shape[1])
				b = np.append(b,0.); b = np.append(b,0.)
			# Y=0
			elif np.allclose(node[1],0):
				b = np.array([[],0.,[],0.,0.])
				# b = [[],0,[],0,0]

			else:
				# b=[]
				b = np.array([[],[],[],0,0])

			return b


			# #######################################################################

			# # Coupled # analytical solution applied everywhere
			# if np.allclose(node[0],0):
			# 	x=np.where(points[:,1]==node[1])[0]
			# 	if x.shape[0]!=0:

			# 		theta = np.arctan(node[1]/node[0])

			# 		ux = displacements[x[0]]*np.cos(theta)    	# Note that at this edge it gives ux=0 which should be the case
			# 		uy = displacements[x[0]]*np.sin(theta)
			# 		phi = potentials[x[0]]
			# 		# ux=0; uy=0;

			# 		b = np.array([ux,uy,phi])

			# elif np.allclose(node[1],0):
			# 	x=np.where(points[:,0]==node[0])[0]
			# 	if x.shape[0]!=0:

			# 		theta = np.arctan(node[1]/node[0])

			# 		ux = displacements[x[0]]*np.cos(theta)
			# 		uy = displacements[x[0]]*np.sin(theta) 		# Note that at this edge it gives uy=0 which should be the case
			# 		phi = potentials[x[0]]
			# 		# ux=0; uy=0;

			# 		b = np.array([ux,uy,phi])

			# elif np.allclose(np.sqrt(node[0]**2+node[1]**2),1,atol=1e-1) or np.allclose(np.sqrt(node[0]**2+node[1]**2),2,atol=1e-1):

			# 	x=np.where(points[:,0]==node[0])[0]
			# 	if x.shape[0]!=0:

			# 		theta = np.arctan(node[1]/node[0])

			# 		ux = displacements[x[0]]*np.cos(theta)
			# 		uy = displacements[x[0]]*np.sin(theta)
			# 		phi = potentials[x[0]]
			# 		print np.sqrt(node[0]**2+node[1]**2), phi

			# 		b = np.array([ux,uy,phi])

			# else:
			# 	b=[]
			# 	# print b
			# 	print np.sqrt(node[0]**2+node[1]**2)


			# return b



		def NeumannCriterion(self,NeuArgs):
			node_1 = NeuArgs.node1
			node_2 = NeuArgs.node2
			# # print node_1, node_2
			# r = np.sqrt((node_2[0]-node_1[0])**2 + (node_2[1]-node_1[1])**2)
			# r1 = np.sqrt((node_1[0])**2 + (node_1[1])**2)
			# r2 = np.sqrt((node_2[0])**2 + (node_2[1])**2)
			# # print r1,r2
			# if np.allclose(r1,1) and np.allclose(r2,1):
			# 	p=np.array([0,-5e09,0])
			# elif np.allclose(r1,2) and np.allclose(r2,2):
			# 	p=np.array([0,5e09,0])
			# else:
			# 	p=np.zeros((3))
			# 	# print p

			# This is applied for hollow arc benchmark problem
			r1 = np.sqrt((node_1[0])**2 + (node_1[1])**2)
			r2 = np.sqrt((node_2[0])**2 + (node_2[1])**2)
			if np.allclose(r1,1) and np.allclose(r2,1):
				p=np.array([-5e+1,-5e+1,0])
			elif np.allclose(r1,2) and np.allclose(r2,2):
				p=np.array([-10e+1,-10e+1,0])
			else:
				p=np.zeros((3))
			return p


			# # This is applied for hollow arc benchmark problem - at mesh
			# node = NeuArgs.node
			# p_i = NeuArgs.p_i
			# p_o = NeuArgs.p_o
			# r = np.sqrt((node[0])**2 + (node[1])**2)
			# # Compute the angle
			# theta = np.arctan(node[1]/node[0])
			# if np.allclose(r,1):
			# 	p=np.array([p_i*np.cos(theta),p_i*np.sin(theta),0])
			# 	# p=np.array([0,p_i*np.sin(theta),0])
			# elif np.allclose(r,2):
			# 	p=np.array([p_o*np.cos(theta),p_o*np.sin(theta),0])
			# 	# p=np.array([0,p_o*np.sin(theta),0])
			# else:
			# 	p=np.zeros((3))


			# p=np.zeros((3))

			# return p





	class AnalyticalSolution(object):
		class Args(object):
			node1 = 0
			node2 = 0
			node = 0
			points = 0

		def Get(self,Args):
			node = Args.node

			r = np.sqrt(node[0]**2+node[1]**2)
			theta = np.arctan(node[1]/node[0])

			b=3.0; a=0.5
			# ur_analytic = 1.0/7.0*((4*b-a)*r+(8*a-4*b)/r**2)
			ur_analytic = (b-a)*r+(2*a-b)/r
			ux = ur_analytic*np.cos(theta); uy = ur_analytic*np.sin(theta)

			m = 5; n=10
			phi_analytic = ((n-m)/np.log(2))*np.log(r)+m


			return np.array([ux,uy,phi_analytic])









	return general_data, geo_args, material_args, mesh_info, BoundaryData, AnalyticalSolution