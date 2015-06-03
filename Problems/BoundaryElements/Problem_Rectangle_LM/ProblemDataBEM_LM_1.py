import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from Core.MeshGeneration.MeshGeneration import MeshGeneration

#############################################################################################
# Problem 1 - Rectangular Geometry

# Build a rectangular geometry
def Geometry(lx,ly,nx,ny):
	x = np.linspace(0,lx,nx)
	y = np.linspace(0,ly,ny)

	# The first two columns are (x,y) coordinates of boundary nodes and the third column is information about corner nodes
	boundary = np.zeros((4+2*(nx-2)+2*(ny-2),3))

	for j in range (0,4):
		for i in range(0,nx):
			if j==0:
				boundary[i,0] = x[i]
				boundary[i,1] = y[0]
				if i==nx-1 or i==0:
					boundary[i,2] = 1
			elif j==1:
				if i==ny-1:
					break
				else:
					boundary[i+j*ny,0]=x[-1]
					boundary[i+j*ny,1]=y[i+1]
				if i==ny-2:
					boundary[i+j*ny,2] = 1
			elif j==2:
				if i==0:
					continue
				else:
					boundary[i+j*nx-2,0] = x[nx-i-1]
					boundary[i+j*nx-2,1] = y[-1]
				if i==nx-1:
					boundary[i+j*nx-2,2] = 1
			elif j==3:
				if i==0:
					continue
				elif i==ny-1:
					break
				else:
					boundary[i+j*ny-3,0] = x[0]
					boundary[i+j*ny-3,1] = y[ny-i-1]


	counter =0
	for i in range(0,boundary.shape[0]):
		if boundary[i,2]==1:
			counter+=1

	modified_boundary = np.zeros((boundary.shape[0]+counter,3))
	count = 0
	for i in range(0,boundary.shape[0]):
		modified_boundary[i+count,0:]=boundary[i,0:]
		if boundary[i,2]==1:
			count+=1
			modified_boundary[i+count,0:]=boundary[i,0:]

	firstrow = np.zeros((1,3))
	firstrow[0,0:] = modified_boundary[0,0:]
	modified_boundary = modified_boundary[1:,0:]
	modified_boundary = np.concatenate((modified_boundary,firstrow),axis=0)

	return boundary, modified_boundary



# Meshing for plotting purposes
def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def Meshing(lx,ly):
	# Define Margins
	# Mx = 0.06 
	# My = 0.06
	Mx = 0.2 
	My = 0.2
	points = [(Mx, My), (lx-Mx, My), (lx-Mx,ly-My), (Mx, ly-My)]
	facets = round_trip_connect(0, len(points)-1)
	mesh = MeshGeneration().GenerateMesh(points,facets,max_vol=0.005,gen_faces=False)  

	return mesh


def DiscontinuousGlobalCoord(global_coord,C,geo_args=0):
	
	nelemp1 = geo_args.ndiv_x-1
	nelemp2 = geo_args.ndiv_y-1

	global_coord_discontinous = np.zeros((global_coord.shape[0],3))
	for i in range(0,global_coord.shape[0]):
		if i<=nelemp1*(C+1):
			global_coord_discontinous[i,0:2] = global_coord[i,0:]
		elif i>nelemp1*(C+1) and i<=nelemp1*(C+1)+nelemp2*(C+1):
			global_coord_discontinous[i+1,0:2] = global_coord[i,0:]
			if i==nelemp1*(C+1)+1:
				global_coord_discontinous[i,0:2] = global_coord[i-1,0:]
				global_coord_discontinous[i-1,2] = 2
				global_coord_discontinous[i,2] = -2
		elif i>nelemp1*(C+1)+nelemp2*(C+1) and i<=2*nelemp1*(C+1)+nelemp2*(C+1):
			global_coord_discontinous[i+2,0:2] = global_coord[i,0:]
			if i==nelemp1*(C+1)+nelemp2*(C+1)+1:
				global_coord_discontinous[i+1,0:2] = global_coord[i-1,0:]
				global_coord_discontinous[i,2] = 3
				global_coord_discontinous[i+1,2] = -3
		elif i>2*nelemp1*(C+1)+nelemp2*(C+1) and i<=2*nelemp1*(C+1)+2*nelemp2*(C+1):
			global_coord_discontinous[i+3,0:2] = global_coord[i,0:]
			if i==2*nelemp1*(C+1)+nelemp2*(C+1)+1:
				global_coord_discontinous[i+2,0:2] = global_coord[i-1,0:]
				global_coord_discontinous[i+1,2] = 4
				global_coord_discontinous[i+2,2] = -4
	 # Cheap trick
	global_coord_discontinous[0,2] = 1
	global_coord_discontinous[-1,2] = -1


	global_coord = np.copy(global_coord_discontinous)

	return global_coord


# def ProblemData_BEM():
def ProblemData_BEM(C,nx,ny):

	# Polynomial Degree
	# C=0

	# Geometry
	lx = 2
	ly = 1
	# Descritised geometry
	# nx = 5
	# ny = 5

		# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		# ndiv_curve = 5.
		# ndiv_straight = 5.
		# ndiv_curve = nx
		# ndiv_straight = nx/2.
		ndiv_x = nx
		ndiv_y = ny
		Lagrange_Multipliers = 'activated'
		corners = 4

	boundary, modified_boundary = Geometry(lx,ly,nx,ny)

	# Build element connectivity
	elements = np.zeros((len(boundary+1),2))
	elements[0:,0] = np.linspace(0,len(elements)-1,len(elements))
	elements[0:,1] = elements[0:,0]+1
	# elements[-1,1] = 0

	for i in range(0,elements.shape[0]):
		if i>=nx-1 and i<nx-1+ny-1:
			elements[i,0:]+=1
		elif i>=nx-1+ny-1 and i<2*(nx-1)+ny-1:
			elements[i,0:]+=2
		elif i>=2*(nx-1)+ny-1:
			elements[i,0:]+=3

	# for each line element compute J, nx and ny
	boundary_points = boundary
	boundary_elements = elements

	# Element connectivity for higher order basis
	element_connectivity = np.zeros((len(boundary+1),C+2),dtype=int)
	# element_connectivity[0:,0]=np.linspace(0,(C+1)*(element_connectivity.shape[0]-1),element_connectivity.shape[0])
	for i in range(0,element_connectivity.shape[0]):
		element_connectivity[i,0:]=np.linspace((C+1)*i,(C+1)*(i)+(C+1),element_connectivity.shape[1])
	# element_connectivity[-1,-1]=0


	for i in range(0,element_connectivity.shape[0]):
		if i>=nx-1 and i<nx-1+ny-1:
			element_connectivity[i,0:]+=1
		elif i>=nx-1+ny-1 and i<2*(nx-1)+ny-1:
			element_connectivity[i,0:]+=2
		elif i>=2*(nx-1)+ny-1:
			element_connectivity[i,0:]+=3


	mesh = Meshing(lx,ly)
	internal_points = np.array(mesh.points)

	# nelemp = np.intc(boundary_elements.shape[0]/4) # in case needed


	return C, elements, modified_boundary, element_connectivity, internal_points, mesh, geo_args


def BoundaryConditions(nodal_coordinates,C,geo_args):

							# Zero Flux 

			#############################################
			#											#
			#											#
	# T1	#					Area					#	T2
			#											#
			#											# L
			#                    2L                   	#
			#############################################

							# Zero Flux 


	# Modify boundary data for LM formulation
	nelemp1 = geo_args.ndiv_x-1
	nelemp2 = geo_args.ndiv_y-1

	# Allocate
	boundary_data = np.zeros((nodal_coordinates.shape[0],2))+4   # 4 is for 4 corners change according to the problem

	for i in range(0,nodal_coordinates.shape[0]):
		if i<=nelemp1*(C+1):
			boundary_data[i,0] = -1
			boundary_data[i,1] = 0
		elif i>nelemp1*(C+1) and i<=nelemp1*(C+1)+nelemp2*(C+1)+1:
			boundary_data[i,0] = 3
			boundary_data[i,1] = -1
		elif i>nelemp1*(C+1)+nelemp2*(C+1)+1 and i<=2*nelemp1*(C+1)+nelemp2*(C+1)+2:
			boundary_data[i,0] = -1
			boundary_data[i,1] = 0

		elif i>2*nelemp1*(C+1)+nelemp2*(C+1)+2 and i<=2*nelemp1*(C+1)+2*nelemp2*(C+1)+3:
			boundary_data[i,0] = 5
			boundary_data[i,1] = -1

	mod_boundary_data = np.zeros((boundary_data.shape[0]+4,2))
	mod_boundary_data[0:-4,0:] = boundary_data
	# Lagrange multipliers are unknown
	mod_boundary_data[-4:,0] = -1

	return mod_boundary_data




def ComputeErrorNorms(global_coord,total_sol,opt=0,internal_points=0,POT=0):
	# If opt=0, error at boundary is calculated
	# If opt=1, error at interior is calculated

	rel_err = 0

	# Computing error at boundary
	if opt==0:
		analytic = -global_coord[:,0]+5
		rel_err = la.norm(analytic-total_sol[:,0])/la.norm(analytic)
	# Computing error at internal points
	elif opt==1:
		analytic = -internal_points[:,0]+5
		rel_err = la.norm(POT[:,0]-analytic)/la.norm(analytic)

	return rel_err



def PlotFunc(mesh,POT,Flux1,Flux2,opt=0):

	mesh_points = np.array(mesh.points)
	mesh_tris = np.array(mesh.elements)

	fig = plt.figure()
	ax=fig.gca()
	if opt==0:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, POT.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$Potential$ (V)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_3_Potential_C_0.eps', format='eps', dpi=1000)


	elif opt==1:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$X-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_3_XFlux_C_0.eps', format='eps', dpi=1000)

	elif opt==2:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux2.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$Y-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_3_YFlux_C_0.eps', format='eps', dpi=1000)

	

	# Save before you display images
	plt.show()