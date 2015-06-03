import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import rc
from Core.MeshGeneration.MeshGeneration import MeshGeneration


#############################################################################################
# Problem 1 - Rectangular Geometry

# Build a rectangular geometry
def Geometry(lx,ly,nx,ny):
	x = np.linspace(0,lx,nx)
	y = np.linspace(0,ly,ny)
	# X,Y = np.meshgrid(x,y)

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

	return boundary



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


def ProblemData_BEM(C,nx,ny):

	class geo_args(object):
		lx = 2.
		ly = 1.
		ndiv_x = nx
		ndiv_y = ny
		Lagrange_Multipliers = 'deactivated'
		corners = 0

	boundary = Geometry(geo_args.lx,geo_args.ly,geo_args.ndiv_x,geo_args.ndiv_y)

	# Build element connectivity
	elements = np.zeros((len(boundary+1),2))
	elements[0:,0] = np.linspace(0,len(elements)-1,len(elements))
	elements[0:,1] = elements[0:,0]+1
	elements[-1,1] = 0

	# Element connectivity for higher order basis
	element_connectivity = np.zeros((len(boundary+1),C+2),dtype=int)
	for i in range(0,element_connectivity.shape[0]):
		element_connectivity[i,0:]=np.linspace((C+1)*i,(C+1)*(i)+(C+1),element_connectivity.shape[1])
	element_connectivity[-1,-1]=0


	mesh = Meshing(geo_args.lx,geo_args.ly)
	internal_points = np.array(mesh.points)

	return C, elements, boundary, element_connectivity, internal_points, mesh, geo_args


def BoundaryConditions(nodal_coordinates,C=0,geo_args=0):

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

	# Allocate - first column of boundary_data is Dirichlet and second column Neumann
	boundary_data = np.zeros((nodal_coordinates.shape[0],2))
	for i in range(0,nodal_coordinates.shape[0]):
		# -1 denotes unknown variables
		if nodal_coordinates[i,0]==0 or np.allclose([nodal_coordinates[i,0]],[0],atol=1e-15):
			# Applied Dirichlet T1=5
			boundary_data[i,0]=5
			boundary_data[i,1]=-1
		elif nodal_coordinates[i,0]==np.max(nodal_coordinates[0:,0]) or np.allclose([nodal_coordinates[i,0]],np.max(nodal_coordinates[0:,0]),atol=1e-15):
			boundary_data[i,0]=3
			boundary_data[i,1]=-1
		elif nodal_coordinates[i,1]==0 or np.allclose([nodal_coordinates[i,1]],[0],atol=1e-15) or nodal_coordinates[i,1]==np.max(nodal_coordinates[0:,1]) or np.allclose(nodal_coordinates[i,1],[np.max(nodal_coordinates[0:,1])],atol=1e-15):
			boundary_data[i,0]=-1
			pass # zero flux
	return boundary_data




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
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_1_Potential_C_0.eps', format='eps', dpi=1000)


	elif opt==1:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$X-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_1_XFlux_C_0.eps', format='eps', dpi=1000)

	elif opt==2:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux2.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$Y-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_1_YFlux_C_0.eps', format='eps', dpi=1000)

	

	# Save before you display images
	plt.show()




