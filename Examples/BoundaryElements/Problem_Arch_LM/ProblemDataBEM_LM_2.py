import numpy as np
import numpy.linalg as la
# from Core.MeshGeneration.MeshGeneration import MeshGeneration
# from Core.MeshGeneration.Mesh import MeshPyTri as MeshGeneration
from Core.MeshGeneration.Mesh import*

#############################################################################################
# Problem 2 - Arch Geometry 


def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

# Build a arch geometry
def Geometry(geo_args):

	rin = geo_args.rin
	rout = geo_args.rout
	ndiv = geo_args.ndiv_curve
	ndiv_straight = geo_args.ndiv_straight


	# Geometry for meshing
	dum = np.linspace(0,np.pi/2,ndiv)
	points1 = np.zeros((ndiv,2))
	points3 = np.zeros((ndiv,2))
	for i in range(0,dum.shape[0]):
		points1[i,0] = rin*np.sin(dum[i]) 
		points1[i,1] = rin*np.cos(dum[i]) 

		points3[i,0] = rout*np.sin(dum[i]) 
		points3[i,1] = rout*np.cos(dum[i]) 

	# Stright lines
	points2 = np.zeros((ndiv_straight,2))
	points4 = np.zeros((ndiv_straight,2))
	points2[0:,0] = np.linspace(rin,rout,ndiv_straight)
	points4[0:,1] = np.linspace(rout,rin,ndiv_straight)

	dum1 = np.append(points1,points2[1:,0:],axis=0)
	dum2 = np.append(dum1[0:-1,0:],np.flipud(points3),axis=0)
	points = np.append(dum2,points4[1:,0:],axis=0)
	points = points[0:-1,0:]

	facets = round_trip_connect(0, len(points)-1)
	facets = np.asarray(facets)

	# Modified points for discontinuous meshes
	dum1 = np.append(points1,points2,axis=0)
	dum2 = np.append(dum1,np.flipud(points3),axis=0)
	mod_points = np.append(dum2,points4,axis=0)
	# mod_points = mod_points[0:-1,0:]



	tcol1 = np.zeros(points1.shape); tcol1[0]=1; tcol1[-1]=1
	tcol2 = np.zeros(points2.shape); tcol2[0]=1; tcol2[-1]=1
	tcol3 = np.zeros(points3.shape); tcol3[0]=1; tcol3[-1]=1
	tcol4 = np.zeros(points4.shape); tcol4[0]=1; tcol4[-1]=1

	dum3 = np.append(tcol1,tcol2,axis=0)
	dum4 = np.append(dum3,np.flipud(tcol3),axis=0)
	tcol = np.append(dum4,tcol4,axis=0)
	# tcol = tcol[0:-1,0:]

	points_LM = np.zeros((mod_points.shape[0],3));
	points_LM[:,0:2] = mod_points
	points_LM[:,-1]=tcol[:,0]



	return points, facets, points_LM


# Meshing for plotting purposes
def round_trip_connect(start, end):
    return [(i, i+1) for i in range(start, end)] + [(end, start)]

def Meshing(geo_args):
	rin = geo_args.rin
	rout = geo_args.rout
	ndiv = geo_args.ndiv_curve
	# Define Margins
	Mx = 0.15 
	My = 0.15
	Mr = 0.3

	# Mx = 0.1 
	# My = 0.1
	# Mr = 0.1

	# Mx = 0. 
	# My = 0.
	# Mr = 0.
	# Geometry for meshing
	rin = 1.+Mr
	rout = 2.-Mr
	# ndiv = 10
	dum = np.linspace(0+Mx,np.pi/2.0-My,ndiv)
	points1 = np.zeros((ndiv,2))
	points2 = np.zeros((ndiv,2))
	for i in range(0,dum.shape[0]):
		points1[i,0] = rin*np.sin(dum[i]) 
		points1[i,1] = rin*np.cos(dum[i]) 

		points2[i,0] = rout*np.sin(dum[i]) 
		points2[i,1] = rout*np.cos(dum[i]) 

	points = np.append(points1,np.flipud(points2),axis=0)
	facets = round_trip_connect(0, len(points)-1)
	mesh = MeshGeneration().GenerateMesh(points,facets,max_vol=0.01,gen_faces=True) 

	return mesh


def DiscontinuousGlobalCoord(global_coord,C,geo_args):
	
	nelemp1 = geo_args.ndiv_curve-1
	nelemp2 = geo_args.ndiv_straight-1

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

	# Cheaper trick - the whole problem was here (coordinate of first node was missing)
	global_coord_discontinous[-1,1]=1


	global_coord = np.copy(global_coord_discontinous)

	return global_coord


# def ProblemData_BEM():
def ProblemData_BEM(C,nx,ny):

	# Polynomial Degree
	# C=1

	# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		# ndiv_curve = 5.
		# ndiv_straight = 5.
		# ndiv_curve = nx
		# ndiv_straight = nx/2.
		ndiv_curve = nx
		ndiv_straight = ny
		Lagrange_Multipliers = 'activated'
		corners = 4


	boundary, elements, modified_boundary = Geometry(geo_args)

	# Build element connectivity
	elements = np.zeros((len(boundary+1),2))
	elements[0:,0] = np.linspace(0,len(elements)-1,len(elements))
	elements[0:,1] = elements[0:,0]+1
	# elements[-1,1] = 0

	nx = geo_args.ndiv_curve
	ny = geo_args.ndiv_straight

	for i in range(0,elements.shape[0]):
		if i>=nx-1 and i<nx-1+ny-1:
			elements[i,0:]+=1
		elif i>=nx-1+ny-1 and i<2*(nx-1)+ny-1:
			elements[i,0:]+=2
		elif i>=2*(nx-1)+ny-1:
			elements[i,0:]+=3


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


	mesh = Meshing(geo_args)
	internal_points = np.array(mesh.points)



	return C, elements, modified_boundary, element_connectivity, internal_points, mesh, geo_args


def BoundaryConditions(nodal_coordinates,C=0,geo_args=0):

		 

				##
# Zero Flux 	#  # 	
				#    #											
				#     #
				 #     #
				   #	#	# T2									
	    		    #	 #	
			# T1     #	  #										
			    	 #	   #										
			    	 #     #                                	
			    	 #######  

					# Zero Flux 




	# Modify boundary data for LM formulation
	nelemp1 = geo_args.ndiv_curve-1
	nelemp2 = geo_args.ndiv_straight-1

	# Allocate
	boundary_data = np.zeros((nodal_coordinates.shape[0],2))+4   # 4 is for 4 corners change according to the problem

	for i in range(0,nodal_coordinates.shape[0]):
		if i<=nelemp1*(C+1):
			boundary_data[i,0] = 10
			boundary_data[i,1] = -1
		elif i>nelemp1*(C+1) and i<=nelemp1*(C+1)+nelemp2*(C+1)+1:
			boundary_data[i,0] = -1
			boundary_data[i,1] = 0
		elif i>nelemp1*(C+1)+nelemp2*(C+1)+1 and i<=2*nelemp1*(C+1)+nelemp2*(C+1)+2:
			boundary_data[i,0] = 6
			boundary_data[i,1] = -1

		elif i>2*nelemp1*(C+1)+nelemp2*(C+1)+2 and i<=2*nelemp1*(C+1)+2*nelemp2*(C+1)+3:
			boundary_data[i,0] = -1
			boundary_data[i,1] = 0

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
		analytic = -4./np.log(2)*np.log(np.sqrt(global_coord[:,0]**2+global_coord[:,1]**2))+10
		rel_err = la.norm(analytic-total_sol[:-4,0])/la.norm(analytic)
	# Computing error at internal points
	elif opt==1:
		analytic = -4./np.log(2)*np.log(np.sqrt(internal_points[:,0]**2+internal_points[:,1]**2))+10
		rel_err = la.norm(POT[:,0]-analytic)/la.norm(analytic)


	return rel_err




def PlotFunc(mesh,POT,Flux1,Flux2,opt=0):

	import matplotlib.pyplot as plt


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
		# plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$Potential$ (V)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_4_Potential_C_0.eps', format='eps', dpi=1000)

	elif opt==1:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		# plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$X-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_4_XFlux_C_0.eps', format='eps', dpi=1000)

	elif opt==2:
		plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
		plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
		plt.colorbar()

		plt.xlabel(r'$X-Coordinate$')
		plt.ylabel(r'$Y-Coordinate$')
		# plt.xlim((0,2)); plt.ylim((0,1))

		plt.title(r'$Y-Flux$ (V/$m$)')
		# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_4_YFlux_C_0.eps', format='eps', dpi=1000)

	plt.show()