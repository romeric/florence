import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt 								# For plotting
# from MeshGeneration import MeshGeneration


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

	return points, facets




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

	from Core.MeshGeneration.MeshGeneration import MeshGeneration
	mesh = MeshGeneration().GenerateMesh(points,facets,max_vol=0.01,gen_faces=True) 

	return mesh


def ProblemData_BEM(C,nx,ny):

	# Polynomial Degree
	# C=3

	# Geometry 
	class geo_args(object):
		rin = 1.
		rout = 2.
		# ndiv_curve = 10.
		# ndiv_straight = 5.
		ndiv_curve = nx
		ndiv_straight = nx/2.
		Lagrange_Multipliers = 'deactivated'
		corners = 4

	# Build element connectivity
	boundary_points, elements = Geometry(geo_args)

	# Element connectivity for higher order basis
	element_connectivity = np.zeros((len(boundary_points+1),C+2),dtype=int)
	# element_connectivity[0:,0]=np.linspace(0,(C+1)*(element_connectivity.shape[0]-1),element_connectivity.shape[0])
	for i in range(0,element_connectivity.shape[0]):
		element_connectivity[i,0:]=np.linspace((C+1)*i,(C+1)*(i)+(C+1),element_connectivity.shape[1])
	element_connectivity[-1,-1]=0



	mesh = Meshing(geo_args)
	internal_points = np.array(mesh.points)

	# class inputdata(object):
	# 	order = C
	# 	boundary_points = boundary_points
	# 	element_connectivity = element_connectivity
	# 	internal_points = internal_points
	# 	mesh = mesh
	# 	geo_args = geo_args
			


	return C, elements, boundary_points, element_connectivity, internal_points, mesh, geo_args


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


	ndiv_straight = geo_args.ndiv_straight
	ndiv_curve = geo_args.ndiv_curve
	ndiv_curve = (C+1)*ndiv_curve-C
	ndiv_straight = (C+1)*ndiv_straight-C
	T1=10.; T2=6.;
	boundary_data = np.zeros((nodal_coordinates.shape[0],2))
	for i in range(0,boundary_data.shape[0]):
		if i<ndiv_curve:
			boundary_data[i,0]=T1
			boundary_data[i,1]=-1.
		elif i>= ndiv_curve and i< ndiv_curve+ndiv_straight-2:
			boundary_data[i,0]=-1.
			boundary_data[i,1]=0
		elif i>= ndiv_curve+ndiv_straight-2 and i< 2*ndiv_curve+ndiv_straight-2:
			boundary_data[i,0]=T2
			boundary_data[i,1]=-1.
		elif i>= 2*ndiv_curve+ndiv_straight-2:
			boundary_data[i,0]=-1.
			boundary_data[i,1]=0.

	return boundary_data





def ComputeErrorNorms(global_coord,total_sol,opt=0,internal_points=0,POT=0):
	# If opt=0, error at boundary is calculated
	# If opt=1, error at interior is calculated

	rel_err = 0

	# Computing error at boundary
	if opt==0:
		analytic = -4./np.log(2)*np.log(np.sqrt(global_coord[:,0]**2+global_coord[:,1]**2))+10
		rel_err = la.norm(analytic-total_sol[:,0])/la.norm(analytic)
	# Computing error at internal points
	elif opt==1:
		analytic = -4./np.log(2)*np.log(np.sqrt(internal_points[:,0]**2+internal_points[:,1]**2))+10
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
			# plt.xlim((0,2)); plt.ylim((0,1))

			plt.title(r'$Potential$ (V)')
			# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_2_Potential_C_0.eps', format='eps', dpi=1000)

		elif opt==1:
			plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
			plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
			plt.colorbar()

			plt.xlabel(r'$X-Coordinate$')
			plt.ylabel(r'$Y-Coordinate$')
			# plt.xlim((0,2)); plt.ylim((0,1))

			plt.title(r'$X-Flux$ (V/$m$)')
			# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_2_XFlux_C_0.eps', format='eps', dpi=1000)

		elif opt==2:
			plt.triplot(mesh_points[:,0], mesh_points[:,1], mesh_tris)
			plt.tricontourf(mesh_points[:,0], mesh_points[:,1], mesh_tris, Flux1.reshape(mesh_points.shape[0]), 100)
			plt.colorbar()

			plt.xlabel(r'$X-Coordinate$')
			plt.ylabel(r'$Y-Coordinate$')
			# plt.xlim((0,2)); plt.ylim((0,1))

			plt.title(r'$Y-Flux$ (V/$m$)')
			# plt.savefig('/home/roman/Dropbox/Latex_Images/Example_2_YFlux_C_0.eps', format='eps', dpi=1000)


		# u,v = np.meshgrid(Flux1, Flux2)
		# plt.streamplot(mesh_points[:,0], mesh_points[:,1],u,v)  # Vector plot has not been released yet 


		plt.show()

		# np.savetxt('/home/roman/Desktop/m1.txt', mesh_points[:,0])
		# np.savetxt('/home/roman/Desktop/m2.txt', mesh_points[:,1])
		# np.savetxt('/home/roman/Desktop/f1.txt', Flux1)
		# np.savetxt('/home/roman/Desktop/f2.txt', Flux2)
		# np.savetxt('m2', mesh_points[:,1])

		# np.savetxt('f1', Flux1)
		# np.savetxt('f2', Flux2)
