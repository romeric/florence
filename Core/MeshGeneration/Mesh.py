"""
Mesh class providing most of the pre-processing functionalities of the Core module

Roman Poya - 13/06/2015
"""



from __future__ import division
import os, warnings
from time import time
import numpy as np 
from Core.Supplementary.Where import *
from Core.Supplementary.Tensors import duplicate
from vtk_writer import write_vtu
from NormalDistance import NormalDistance
try:
    import meshpy.triangle as triangle
except ImportError:
    meshpy = None
from ReadSalomeMesh import ReadMesh
from HigherOrderMeshing import *



class Mesh(object):

	"""Mesh class provides the following functionalities:  
	1. Generating higher order meshes based on a linear mesh, for tris, tets, quads and hexes
	2. Generating linear tri and tet meshes based on meshpy back-end
	3. Generating linear tri meshes based on distmesh back-end
	4. Finding bounary edges and faces for tris and tets, in case it is not provided by the mesh generator
	5. Reading Salome meshes in data (.dat/.txt/etc) format
	6. Reading gmsh files .msh 
	7. Writing meshes to unstructured vtk file format (.vtu) based on Luke Olson's script"""

	def __init__(self):
		super(Mesh, self).__init__()
		self.elements = None
		self.points = None
		self.edges = None
		self.faces = None
		self.element_type = None


	def GetBoundaryEdgesTri(self,TotalEdges=False):
		"""Given a linear triangular mesh, find the boundary edges (lines).
		
		input:
			
			elements: 		nx3 numpy.ndarray of integers
			TotalEdges:		if True returns all the edges in the triangular mesh
							otherwise returns an empty list (default False)

		returns:			
			
			arr:			numpy ndarray of all edges provided TotalEdges is True"""


		# CHECK IF IT IS LINEAR MESH
		assert self.elements.shape[1]==3

		edges = []
		if TotalEdges is True:
			# GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
			edges = np.zeros((3*self.elements.shape[0],2),dtype=np.int64)
			edges[:self.elements.shape[0],:] = self.elements[:,[0,1]]
			edges[self.elements.shape[0]:2*self.elements.shape[0],:] = self.elements[:,[1,2]]
			edges[2*self.elements.shape[0]:3*self.elements.shape[0],:] = self.elements[:,[2,0]]

		boundaryEdges = np.zeros((1,2),dtype=np.int64)
		# interiorEdges = np.zeros((1,2),dtype=np.int64)

		# LOOP OVER ALL ELEMENTS
		for i in range(self.elements.shape[0]):
			# FIND HOW MANY ELEMENTS SHARE A SPECIFIC NODE
			x = whereEQ(self.elements,self.elements[i,0])[0]
			y = whereEQ(self.elements,self.elements[i,1])[0]
			z = whereEQ(self.elements,self.elements[i,2])[0]

			# A BOUNDARY EDGE IS ONE WHICH IS NOT SHARED WITH ANY OTHER ELEMENT
			edge0 =  np.intersect1d(x,y)
			edge1 =  np.intersect1d(y,z)
			edge2 =  np.intersect1d(z,x)


			if edge0.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], self.elements[i,1]]])),axis=0)
			elif edge1.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,1], self.elements[i,2]]])),axis=0)
			elif edge2.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,2], self.elements[i,0]]])),axis=0)


		self.edges = boundaryEdges[1:,:]
		return edges

	def GetBoundaryFacesTet(self,TotalFaces=False):
		"""Given a linear tetrahedral mesh, find the boundary faces (surfaces).
		
		input:
			
			TotalFaces:		if True returns all the faces in the tetrahedral mesh
							otherwise returns an empty list (default False)

		returns:			

			arr:			numpy ndarray of all faces provided TotalFaces is True"""

		# CHECK IF IT IS LINEAR MESH
		assert self.elements.shape[1]==4

		edges = []
		if TotalFaces is True:
			# GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
			edges = np.zeros((4*self.elements.shape[0],3),dtype=np.int64)
			edges[:self.elements.shape[0],:] = self.elements[:,[0,1,2]]
			edges[self.elements.shape[0]:2*self.elements.shape[0],:] = self.elements[:,[0,1,3]]
			edges[2*self.elements.shape[0]:3*self.elements.shape[0],:] = self.elements[:,[0,2,3]]
			edges[2*self.elements.shape[0]:3*self.elements.shape[0],:] = self.elements[:,[1,2,3]]

		boundaryEdges = np.zeros((1,3),dtype=np.int64)
		# interiorEdges = np.zeros((1,3),dtype=np.int64)

		# LOOP OVER ALL ELEMENTS
		for i in range(self.elements.shape[0]):
			# FIND HOW MANY ELEMENTS SHARE A SPECIFIC NODE
			x = whereEQ(self.elements,self.elements[i,0])[0]
			y = whereEQ(self.elements,self.elements[i,1])[0]
			z = whereEQ(self.elements,self.elements[i,2])[0]
			w = whereEQ(self.elements,self.elements[i,3])[0]

			# A BOUNDARY EDGE IS ONE WHICH IS NOT SHARED WITH ANY OTHER ELEMENT
			edge0 =  np.intersect1d(np.intersect1d(x,y),z)
			edge1 =  np.intersect1d(np.intersect1d(x,y),w)
			edge2 =  np.intersect1d(np.intersect1d(x,z),w)
			edge3 =  np.intersect1d(np.intersect1d(y,z),w)

			if edge0.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], self.elements[i,1], self.elements[i,2]]])),axis=0)
			elif edge1.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], self.elements[i,1], self.elements[i,3]]])),axis=0)
			elif edge2.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], self.elements[i,2], self.elements[i,3]]])),axis=0)
			elif edge3.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,1], self.elements[i,2], self.elements[i,3]]])),axis=0)


		self.faces = boundaryEdges[1:,:]
		return edges

	def GetBoundaryEdgesTet(self):
	# def GetBoundaryEdgesTet(self,elements,TotalEdges=False):
		"""Given a linear tetrahedral mesh, find the boundary edges (lines).
			Note that for tetrahedrals this function is more robust than Salome's default edge generator"""

		if self.faces is None or self.faces is list:
			raise AttributeError('Tetrahedral edges cannot be computed independent of tetrahedral faces. Compute faces first')

		# FIRST GET BOUNDARY FACES
		self.GetBoundaryFacesTet()

		elements = np.copy(self.elements)
		self.elements = np.copy(self.faces)
		# ALL THE EDGES CORRESPONDING TO THESE BOUNDARY FACES ARE BOUNDARY EDGES
		# TotalEdgesDuplicated = self.GetBoundaryEdgesTri(faces,TotalEdges=True)[1]
		TotalEdgesDuplicated = self.GetBoundaryEdgesTri(TotalEdges=True)

		duplicates = duplicate(np.sort(TotalEdgesDuplicated,axis=1))
		remaining = np.zeros(len(duplicates),dtype=np.int64)
		for i in range(len(duplicates)):
			remaining[i]=duplicates[i][0]

		self.elements = elements
		self.edges = TotalEdgesDuplicated[remaining,:]


	def BoundaryEdgesfromPhysicalParametrisation(self,points,facets,mesh_points,mesh_edges):
	    """Given a 2D planar mesh (mesh_points,mesh_edges) and the parametrisation of the physical geometry (points and facets)
	    	finds boundary edges

	    input:

	    	points:			nx2 numpy array of points given to meshpy/distmesh/etc API for the actual geometry
	    	facets:			mx2 numpy array connectivity of points given to meshpy/distmesh/etc API for the actual geometry
	    	mesh_points:	px2 numpy array of points taken from the mesh generator
	    	mesh_edges:		qx2 numpy array of edges taken from the mesh generator

	    returns:

	    	arr_1:			nx2 numpy array of boundary edges

	    Note that this method is useful for finding arbitrary edges"""

	    # COMPUTE SLOPE OF THE GEOMETRY EDGE
	    geo_edges = np.array(facets); geo_points = np.array(points)
	    # ALLOCATE
	    mesh_boundary_edges = np.zeros((1,2),dtype=int)
	    # LOOP OVER GEOMETRY EDGES
	    for i in range(0,geo_edges.shape[0]):
	    	# GET COORDINATES OF BOTH NODES AT THE EDGE
	    	geo_edge_coord = geo_points[geo_edges[i]]
	    	geo_x1 = geo_edge_coord[0,0];		geo_y1 = geo_edge_coord[0,1]
	    	geo_x2 = geo_edge_coord[1,0];		geo_y2 = geo_edge_coord[1,1]

	    	# COMPUTE SLOPE OF THIS LINE
	    	geo_angle = np.arctan((geo_y2-geo_y1)/(geo_x2-geo_x1)) #*180/np.pi

	    	# NOW FOR EACH OF THESE GEOMETRY LINES LOOP OVER ALL MESH EDGES
	    	for j in range(0,mesh_edges.shape[0]):
	    		mesh_edge_coord = mesh_points[mesh_edges[j]]
	    		mesh_x1 = mesh_edge_coord[0,0];			mesh_y1 = mesh_edge_coord[0,1]
	    		mesh_x2 = mesh_edge_coord[1,0];			mesh_y2 = mesh_edge_coord[1,1]

	    		# FIND SLOPE OF THIS LINE 
	    		mesh_angle = np.arctan((mesh_y2-mesh_y1)/(mesh_x2-mesh_x1))

	    		# CHECK IF GEOMETRY AND MESH EDGES ARE PARALLEL
	    		if np.allclose(geo_angle,mesh_angle,atol=1e-12):
	    			# IF SO THEN FIND THE NORMAL DISTANCE BETWEEN THEM
	    			P1 = np.array([geo_x1,geo_y1,0]);				P2 = np.array([geo_x2,geo_y2,0])		# 1st line's coordinates
	    			P3 = np.array([mesh_x1,mesh_y1,0]);				P4 = np.array([mesh_x2,mesh_y2,0])		# 2nd line's coordinates

	    			dist = NormalDistance(P1,P2,P3,P4)
	    			# IF NORMAL DISTANCE IS ZEROS THEN MESH EDGE IS ON THE BOUNDARY
	    			if np.allclose(dist,0,atol=1e-14):
	    				mesh_boundary_edges = np.append(mesh_boundary_edges,mesh_edges[j].reshape(1,2),axis=0)

	    # return np.delete(mesh_boundary_edges,0,0)
	    return mesh_boundary_edges[1:,:]







	def GetHighOrderMesh(self,C,**kwargs):
		"""Given a linear tri, tet, quad or hex mesh compute high order mesh based on it.
		This is a static method linked to the HigherOrderMeshing module"""

		print 'Generating p = '+str(C+1)+' mesh based on the linear mesh...'
		t_mesh = time()
		# BUILD A NEW MESH BASED ON THE LINEAR MESH
		if self.element_type == 'tri':
			# BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TRIANGLES  
			# nmesh = HighOrderMeshTri(C,self,**kwargs)
			nmesh = HighOrderMeshTri_UNSTABLE(C,self,**kwargs)

		elif self.element_type == 'tet':
			# BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TETRAHEDRALS
			# nmesh = HighOrderMeshTet(C,self,**kwargs)
			nmesh = HighOrderMeshTet_UNSTABLE(C,self,**kwargs)

		elif self.element_type == 'quad':
			# BUILD A NEW MESH USING THE GAUSS-LOBATTO NODAL POINTS FOR QUADS
			nmesh = HighOrderMeshQuad(C,self,**kwargs)

		self.points = nmesh.points
		self.elements = nmesh.elements
		self.edges = nmesh.edges
		self.faces = nmesh.faces
		self.nelem = nmesh.nelem
		self.element_type = nmesh.info 
		
		print 'Finished generating the high order mesh. Time taken', time()-t_mesh,'sec'









	def Read(self,*args,**kwargs):
		"""Default mesh reader for binary and text files used for reading Salome meshes mainly.
		The method checks if edges/faces are provided by the mesh generator and if not computes them"""

		mesh = ReadMesh(*args,**kwargs)
		self.points = mesh.points
		self.elements = mesh.elements
		self.edges = mesh.edges
		self.faces = mesh.faces
		self.nelem = mesh.nelem
		self.element_type = mesh.info 

		# RETRIEVE FACE/EDGE ATTRIBUTE
		if self.element_type == 'tri' and (type(getattr(self,'edges')) is None or type(getattr(self,'edges')) is list):
			# COMPUTE EDGES
			self.GetBoundaryEdgesTri()
		if self.element_type == 'tet' and (type(getattr(self,'faces')) is None or type(getattr(self,'faces')) is list or\
		type(getattr(self,'edges')) is None or type(getattr(self,'edges')) is list):
			# COMPUTE FACES & EDGES
			self.GetBoundaryFacesTet()
			self.GetBoundaryEdgesTet()


	def ReadSeparate(self,connectivity_file,coordinates_file,mesh_type,delimiter_connectivity=' ',delimiter_coordinates=' ',
		ignore_cols_connectivity=None,ignore_cols_coordinates=None,index_style='c'):
		"""Read meshes when the element connectivity and nodal coordinates are written in separate files

		input:

			connectivity_file:				[str] filename containing element connectivity
			coordinates_file:				[str] filename containing nodal coordinates
			mesh_type:						[str] type of mesh tri/tet/quad/hex 
			delimiter_connectivity:			[str] delimiter for connectivity_file - default is white space/tab
			delimiter_coordinates:			[str] delimiter for coordinates_file - default is white space/tab
			ignore_cols_connectivity:		[int] no of columns to be ignored (from the start) in the connectivity_file 
			ignore_cols_coordinates:		[int] no of columns to be ignored (from the start) in the coordinates_file
			index_style:					[str] either 'c' C-based (zero based) indexing or 'f' fortran-based
											  	  (one based) indexing for elements connectivity - default is 'c'

			"""

		index = 0
		if index_style == 'c':
			index = 1

		self.elements = np.loadtxt(connectivity_file,dtype=np.int64,delimiter=delimiter_connectivity) - index 
		self.points = np.loadtxt(coordinates_file,dtype=np.float64,delimiter=delimiter_coordinates)

		if ignore_cols_connectivity != None:
			self.elements = self.elements[ignore_cols_connectivity:,:]
		if ignore_cols_coordinates != None:
			self.points = self.points[ignore_cols_coordinates:,:]

		if (mesh_type == 'tri' or mesh_type == 'quad') and self.points.shape[1]>2:
			self.points = self.points[:,:2]


		self.element_type = mesh_type
		self.nelem = self.elements.shape[0]
		self.edges = None 
		self.GetBoundaryEdgesTri()




		

	def Readgmsh(self,filename):
		"""Read gmsh (.msh) file. TO DO"""
		from gmsh import Mesh as msh
		mesh = msh()
		mesh.read_msh(filename)
		self.points = mesh.Verts
		print dir(mesh)
		self.elements = mesh.Elmts 
		print self.elements
		# print mesh.Phys

		


	def WriteVTK(self,**kwargs):
		"""Write mesh/results to vtu"""

		cellflag = None
		if self.element_type =='tri':
			cellflag = 5
		elif self.element_type =='quad':
			cellflag = 9
		if self.element_type =='tet':
			cellflag = 10
		elif self.element_type == 'hex':
			cellflag = 12

		# CHECK IF THE OUTPUT FILE NAME IS SPECIFIED 
		FNAME = False
		for i in kwargs.items():
			if 'fname' in i:
				FNAME = True
		if not FNAME:
			# IF NOT WRITE TO THE TOP LEVEL DIRECTORY
			pathvtu = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
			warnings.warn('No name given to the vtu output file. I am going to write one at the top level directory')
			fname = pathvtu+str('/output.vtu')
			write_vtu(Verts=self.points, Cells={cellflag:self.elements},fname=fname,**kwargs)
		else:
			write_vtu(Verts=self.points, Cells={cellflag:self.elements},**kwargs)



	@staticmethod
	def MeshPyTri(points,facets,*args,**kwargs):
		"""MeshPy backend for generating linear triangular mesh"""
		info = triangle.MeshInfo()
		info.set_points(points)
		info.set_facets(facets)

		return triangle.build(info,*args,**kwargs)


	def UniformHollowCircle(self,center=(0,0),inner_radius=1.0,outer_radius=2.,element_type='tri',isotropic=True,nrad=5,ncirc=10):
		"""Generates isotropic and anisotropic tri and quad meshes on a hollow circle.

		input:

			center:				[tuple] containing the (x,y) coordinates of the circle
			inner_radius:		[double] radius of inner circle 
			outer_radius:		[double] radius of outer circle
			element_type:		[str] tri for triangular mesh and quad for quadrilateral mesh 
			isotropic:			[Boolean] option for isotropy or anisotropy of the mesh 
			nrad:				[int] number of disrectisation in the radial direction
			ncirc:				[int] number of disrectisation in the circumferential direction

		output:					[Mesh] an instance of the Mesh class  
		"""

		if np.allclose(inner_radius,0):
			raise ValueError('inner_radius cannot be zero')

		t = np.linspace(0,2*np.pi,ncirc+1)
		if isotropic is True:
			radii = np.linspace(inner_radius,outer_radius,nrad+1)
		else:
			base = 3
			radii = np.zeros(nrad+1,dtype=np.float64)
			mm = np.linspace(np.power(inner_radius,1./base),np.power(outer_radius,1./base),nrad+1)
			for i in range(0,nrad+1):
				radii[i] = mm[i]**base


			# base = 3
			# mm = np.linspace(np.power(inner_radius,1./base),np.power(2.,1./base),nrad+1)
			# mm = np.append(mm,np.linspace(2,outer_radius,nrad+1)) 
			# radii = np.zeros(mm.shape[0],dtype=np.float64)
			# for i in range(0,mm.shape[0]):
			# 	radii[i] = mm[i]**base


		# dd =   np.logspace(inner_radius,outer_radius,nrad+1,base=2)/2**np.linspace(inner_radius,outer_radius,nrad+1)
		# print dd*np.linspace(inner_radius,outer_radius,nrad+1)
		# print dd
		# print np.logspace(0,1.5,nrad+1,base=2)
		


		xy = np.zeros((radii.shape[0]*t.shape[0],2),dtype=np.float64)
		# cost = np.cos(t)
		# sint = np.sin(t)
		for i in range(0,radii.shape[0]):
			# xy[i*t.shape[0]:(i+1)*t.shape[0],0] = radii[i]*cost
			# xy[i*t.shape[0]:(i+1)*t.shape[0],1] = radii[i]*sint
			xy[i*t.shape[0]:(i+1)*t.shape[0],0] = radii[i]*np.cos(t)
			xy[i*t.shape[0]:(i+1)*t.shape[0],1] = radii[i]*np.sin(t)


		# REMOVE DUPLICATES GENERATED BY SIN/COS OF LINSPACE
		xy = xy[np.setdiff1d( np.arange(xy.shape[0]) , np.linspace(t.shape[0]-1,xy.shape[0]-1,radii.shape[0]).astype(int) ),:]

		connec = np.zeros((1,4),dtype=np.int64)
		# connec = ((),)

		for j in range(1,radii.shape[0]):	
			for i in range((j-1)*(t.shape[0]-1),j*(t.shape[0]-1)):
				if i<j*(t.shape[0]-1)-1:
					connec = np.concatenate((connec,np.array([[i,t.shape[0]-1+i,t.shape[0]+i,i+1 ]])),axis=0)
					# connec = connec + ((i,t.shape[0]-1+i,t.shape[0]+i,i+1),)
				else:
					connec = np.concatenate((connec,np.array([[i,t.shape[0]-1+i,j*(t.shape[0]-1),(j-1)*(t.shape[0]-1) ]])),axis=0)
					# connec = connec + ((i,t.shape[0]-1+i,j*(t.shape[0]-1),(j-1)*(t.shape[0]-1)),)

		connec = connec[1:,:]
		# connec = np.asarray(connec[1:])


		if element_type == 'tri':
			connec_tri = np.zeros((2*connec.shape[0],3),dtype=np.int64)
			for i in range(connec.shape[0]):
				connec_tri[2*i,:] = np.array([connec[i,0],connec[i,1],connec[i,3]])
				connec_tri[2*i+1,:] = np.array([connec[i,2],connec[i,3],connec[i,1]])

			self.elements = connec_tri
			# OBTAIN MESH EDGES
			self.GetBoundaryEdgesTri()
		elif element_type == 'quad':
			self.elements = connec
			raise NotImplementedError('GetBoundaryEdgesQuad is not yet implemented')


		# ASSIGN NODAL COORDINATES
		self.points = xy
		# IF CENTER IS DIFFERENT FROM (0,0)
		self.points[:,0] += center[0]
		self.points[:,1] += center[1]
		# ASSIGN PROPERTIES
		self.element_type = element_type
		self.nelem = self.elements.shape[0]
		self.nnode = self.points.shape[0]
