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
from SalomeMeshReader import ReadMesh
from HigherOrderMeshing import *
from warnings import warn


"""
Mesh class providing most of the pre-processing functionalities of the Core module

Roman Poya - 13/06/2015
"""



class Mesh(object):

	"""Mesh class provides the following functionalities:  
	1. Generating higher order meshes based on a linear mesh, for tris, tets, quads and hexes
	2. Generating linear tri and tet meshes based on meshpy back-end
	3. Generating linear tri meshes based on distmesh back-end
	4. Finding bounary edges and faces for tris and tets, in case it is not provided by the mesh generator
	5. Reading Salome meshes in data (.dat/.txt/etc) format
	6. Reading gmsh files .msh 
	7. Checking for node numbering order of elements and fixing it if desired
	8. Writing meshes to unstructured vtk file format (.vtu) based on Luke Olson's script"""

	def __init__(self):
		super(Mesh, self).__init__()
		self.elements = None
		self.points = None
		self.edges = None
		self.faces = None
		self.element_type = None

	def SetElements(self,arr):
		self.elements = arr

	def SetPoints(self,arr):
		self.points = arr

	def SetEdges(self,arr):
		self.edges = arr

	def SetFaces(self,arr):
		self.faces = arr

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
		for i in range(self.nelem):
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
			if edge1.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,1], self.elements[i,2]]])),axis=0)
			if edge2.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,2], self.elements[i,0]]])),axis=0)

		self.edges = boundaryEdges[1:,:]
		return edges


	def GetInteriorEdgesTri(self):
		newMesh = Mesh()
		newMesh.element_type = "tri"
		newMesh.nelem = self.elements.shape[0]
		newMesh.elements = self.elements[:,:3].astype(np.int64)
		newMesh_nnode = np.max(newMesh.elements)+1 # NO MONKEY PATCHING
		newMesh.points = self.points[:newMesh_nnode,:] 
		all_edges = newMesh.GetBoundaryEdgesTri(TotalEdges=True)
		print all_edges.shape
		print self.edges.shape
		print newMesh.edges.shape


	def GetBoundaryFacesTet(self,TotalFaces=False):
		"""Given a linear tetrahedral mesh, find the boundary faces (surfaces).
		
		input:
			
			TotalFaces:		if True returns all the faces in the tetrahedral mesh
							otherwise returns an empty list (default False)

		returns:			

			arr:			numpy ndarray of all faces provided TotalFaces is True"""

		# CHECK IF IT IS LINEAR MESH
		assert self.elements.shape[1]==4

		edges = [] # This is faces indeed
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
		for i in range(self.nelem):
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
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], 
					self.elements[i,1], self.elements[i,2]]])),axis=0)
			if edge1.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], 
					self.elements[i,1], self.elements[i,3]]])),axis=0)
			if edge2.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,0], 
					self.elements[i,2], self.elements[i,3]]])),axis=0)
			if edge3.shape[0]==1:
				boundaryEdges = np.concatenate((boundaryEdges,np.array([[self.elements[i,1], 
					self.elements[i,2], self.elements[i,3]]])),axis=0)


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


	def GetBoundaryFacesHigherTet(self):
		""" Get high order planar tetrahedral faces given high order element connectivity
			and nodal coordinates

			NOT TESTED

			"""

		assert self.elements is not None
		assert self.points is not None
		assert self.nelem is not None

		from Core.Supplementary.Tensors import itemfreq_py
		from Core.Supplementary.Where import whereEQ
		from Core.QuadratureRules.NodeArrangement import NodeArrangement


		# CHANGE TYPE 
		if isinstance(self.elements[0,0],np.uint64):
			self.elements = self.elements.astype(np.int64)
			self.faces = self.faces.astype(np.int64)

		# print self.faces.shape[1]
		if self.faces.shape[1] == 0:
			newMesh = Mesh()
			newMesh.nelem = self.elements.shape[0]
			newMesh.elements = self.elements[:,:4]
			newMesh_nnode = np.max(newMesh.elements)+1 # NO MONKEY PATCHING
			newMesh.points = self.points[:newMesh_nnode]
			newMesh.GetBoundaryFacesTet()
			self.faces = newMesh.faces

		if self.faces.shape[1] > 3:
			raise UserWarning("High order tetrahedral faces seem to be already computed. Do you want me to recompute them?")
			pass


		# INFER POLYNOMIAL DEGREE AND ORDER OF CONTINUITY
		C = self.InferPolynomialDegree()-1
		# GET NUMBER OF COLUMNS FOR HIGHER ORDER FACES
		fsize = int((C+2.)*(C+3.)/2.)
		# GET THE NODE ARRANGEMENT FOR FACES
		face_node_arrangment = NodeArrangement(C)[0]

		higher_order_faces = np.zeros((self.faces.shape[0],fsize),dtype = np.int64)
		elements_containing_faces = []
		counter = 0
		# FIND IF THE 3 NODES OF THE LINEAR FACE IS IN THE ELEMENT
		for i in range(self.faces.shape[0]):
			face_nodes = [];  face_cols = []
			for j in range(3):
				# rows, cols = np.where(self.elements[:,:4]==self.faces[i,j])
				rows, cols = whereEQ(self.elements[:,:4],self.faces[i,j])
				# STORE ALL THE OCCURENCES OF CURRENT NODAL FACE IN THE ELEMENT CONNECTIVITY
				face_nodes = np.append(face_nodes,rows)
				face_cols = np.append(face_cols,cols)
			# CHANGE TYPE
			face_nodes = face_nodes.astype(np.int64)
			face_cols = face_cols.astype(np.int64)
			# COUNT THE NUMBER OF OCCURENCES
			occurence_within_an_element = itemfreq_py(face_nodes)
			# SEE WHICH ELEMENT CONTAINS ALL THE THREE NODES OF THE FACE
			counts = np.where(occurence_within_an_element[:,1]==3)[0]
			if counts.shape[0] != 0:
				# COMPUTE ONLY IF NEEDED LATER ON
				# elements_containing_faces = np.append(elements_containing_faces, 
				# 	occurence_within_an_element[counts,0])

				# GET THE POSITIONS FROM CONCATENATION
				inv_uniques = np.unique(face_nodes,return_inverse=True)[1]
				which_connectivity_cols_idx = np.where(inv_uniques==counts)[0]
				# FROM THE POSITIONS GET THE COLUMNS AT WHICH THESE NODES OCCUR IN THE CONNECTIVITY
				which_connectivity_cols =  face_cols[which_connectivity_cols_idx]
				if which_connectivity_cols_idx.shape[0] != 0:
					# BASED ON THE OCCURENCE, DECIDE WHICH FACE OF THAT ELEMENT WE ARE ON
					if which_connectivity_cols[0]==0 and which_connectivity_cols[1]==1 \
						and which_connectivity_cols[2]==2:
						higher_order_faces[counter,:] = self.elements[
						occurence_within_an_element[counts,0],face_node_arrangment[0,:]]
						counter += 1
					elif which_connectivity_cols[0]==0 and which_connectivity_cols[1]==1 \
						and which_connectivity_cols[2]==3:
						higher_order_faces[counter,:] = self.elements[
						occurence_within_an_element[counts,0],face_node_arrangment[1,:]]
						counter += 1
					elif which_connectivity_cols[0]==0 and which_connectivity_cols[1]==2 \
						and which_connectivity_cols[2]==3:
						higher_order_faces[counter,:] = self.elements[
						occurence_within_an_element[counts,0],face_node_arrangment[2,:]]
						counter += 1
					elif which_connectivity_cols[0]==1 and which_connectivity_cols[1]==2 \
						and which_connectivity_cols[2]==3:
						higher_order_faces[counter,:] = self.elements[
						occurence_within_an_element[counts,0],face_node_arrangment[3,:]]
						counter += 1

		# CHECK WITH AN ALREADY EXISTING MESH GENERATOR
		# print np.linalg.norm(np.sort(higher_order_faces,axis=1) - np.sort(self.faces,axis=1))
				
		self.faces = higher_order_faces
		self.ChangeType()





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

		Note that this method could be useful for finding arbitrary edges not necessarily lying on the boundary"""

		# COMPUTE SLOPE OF THE GEOMETRY EDGE
		geo_edges = np.array(facets); geo_points = np.array(points)
		# ALLOCATE
		mesh_boundary_edges = np.zeros((1,2),dtype=int)
		# LOOP OVER GEOMETRY EDGES
		for i in range(geo_edges.shape[0]):
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
		self.elements = nmesh.elements.astype(np.uint64)
		self.edges = nmesh.edges.astype(np.uint64)
		if isinstance(self.faces,np.ndarray):
			self.faces = nmesh.faces.astype(np.uint64)
		self.nelem = nmesh.nelem
		self.element_type = nmesh.info 
		
		print 'Finished generating the high order mesh. Time taken', time()-t_mesh,'sec'





	def Read(self,*args,**kwargs):
		"""Default mesh reader for binary and text files used for reading Salome meshes mainly.
		The method checks if edges/faces are provided by the mesh generator and if not computes them"""


		mesh = ReadMesh(*args,**kwargs)

		self.points = mesh.points
		self.elements = mesh.elements.astype(np.int64)
		if isinstance(mesh.edges,np.ndarray):
			self.edges = mesh.edges.astype(np.int64)
		if isinstance(mesh.faces,np.ndarray):
			self.faces = mesh.faces.astype(np.int64)
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


	def ReadSeparate(self,connectivity_file,coordinates_file,mesh_type, edges_file = None, faces_file = None,
		delimiter_connectivity=' ',delimiter_coordinates=' ', delimiter_edges=' ', delimiter_faces=' ',
		ignore_cols_connectivity=None,ignore_cols_coordinates=None,ignore_cols_edges=None,
		ignore_cols_faces=None,index_style='c'):
		"""Read meshes when the element connectivity and nodal coordinates are written in separate files

		input:

			connectivity_file:				[str] filename containing element connectivity
			coordinates_file:				[str] filename containing nodal coordinates
			mesh_type:						[str] type of mesh tri/tet/quad/hex 
			edges_file:						[str] filename containing edges of the mesh (if not given gets computed)
			faces_file:						[str] filename containing faces of the mesh (if not given gets computed)
			delimiter_connectivity:			[str] delimiter for connectivity_file - default is white space/tab
			delimiter_coordinates:			[str] delimiter for coordinates_file - default is white space/tab
			delimiter_edges:				[str] delimiter for edges_file - default is white space/tab
			delimiter_faces:				[str] delimiter for faces_file - default is white space/tab
			ignore_cols_connectivity:		[int] no of columns to be ignored (from the start) in the connectivity_file 
			ignore_cols_coordinates:		[int] no of columns to be ignored (from the start) in the coordinates_file
			ignore_cols_edges: 				[int] no of columns to be ignored (from the start) in the connectivity_file 
			ignore_cols_faces:				[int] no of columns to be ignored (from the start) in the coordinates_file
			index_style:					[str] either 'c' C-based (zero based) indexing or 'f' fortran-based
												  (one based) indexing for elements connectivity - default is 'c'

			"""

		index = 0
		if index_style == 'c':
			index = 1

		from time import time; t1=time()	
		self.elements = np.loadtxt(connectivity_file,dtype=np.int64,delimiter=delimiter_connectivity) - index 
		# self.elements = np.fromfile(connectivity_file,dtype=np.int64,count=-1) - index
		self.points = np.loadtxt(coordinates_file,dtype=np.float64,delimiter=delimiter_coordinates)


		if ignore_cols_connectivity != None:
			self.elements = self.elements[ignore_cols_connectivity:,:]
		if ignore_cols_coordinates != None:
			self.points = self.points[ignore_cols_coordinates:,:]

		if (mesh_type == 'tri' or mesh_type == 'quad') and self.points.shape[1]>2:
			self.points = self.points[:,:2]

		self.element_type = mesh_type
		self.nelem = self.elements.shape[0]
		# self.edges = None 
		if edges_file is None:
			if mesh_type == "tri":
				self.GetBoundaryEdgesTri()
			elif mesh_type == "tet":
				self.GetBoundaryEdgesTet()
		else:
			self.edges = np.loadtxt(edges_file,dtype=np.int64,delimiter=delimiter_edges) - index
			if ignore_cols_edges !=None:
				self.edges = self.edges[ignore_cols_edges:,:] 

		if faces_file is None:
			if mesh_type == "tet":
				self.GetBoundaryFacesTet()
		else:
			self.faces = np.loadtxt(faces_file,dtype=np.int64,delimiter=delimiter_edges) - index
			if ignore_cols_faces !=None:
				self.faces = self.faces[ignore_cols_faces:,:]



	def ReadGIDMesh(self,filename,mesh_type,polynomial_order = 0):
		"""Read GID meshes"""

		self.element_type = mesh_type
		ndim, self.nelem, nnode, nboundary = np.fromfile(filename,dtype=np.int64,count=4,sep=' ')
	
		if ndim==2 and mesh_type=="tri":
			content = np.fromfile(filename,dtype=np.float64,count=4+3*nnode+4*self.nelem,sep=' ')
			self.points = content[4:4+3*nnode].reshape(nnode,3)[:,1:]
			self.elements = content[4+3*nnode:4+3*nnode+4*self.nelem].reshape(self.nelem,4)[:,1:].astype(np.int64)
			self.elements -= 1 

			self.GetBoundaryEdgesTri()

		if ndim==3 and mesh_type=="tet":
			content = np.fromfile(filename,dtype=np.float64,count=4+4*nnode+5*self.nelem,sep=' ')
			self.points = content[4:4+4*nnode].reshape(nnode,4)[:,1:]
			self.elements = content[4+4*nnode:4+4*nnode+5*self.nelem].reshape(self.nelem,5)[:,1:].astype(np.int64)
			self.elements -= 1

			self.GetBoundaryFacesTet()
			# self.GetBoundaryEdgesTet()




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



	def CheckNodeNumberingTri(self,change_order_to='retain'):
		"""Checks for node numbering order of the imported triangular mesh

		input:

			change_order_to:			[str] {'clockwise','anti-clockwise','retain'} changes the order to clockwise, 
										anti-clockwise or retains the numbering order - default is 'retain' 

		output: 

			original_order:				[str] {'clockwise','anti-clockwise','retain'} returns the original numbering order"""


		assert self.elements is not None
		assert self.points is not None

		# CHECK IF IT IS LINEAR MESH
		assert self.elements.shape[1]==3

		# C	##
			# #
			#  #
			#   #
			#    #
			#     #
		# A	######## B

		original_order = ''

		points = np.ones((self.points.shape[0],3),dtype=np.float64)
		points[:,:2]=self.points

		# FIND AREAS OF ALL THE ELEMENTS
		area = 0.5*np.linalg.det(points[self.elements,:])
		# CHECK NUMBERING
		if (area > 0).all():
			original_order = 'anti-clockwise'
			if change_order_to == 'clockwise':
				self.elements = np.fliplr(self.elements)
		elif (area < 0).all():
			original_order = 'clockwise'
			if change_order_to == 'anti-clockwise':
				self.elements = np.fliplr(self.elements)
		else:
			original_order = 'mixed'
			if change_order_to == 'clockwise':
				self.elements[area>0,:] = np.fliplr(self.elements[area>0,:])
			elif change_order_to == 'anti-clockwise':
				self.elements[area<0,:] = np.fliplr(self.elements[area<0,:])


		if original_order == 'anti-clockwise':
			print u'\u2713'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering'
		else:
			print u'\u2717'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering'

		return original_order


	def GetElementsWithBoundaryEdgesTri(self):
		""" Computes elements which have edges on the boundary. Mesh can be linear or higher order.
			Note that this assumes that a triangular element can only have one edge on the boundary.

		output: 

			edge_elements:				[1D array] array containing elements which have edge
										on the boundary. Row number is edge number"""


		assert self.edges is not None or self.elements is not None
		# CYTHON SOLUTION
		# from GetElementsWithBoundaryEdgesTri_Cython import GetElementsWithBoundaryEdgesTri_Cython
		# return GetElementsWithBoundaryEdgesTri_Cython(self.elements,self.edges)
		
		from Core.Supplementary.Where import whereEQ
		edge_elements = np.zeros(self.edges.shape[0],dtype=np.int64)
		for i in range(self.edges.shape[0]):
			x = []
			for j in range(self.edges.shape[1]):
				# x = np.append(x,np.where(self.elements==self.edges[i,j])[0])
				x = np.append(x,whereEQ(self.elements,self.edges[i,j])[0])
			# x = x.astype(np.int64)
			for k in range(len(x)):
				y = np.where(x==x[k])[0]
				# y = np.asarray(whereEQ(np.array([x]),np.int64(x[k]))[0])
				if y.shape[0]==self.edges.shape[1]:
					edge_elements[i] = np.int64(x[k])
					break

		return edge_elements


	def GetElementsWithBoundaryFacesTet(self):
		""" Computes elements which have faces on the boundary. Mesh can be linear or higher order.
			At most a tetrahedral can have all its four faces at boundary. It is assumed that a tetrahedral
			can only have one face at the boundary 

		output: 

			face_elements:				[1D array] array containing elements which have face
										on the boundary"""

		assert self.faces is not None or self.elements is not None

		face_elements = np.zeros(self.faces.shape[0],dtype=np.int64)
		for i in range(self.faces.shape[0]):
			x = []
			for j in range(self.faces.shape[1]):
				x = np.append(x,np.where(self.elements==self.faces[i,j])[0])
			# x = x.astype(np.int64)
			for k in range(len(x)):
				y = np.where(x==x[k])[0]
				if y.shape[0]==self.faces.shape[1]:
					face_elements[i] = np.int64(x[k])
					break

		return face_elements


	def ArrangeFacesTet(self):
		""" Computes elements which have faces on the boundary. Mesh can be linear or higher order.
			At most a tetrahedral can have all its four faces at boundary. It is assumed that a tetrahedral
			can only have one face at the boundary 

		output: 

			face_elements:				[1D array] array containing elements which have face
										on the boundary"""

		assert self.faces is not None or self.elements is not None

		# GET THE DEGREE
		p = 0
		
		for p in range(1,100):
			if (p+1)*(p+2)*(p+3)/6 == self.elements.shape[1]:
				break 
		fsize = int((p+1)*(p+2)/2) 

		from Core.Supplementary.Tensors import itemfreq_py
		# face_elements = -1*np.ones((self.faces.shape[0],4),dtype=np.int64)
		face_elements = []
		for i in range(self.faces.shape[0]):
			x,y = [],[]
			for j in range(self.faces.shape[1]):
				x = np.append(x,np.where(self.elements==self.faces[i,j])[0])
				y = np.append(y,np.where(self.elements==self.faces[i,j])[1])
				# print np.where(self.faces==self.elements[elem,j])
			if x.shape[0] > 0:
				freqs = itemfreq_py(x).astype(np.int64)
				# print freqs
				for k in range(freqs.shape[0]):
					if freqs[k,1] >= fsize:
						# print freqs[k,0]
						face_elements = np.append(face_elements,freqs[k,0])
						# print face_elements
						pass
			# print y

		# print face_elements
		return face_elements



	def SimplePlot(self):
		"""Simple mesh plot. Just a wire plot"""

		import matplotlib.pyplot as plt 
		fig = plt.figure()
		if self.element_type == "tri":
			plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3])

		elif self.element_type == "tet":
			# assert self.elements.shape[1] == 4
			if self.faces is None:
				raise ValueError('Mesh faces not available. Compute it first')
			from mpl_toolkits.mplot3d import Axes3D

			# FOR PLOTTING ELEMENTS
			# for elem in range(self.elements.shape[0]):
			# 	coords = self.points[self.elements[elem,:],:]
			# 	plt.gca(projection='3d')
			# 	plt.plot(coords[:,0],coords[:,1],coords[:,2],'-bo')

			# FOR PLOTTING ONLY BOUNDARY FACES
			if self.faces.shape[1] == 3:
				for face in range(self.faces.shape[0]):
					coords = self.points[self.faces[face,:3],:]
					plt.gca(projection='3d')
					plt.plot(coords[:,0],coords[:,1],coords[:,2],'-bo')
			else:
				for face in range(self.faces.shape[0]):
					coords = self.points[self.faces[face,:3],:]
					coords_all = self.points[self.faces[face,:],:]
					plt.gca(projection='3d')
					plt.plot(coords[:,0],coords[:,1],coords[:,2],'-b')
					plt.plot(coords_all[:,0],coords_all[:,1],coords_all[:,2],'bo')
		else:
			raise NotImplementedError("SimplePlot for "+self.element_type+" not implemented yet")

		plt.axis("equal")
		# plt.axis("off")
		# plt.savefig('/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/initial_plots/mech2d_original_mesh.eps',format='eps', dpi=1000)
		plt.show()



	def PlotMeshNumberingTri(self):
		"""Plots element and node numbers on top of the triangular mesh"""

		import matplotlib.pyplot as plt

		fig = plt.figure()
		plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3])
		plt.tricontourf(self.points[:,0], self.points[:,1], self.elements[:,:3], np.ones(self.points.shape[0]), 100,alpha=0.3)

		for i in range(0,self.elements.shape[0]):
			coord = self.points[self.elements[i,:],:]
			x_avg = np.sum(coord[:,0])/self.elements.shape[1]
			y_avg = np.sum(coord[:,1])/self.elements.shape[1]
			plt.text(x_avg,y_avg,str(i),backgroundcolor='#F88379',ha='center')

		for i in range(0,self.points.shape[0]):
			plt.text(self.points[i,0],self.points[i,1],str(i),backgroundcolor='#0087BD',ha='center')

		plt.axis('equal')
		# plt.show(block=False)
		plt.show()



	def WriteVTK(self,*args,**kwargs):
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
				fname = i
		for i in range(len(args)):
			if isinstance(args[i],str):
				FNAME = True 
				fname = args[i]
		if not FNAME:
			# IF NOT WRITE TO THE TOP LEVEL DIRECTORY
			pathvtu = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
			warnings.warn('No name given to the vtu output file. I am going to write one at the top level directory')
			fname = pathvtu+str('/output.vtu')
			write_vtu(Verts=self.points, Cells={cellflag:self.elements},fname=fname,**kwargs)
		else:
			write_vtu(Verts=self.points, Cells={cellflag:self.elements},fname=fname,**kwargs)



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

			center:				[tuple] containing the (x,y) coordinates of the center of the circle
			inner_radius:		[double] radius of inner circle 
			outer_radius:		[double] radius of outer circle
			element_type:		[str] tri for triangular mesh and quad for quadrilateral mesh 
			isotropic:			[boolean] option for isotropy or anisotropy of the mesh 
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


	def Sphere(self,radius=1,points=10):
		"""Create a tetrahedral mesh on an sphere

		input:

			radius:			[double] radius of sphere
			points:			[int] no of disrectisation"""

		from math import pi, cos, sin
		from meshpy.tet import MeshInfo, build
		from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder

		r = radius

		dphi = pi/points

		def truncate(r):
			if abs(r) < 1e-10:
				return 0
			else:
				return r

		rz = [(truncate(r*sin(i*dphi)), r*cos(i*dphi)) for i in range(points+1)]

		geob = GeometryBuilder()
		geob.add_geometry(*generate_surface_of_revolution(rz,
				closure=EXT_OPEN, radial_subdiv=10))

		mesh_info = MeshInfo()
		geob.set(mesh_info)

		mesh = build(mesh_info)

		self.points = np.asarray(mesh.points)
		self.elements = np.asarray(mesh.elements)
		# self.faces = np.asarray(mesh.faces)
		# self.edges = np.asarray(self.edges)
		self.nelem = self.elements.shape[0]
		self.element_type = "tet"


		# GET EDGES & FACES
		self.GetBoundaryFacesTet()
		# self.GetBoundaryEdgesTet()
		# self.SimplePlot()

		# print np.linalg.norm(self.points[np.unique(self.faces),:],axis=1)
		# print self.elements.shape, self.points.shape
		# import sys; sys.exit()



	def RemoveElements(self,(x_min,y_min,x_max,y_max),element_removal_criterion="all",keep_boundary_only=False,
			compute_edges=True,compute_faces=True,plot_new_mesh=True):
		"""Removes elements with some specified criteria

		input:				
			(x_min,y_min,x_max,y_max)		[tuple of doubles] box selection. Deletes all the elements apart 
											from the one within this box 
			element_removal_criterion 		[str]{"all","any"} the criterion for element removal with box selection. 
											How many nodes of the element should be within the box in order 
											not to be removed. Default is "all". "any" implies at least one node 
			keep_boundary_only				[bool] delete all elements apart from the boundary ones 
			compute_edges					[bool] if True also compute new edges
			compute_faces					[bool] if True also compute new faces (only 3D)
			plot_new_mesh					[bool] if True also plot the new mesh

		1. Note that this method computes a new mesh without maintaining a copy of the original 
		2. Different criteria can be mixed for instance removing all elements in the mesh apart from the ones
		in the boundary which are within a box
		"""

		assert self.elements != None
		assert self.points != None
		assert self.edges != None

		new_elements = np.zeros((1,3),dtype=np.int64)

		edge_elements = np.arange(self.nelem)
		if keep_boundary_only == True:
			if self.element_type == "tri":
				edge_elements = self.GetElementsWithBoundaryEdgesTri()

		for elem in edge_elements:
			xe = self.points[self.elements[elem,:],0]
			ye = self.points[self.elements[elem,:],1]

			if element_removal_criterion == "all":
				if ( (xe > x_min).all() and (ye > y_min).all() and (xe < x_max).all() and (ye < y_max).all() ):
					new_elements = np.vstack((new_elements,self.elements[elem,:]))

			elif element_removal_criterion == "any":
				if ( (xe > x_min).any() and (ye > y_min).any() and (xe < x_max).any() and (ye < y_max).any() ):
					new_elements = np.vstack((new_elements,self.elements[elem,:]))

		self.elements = new_elements[1:,:]
		self.nelem = self.elements.shape[0]
		unique_elements, inv_elements =  np.unique(self.elements,return_inverse=True)
		self.points = self.points[unique_elements,:]
		# RE-ORDER ELEMENT CONNECTIVITY
		remap_elements =  np.arange(self.points.shape[0])
		self.elements = remap_elements[inv_elements].reshape(self.nelem,self.elements.shape[1])

		# RECOMPUTE EDGES 
		if compute_edges == True:
			if self.element_type == "tri":
				self.GetBoundaryEdgesTri()

		# RECOMPUTE FACES 
		if compute_faces == True:
			if self.element_type == "tet":
				# FACES WILL BE COMPUTED AUTOMATICALLY
				self.GetBoundaryEdgesTet()

		# PLOT THE NEW MESH
		if plot_new_mesh == True:
			import matplotlib.pyplot as plt
			fig = plt.figure()
			plt.triplot(self.points[:,0],self.points[:,1],self.elements[:,:3])
			plt.axis('equal')
			plt.show()


	def ChangeType(self):
		""" Change mesh data type from signed to unsigned"""
		self.elements = self.elements.astype(np.uint64)
		if isinstance(self.edges,np.ndarray):
			self.edges = self.edges.astype(np.uint64)
		if isinstance(self.faces,np.ndarray):
			self.faces = self.faces.astype(np.uint64)


	def InferPolynomialDegree(self):
		""" Infer the degree of interpolation (p) based on the shape of 
			self.elements

			returns:		[int] polynomial degree
			"""

		assert self.element_type is not None
		assert self.elements is not None

		p = 0
		if self.element_type == "tet":
			for i in range(100):
				if (i+1)*(i+2)*(i+3)/6==self.elements.shape[1]:
					p = i
					break

		if self.element_type == "tri":
			for i in range(100):
				if (i+1)*(i+2)/2==self.elements.shape[1]:
					p = i
					break 

		return p 
