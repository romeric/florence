"""
main interface between python and cpp's occ_frontend 
"""

import cython
import ctypes
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
np.import_array()

ctypedef np.int64_t Integer
ctypedef np.float64_t Real 

cdef extern from "py_to_occ_frontend.hpp": 
	struct to_python_structs:
		vector[Real] displacement_BC_stl
		vector[Integer] nodes_dir_out_stl
		Integer nodes_dir_size

	to_python_structs PyCppInterface (const char* iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols, 
		Integer* elements_array, Integer element_rows, Integer element_cols, Integer* edges, Integer edges_rows, Integer edges_cols,
		Integer* faces, Integer faces_rows, Integer faces_cols, Real condition, Real* boundary_fekete, Integer fekete_rows, Integer fekete_cols)


cdef class PyInterface_OCC_FrontEnd:
	"""
		wrapper for occ_frontend
	"""

	cdef bytes iges_filename
	cdef Integer ndim 
	cdef Real scale 
	cdef Real condition 
	# cdef np.ndarray[Real, ndim=2] self.points = None 
	# cdef np.ndarray[Integer, ndim=2] self.elements = None
	# cdef np.ndarray[Integer, ndim=2] self.edges = None
	# cdef np.ndarray[Integer, ndim=2] self.faces  = None
	# cdef np.ndarray[Real, ndim=2] self.boundary_fekete  = None
	# cdef np.int_t[:] testarray1
	cdef Real[:,:] points 
	cdef Integer[:,:] elements
	cdef Integer[:,:] edges 
	cdef Integer[:,:] faces 
	cdef Real[:,:] boundary_fekete 

	def __init__(self,dimension=2):
		self.ndim = dimension
		# NUMPY ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
		# cdef np.ndarray[Real, ndim=2, mode="c"] self.points = None 
		# cdef np.ndarray[Integer, ndim=2, mode="c"] self.elements = None
		# cdef np.ndarray[Integer, ndim=2, mode="c"] self.edges = None
		# cdef np.ndarray[Integer, ndim=2, mode="c"] self.faces  = None
		# cdef np.ndarray[Real, ndim=2, mode="c"] self.boundary_fekete  = None
		# cdef np.ndarray[Real, ndim=2] self.points = None 
		# cdef np.ndarray[Integer, ndim=2] self.elements = None
		# cdef np.ndarray[Integer, ndim=2] self.edges = None
		# cdef np.ndarray[Integer, ndim=2] self.faces  = None
		# cdef np.ndarray[Real, ndim=2] self.boundary_fekete  = None

	def SetMesh(self,np.ndarray[Real, ndim=2, mode="c"] points, np.ndarray[Integer, ndim=2, mode="c"] elements,
		np.ndarray[Integer, ndim=2, mode="c"] edges, np.ndarray[Integer, ndim=2, mode="c"] faces=None):

		self.elements = elements 
		self.points = points
		self.edges = edges 
		self.faces = faces 

		return self

	def SetCADGeometry(self,bytes iges_filename):
		self.iges_filename = iges_filename

	def SetDimension(self,Integer dim):
		self.ndim = dim 

	def SetScale(self,Real scale):
		self.scale = scale 

	def SetCondition(self,Real condition):
		self.condition = condition

	def SetBoundaryFeketePoints(self,np.ndarray[Real, ndim=2, mode="c"] boundary_fekete):
		self.boundary_fekete = boundary_fekete

	def ComputeDirichletBoundaryConditions2(self):
		return ComputeDirichletBoundaryConditions(self.iges_filename, self.scale, np.asarray(self.points), np.asarray(self.elements), 
			np.asarray(self.edges), np.asarray(self.faces), self.condition, np.asarray(self.boundary_fekete))
		# return 0
		# cdef Integer element_rows, points_rows, element_cols, points_cols, edges_rows, edges_cols, faces_rows, faces_cols

		# elements_rows, elements_cols = self.elements.shape[0], self.elements.shape[1]
		# points_rows, points_cols = self.points.shape[0], self.points.shape[1]
		# edges_rows, edges_cols = self.edges.shape[0], self.edges.shape[1]
		# faces_rows, faces_cols = self.faces.shape[0], self.faces.shape[1]

		# cdef Integer fekete_rows, fekete_cols
		# fekete_rows, fekete_cols = self.boundary_fekete.shape[0], self.boundary_fekete.shape[1]


		# # AT THE MOMENT THE ARRAYS ARE DEEPLY COPIED FROM CPP TO PYTHON. TO AVOID THIS THE OBJECTS HAVE TO BE 
		# # INITIALISED AND ALLOCATED IN PYTHON (CYTHON) THEN PASSED TO CPP. HOWEVER THIS IS ONLY POSSIBLE FOR FIXED
		# # ARRAY SIZES. LOOK AT THE END OF THE FILE FOR THESE ALTERNATIVE APPROACHES

		# cdef to_python_structs struct_to_python
		# # CALL CPP ROUTINE
		# struct_to_python = PyCppInterface (<const char*>self.iges_filename, self.scale, 
		# 	<Real *>self.points.data, points_rows, points_cols, <Integer *>self.elements.data, 
		# 	elements_rows, elements_cols,<Integer *>self.edges.data, edges_rows, edges_cols, <Integer *>self.faces.data, faces_rows, faces_cols, 
		# 	self.condition, <Real *>self.boundary_fekete.data, fekete_rows, fekete_cols)









# BACKUP

# """
# main interface between python and cpp's occ_frontend 
# """

# import cython
# import ctypes
# from libcpp.vector cimport vector

# import numpy as np
# cimport numpy as np
# np.import_array()

# ctypedef np.int64_t Integer
# ctypedef np.float64_t Real 

# cdef extern from "py_to_occ_frontend.hpp": 
# 	struct to_python_structs:
# 		vector[Real] displacement_BC_stl
# 		vector[Integer] nodes_dir_out_stl
# 		Integer nodes_dir_size

# 	to_python_structs PyCppInterface (const char* iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols, 
# 		Integer* elements_array, Integer element_rows, Integer element_cols, Integer* edges, Integer edges_rows, Integer edges_cols,
# 		Integer* faces, Integer faces_rows, Integer faces_cols, Real condition, Real* boundary_fekete, Integer fekete_rows, Integer fekete_cols)


# @cython.nonecheck(False)
# @cython.boundscheck(False)
# @cython.profile(False)
# @cython.wraparound(False)
# def ComputeDirichletBoundaryConditions(bytes iges_filename not None, Real scale, np.ndarray[Real, ndim=2, mode="c"] points not None, 
# 	np.ndarray[Integer, ndim=2, mode="c"] elements not None, np.ndarray[Integer, ndim=2, mode="c"] edges not None, 
# 	np.ndarray[Integer, ndim=2, mode="c"] faces not None, Real condition, np.ndarray[Real, ndim=2, mode="c"] boundary_fekete not None):
# 	"""
# 	wrapper for occ_frontend
# 	"""

# 	cdef Integer element_rows, points_rows, element_cols, points_cols, edges_rows, edges_cols, faces_rows, faces_cols

# 	elements_rows, elements_cols = elements.shape[0], elements.shape[1]
# 	points_rows, points_cols = points.shape[0], points.shape[1]
# 	edges_rows, edges_cols = edges.shape[0], edges.shape[1]
# 	faces_rows, faces_cols = faces.shape[0], faces.shape[1]

# 	cdef Integer fekete_rows, fekete_cols
# 	fekete_rows, fekete_cols = boundary_fekete.shape[0], boundary_fekete.shape[1]


# 	# AT THE MOMENT THE ARRAYS ARE DEEPLY COPIED FROM CPP TO PYTHON. TO AVOID THIS THE OBJECTS HAVE TO BE 
# 	# INITIALISED AND ALLOCATED IN PYTHON (CYTHON) THEN PASSED TO CPP. HOWEVER THIS IS ONLY POSSIBLE FOR FIXED
# 	# ARRAY SIZES. LOOK AT THE END OF THE FILE FOR THESE ALTERNATIVE APPROACHES

# 	cdef to_python_structs struct_to_python
# 	# CALL CPP ROUTINE
# 	struct_to_python = PyCppInterface (<const char*>iges_filename, scale, &points[0,0], points_rows, points_cols, &elements[0,0], 
# 		elements_rows, elements_cols,&edges[0,0], edges_rows, edges_cols, &faces[0,0], faces_rows, faces_cols, 
# 		condition, &boundary_fekete[0,0], fekete_rows, fekete_cols)


# 	cdef np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
# 	cdef Integer i 
# 	for i in range(struct_to_python.nodes_dir_size):
# 		nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
# 	cdef np.ndarray displacements_BC = np.zeros((points_cols*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
# 	for i in range(points_cols*struct_to_python.nodes_dir_size):
# 		displacements_BC[i] = struct_to_python.displacement_BC_stl[i]


# 	return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,points_cols) 