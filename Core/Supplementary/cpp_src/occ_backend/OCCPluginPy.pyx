"""
main interface between python and cpp's occ_frontend 
"""

import cython
import ctypes
from libcpp.vector cimport vector
from libcpp.string cimport string

import numpy as np
cimport numpy as np

ctypedef np.int64_t Integer
ctypedef np.float64_t Real 

cdef extern from "OCCPluginInterface.hpp": 
	struct to_python_structs:
		vector[Real] displacement_BC_stl
		vector[Integer] nodes_dir_out_stl
		Integer nodes_dir_size

	to_python_structs PyCppInterface (const char* iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols, 
		Integer* elements_array, Integer element_rows, Integer element_cols, Integer* edges, Integer edges_rows, Integer edges_cols,
		Integer* faces, Integer faces_rows, Integer faces_cols, Real condition, Real* boundary_fekete, Integer fekete_rows, Integer fekete_cols,
		const char* projection_method)

cdef extern from "OCCPlugin.hpp":
	cdef cppclass OCCPlugin:
		OCCPlugin()
		OCCPlugin(string &element_type, Integer &dim)
		void setarray(Integer *arr, Integer &rows, Integer &cols)

cdef class OCCPluginPy2:

	cdef bytes iges_filename
	cdef Integer ndim 
	cdef Real scale 
	cdef Real condition 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
	cdef Real[:,:] points 
	cdef Integer[:,:] elements
	cdef Integer[:,:] edges 
	cdef Integer[:,:] faces 
	cdef Real[:,:] boundary_fekete 
	cdef bytes projection_method

	cdef OCCPlugin *thisptr

	def __cinit__(self,dimension=2):
		self.ndim = dimension
		self.thisptr = new OCCPlugin()

	def __dealloc__(self):
		del self.thisptr

	def setarray(self,Integer[:,::1] elements):
		pass
		# print self.thisptr
		# cdef OCCPlugin interf = OCCPlugin()
		self.thisptr.setarray(&elements[0,0],elements.shape[0],elements.shape[1])


cdef class OCCPluginPy:
	"""
		Cython wrapper for OCCPlugin
	"""

	cdef bytes iges_filename
	cdef Integer ndim 
	cdef Real scale 
	cdef Real condition 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
	cdef Real[:,:] points 
	cdef Integer[:,:] elements
	cdef Integer[:,:] edges 
	cdef Integer[:,:] faces 
	cdef Real[:,:] boundary_fekete 
	cdef bytes projection_method

	def __init__(self,dimension=2):
		self.ndim = dimension

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

	def SetProjectionMethod(self, bytes projection_method):
		self.projection_method = projection_method

	def ComputeDirichletBoundaryConditions(self):
		# CALLS STATIC FUNCTION __ComputeDirichletBoundaryConditions__
		# NOTE THAT np.asarray IS USED TO CONVERT CYTHON'S MEMORY VIEW TO ndarray. 
		# ALTHOUGH THIS INVOLVES OVERHEAD THIS FUNCTION IS CALLED ONLY ONCE.
		return __ComputeDirichletBoundaryConditions__(self.iges_filename, self.scale, np.asarray(self.points), np.asarray(self.elements), 
			np.asarray(self.edges), np.asarray(self.faces), self.condition, np.asarray(self.boundary_fekete), self.projection_method)



@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.profile(False)
@cython.wraparound(False)
def __ComputeDirichletBoundaryConditions__(bytes iges_filename not None, Real scale, np.ndarray[Real, ndim=2, mode="c"] points not None, 
	np.ndarray[Integer, ndim=2, mode="c"] elements not None, np.ndarray[Integer, ndim=2, mode="c"] edges not None, 
	np.ndarray[Integer, ndim=2, mode="c"] faces not None, Real condition, np.ndarray[Real, ndim=2, mode="c"] boundary_fekete not None,
	bytes projection_method not None):
	"""
	Actual wrapper for occ_frontend, kept as a 'static' function for debugging purposes 
	"""

	cdef Integer element_rows, points_rows, element_cols, points_cols, edges_rows, edges_cols, faces_rows, faces_cols

	elements_rows, elements_cols = elements.shape[0], elements.shape[1]
	points_rows, points_cols = points.shape[0], points.shape[1]
	edges_rows, edges_cols = edges.shape[0], edges.shape[1]
	faces_rows, faces_cols = faces.shape[0], faces.shape[1]

	cdef Integer fekete_rows, fekete_cols
	fekete_rows, fekete_cols = boundary_fekete.shape[0], boundary_fekete.shape[1]


	# AT THE MOMENT THE ARRAYS ARE DEEPLY COPIED FROM CPP TO PYTHON. TO AVOID THIS THE OBJECTS HAVE TO BE 
	# INITIALISED AND ALLOCATED IN PYTHON (CYTHON) THEN PASSED TO CPP. HOWEVER THIS IS ONLY POSSIBLE FOR FIXED
	# ARRAY SIZES. LOOK AT THE END OF THE FILE FOR THESE ALTERNATIVE APPROACHES

	cdef to_python_structs struct_to_python
	# CALL CPP ROUTINE
	struct_to_python = PyCppInterface (<const char*>iges_filename, scale, &points[0,0], points_rows, points_cols, &elements[0,0], 
		elements_rows, elements_cols,&edges[0,0], edges_rows, edges_cols, &faces[0,0], faces_rows, faces_cols, 
		condition, &boundary_fekete[0,0], fekete_rows, fekete_cols,<const char*>projection_method)


	cdef np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
	cdef Integer i 
	for i in range(struct_to_python.nodes_dir_size):
		nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
	cdef np.ndarray displacements_BC = np.zeros((points_cols*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
	for i in range(points_cols*struct_to_python.nodes_dir_size):
		displacements_BC[i] = struct_to_python.displacement_BC_stl[i]


	return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,points_cols) 















# ALTERNATIVE APPROACHES
#-------------------------------------------------------------------------------------------------
# ON ALL THE FOLLOWING APPROACHES NUMPY MUST BE INITIALISED (JUST AFTER NUMPY IMORT)
# np.import_array()
#-------------------------------------------------------------------------------------------------
# 1 - APPROACH 1: COPY-LESS METHOD USING PyArray_SimpleNewFromData AND FREEING THE MEMORY AUTOMATICALLY
# cdef pointer_to_numpy_array_int64(void * ptr, np.npy_intp size):
# # cdef pointer_to_numpy_array_int64(void * ptr, ssize_t size):
# 	'''Convert c pointer to numpy array.
# 	The memory will be freed as soon as the ndarray is deallocated.
# 	'''
# 	cdef extern from "numpy/arrayobject.h":
# 		# void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
# 		void PyArray_ENABLEFLAGS(np.ndarray arr, Integer flags)
# 	cdef np.ndarray[Integer, ndim=1, mode="c"] arr = np.PyArray_SimpleNewFromData(1, &size, np.NPY_INT64, ptr)
# 	PyArray_ENABLEFLAGS(arr, np.NPY_OWNDATA)
# 	return arr

# cdef np.ndarray[Integer] arr= pointer_to_numpy_array_int64(nodes_dir_out, (nodes_dir_out_size,1) )
# cdef np.ndarray[Integer,ndim=1, mode="c"] arr= pointer_to_numpy_array_int64(nodes_dir_out, nodes_dir_out_size )
# print arr
#-------------------------------------------------------------------------------------------------
# 2 - APPROACH 2: COPY-LESS METHOD USING PyArray_SimpleNewFromData AND NOT FREEING THE MEMORY AUTOMATICALLY
# cdef np.npy_intp dimss[2]
# dimss[0]=24; dimss[1]=2
# cdef np.ndarray[Real, ndim=2] arr2 = np.PyArray_SimpleNewFromData(2, dimss, np.NPY_FLOAT64, c_displacement_BC)
#-------------------------------------------------------------------------------------------------
# 3 - APPROACH 3: COPY-LESS METHOD USING np.frombuffer. THE OBJECT (ARRAY) HAS TO BE A PYTHON OJBECT
# cdef np.ndarray[Real,ndim=2] np_displacement_BC = np.asarray(<Real> c_displacement_BC)
# np.frombuffer(nodes_dir_out,nodes_dir_out_size)
#-------------------------------------------------------------------------------------------------
# 4 - APPROACH 4: COPY-LESS METHOD USING CYTHON'S TYPED MEMORY VIEWS. 
# 				  CRASHES ON CYTHON 0.22.1 (SIMILAR BUGS REPORTED ON GITHUB FOR EARLIER VERSIONS OF CYTHON)

# numpy_array = np.asarray(<Integer[:nodes_dir_out_size, :2]> nodes_dir_out)
# numpy_array = np.asarray(<Integer[:nodes_dir_out_size]> nodes_dir_out)
# cdef np.ndarray[Integer] view = <Integer[:nodes_dir_out_size]> nodes_dir_out
#-------------------------------------------------------------------------------------------------
# 5 - APPROACH 5: COPY-LESS METHOD USING NUMPY C-API (FROM WITHIN CPP CODE)
# #include <Python.h>
# #include <numpy/arrayobject.h>
# #include <numpy/npy_common.h>
# PyArrayObject *CreatePyArrayObject(Integer *data, npy_intp size)
# {
# //    _import_array();
#     PyObject *py_array = PyArray_SimpleNewFromData(1,&size,NPY_INT64,data);
#     return (PyArrayObject*)py_array;
# } 
#-------------------------------------------------------------------------------------------------
# NOTE THAT PASSING NUMPY ARRAYS TO C CAN ALSO BE ACCOMPLISHED USING THE THE WELL KNOWN data, 
# HOWEVER THIS IS NOT NUMPY'S RECOMMENDED APPROACH
# struct_to_python = PyCppInterface (<const char*>self.iges_filename, self.scale, 
# 	<Real *>self.points.data, points_rows, points_cols, <Integer *>self.elements.data, 
# 	elements_rows, elements_cols,<Integer *>self.edges.data, edges_rows, edges_cols, <Integer *>self.faces.data, faces_rows, faces_cols, 
# 	self.condition, <Real *>self.boundary_fekete.data, fekete_rows, fekete_cols) 
#-------------------------------------------------------------------------------------------------


# USEFUL LINKS:
# https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
# https://gist.github.com/GaelVaroquaux/1249305
# http://stackoverflow.com/questions/20978377/cython-convert-memory-view-to-numpy-array
# http://codextechnicanum.blogspot.co.uk/2013/12/embedding-python-in-c-converting-c.html
# http://docs.scipy.org/doc/numpy-1.6.0/reference/c-api.array.html#creating-arrays
# http://stackoverflow.com/questions/18780570/passing-a-c-stdvector-to-numpy-array-in-python
