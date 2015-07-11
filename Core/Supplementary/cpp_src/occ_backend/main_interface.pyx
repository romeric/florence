"""
main interface between python and cpp's occ_frontend 
"""

import cython
import ctypes
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

# numpy array must be initialised
np.import_array()
# np._import_array()

ctypedef np.int64_t Integer
ctypedef np.float64_t Real 


cdef extern from "py_to_occ_frontend.hpp": 
	struct to_python_structs:
		vector[Real] displacement_BC_stl
		vector[Integer] nodes_dir_out_stl
		Integer nodes_dir_size

	to_python_structs py_cpp_interface (Real* points_array, Integer points_rows, Integer points_cols, 
		Integer* elements_array, Integer element_rows, Integer element_cols,
		Integer* edges, Integer edges_rows, Integer edges_cols, Integer* faces, Integer faces_rows, Integer faces_cols, 
		Real* c_displacement_BC, Integer *nodes_dir_out, Integer &nodes_dir_out_size)



@cython.boundscheck(False)
@cython.wraparound(False)
def main_interface(np.ndarray[Real, ndim=2, mode="c"] points not None, np.ndarray[Integer, ndim=2, mode="c"] elements not None,
	np.ndarray[Integer, ndim=2, mode="c"] edges not None, np.ndarray[Integer, ndim=2, mode="c"] faces not None):
	"""
	wrapper for occ_frontend
	"""

	cdef Integer element_rows, points_rows, element_cols, points_cols, edges_rows, edges_cols, faces_rows, faces_cols

	elements_rows, elements_cols = elements.shape[0], elements.shape[1]
	points_rows, points_cols = points.shape[0], points.shape[1]
	edges_rows, edges_cols = edges.shape[0], edges.shape[1]
	faces_rows, faces_cols = faces.shape[0], faces.shape[1]

	cdef Integer *nodes_dir_out
	cdef Integer nodes_dir_out_size
	cdef Real *c_displacement_BC

	cdef to_python_structs struct_to_python

	# AT THE MOMENT THE ARRAYS ARE DEEPLY COPIED FROM CPP TO PYTHON. TO AVOID THIS THE OBJECTS HAVE TO BE 
	# INITIALISED AND ALLOCATED IN PYTHON (CYTHON) THEN PASSED TO CPP. HOWEVER THIS IS ONLY POSSIBLE FOR FIXED
	# ARRAY SIZES. LOOK AT THE END OF THE FILE FOR THESE ALTERNATIVE APPROACHES 

	struct_to_python = py_cpp_interface (&points[0,0], points_rows, points_cols, &elements[0,0], 
		elements_rows, elements_cols,&edges[0,0], 
		edges_rows, edges_cols, &faces[0,0], faces_rows, faces_cols,
		c_displacement_BC, nodes_dir_out, nodes_dir_out_size)

	cdef np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
	cdef Integer i 
	for i in range(struct_to_python.nodes_dir_size):
		nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
	cdef np.ndarray displacements_BC = np.zeros((points_cols*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
	for i in range(points_cols*struct_to_python.nodes_dir_size):
		displacements_BC[i] = struct_to_python.displacement_BC_stl[i]

	cdef np.npy_intp dims[1]
	dims[0]=24
	cdef np.ndarray[Integer, ndim=1, mode="c"] arr = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT64, nodes_dir_out)


	return nodes_dir , displacements_BC.reshape(struct_to_python.nodes_dir_size,points_cols) 





# ALTERNATIVE APPROACHES
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

# USEFUL LINKS:
# https://gist.github.com/GaelVaroquaux/1249305
# http://stackoverflow.com/questions/20978377/cython-convert-memory-view-to-numpy-array
# http://codextechnicanum.blogspot.co.uk/2013/12/embedding-python-in-c-converting-c.html
# http://docs.scipy.org/doc/numpy-1.6.0/reference/c-api.array.html#creating-arrays
# http://stackoverflow.com/questions/18780570/passing-a-c-stdvector-to-numpy-array-in-python
