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
ctypedef np.uint64_t UInteger
ctypedef np.float64_t Real 


cdef extern from "OCCPluginInterface.hpp": 
	struct PassToPython:
		vector[Real] displacement_BC_stl
		vector[Integer] nodes_dir_out_stl
		Integer nodes_dir_size

	PassToPython PyCppInterface (const char* iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols, 
		UInteger* elements_array, const Integer element_rows, const Integer element_cols, 
		UInteger* edges, const Integer edges_rows, const Integer edges_cols,
		UInteger* faces, const Integer faces_rows, const Integer faces_cols, Real condition, 
		Real* boundary_fekete, const Integer fekete_rows, const Integer fekete_cols, const char* projection_method)

cdef extern from "OCCPlugin.hpp":
	cdef cppclass OCCPlugin:
		OCCPlugin() except +
		OCCPlugin(string &element_type, const UInteger &dim) except +
		UInteger ndim
		void Init(string &element_type, const UInteger &dim)
		void SetScale(Real &scale)
		void SetCondition(Real &condition)
		void SetDimension(const UInteger &dim)
		void SetMeshElementType(string &type)
		void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
		void SetMeshPoints(Real *arr, Integer &rows, Integer &cols)
		void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
		void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
		void ScaleMesh()
		string GetMeshElementType()
		void SetFeketePoints(Real *arr, const Integer &rows, const Integer &cols)
		void ReadIGES(const char* filename)
		void GetGeomVertices()
		void GetGeomEdges()
		void GetGeomFaces()
		void GetCurvesParameters()
		void GetCurvesLengths()
		void GetGeomPointsOnCorrespondingEdges()
		void IdentifyCurvesContainingEdges()
		void ProjectMeshOnCurve(const char *projection_method)
		void ProjectMeshOnSurface()
		void RepairDualProjectedParameters()
		void MeshPointInversionCurve()
		void MeshPointInversionSurface()
		void GetBoundaryPointsOrder()
		PassToPython GetDirichletData()



cdef class OCCPluginPy2:

	cdef UInteger ndim 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
	cdef Real[:,:] boundary_fekete 

	cdef OCCPlugin *thisptr

	def __cinit__(self, bytes py_element_type, UInteger dimension=2):
		# Convert to cpp string explicitly
		cdef string cpp_element_type = py_element_type
		# self.thisptr = new OCCPlugin()
		self.thisptr = new OCCPlugin(cpp_element_type,dimension)

	def Init(self, bytes py_element_type, UInteger dimension):
		cdef string cpp_element_type = py_element_type
		self.thisptr.Init(cpp_element_type,dimension)

	def SetScale(self,Real scale):
		self.thisptr.SetScale(scale)

	def SetCondition(self,Real condition):
		self.thisptr.SetCondition(condition)

	def SetDimension(self,Integer dimension):
		self.thisptr.SetDimension(dimension)

	def SetMeshElementType(self,bytes py_element_type):
		cdef string cpp_element_type = py_element_type
		self.thisptr.SetMeshElementType(cpp_element_type) 

	def SetMeshElements(self,UInteger[:,::1] elements):
		self.thisptr.SetMeshElements(&elements[0,0],elements.shape[0],elements.shape[1])
	
	def SetMeshPoints(self,Real[:,::1] points):
		self.thisptr.SetMeshPoints(&points[0,0],points.shape[0],points.shape[1])
	
	def SetMeshEdges(self,UInteger[:,::1] edges):
		self.thisptr.SetMeshEdges(&edges[0,0],edges.shape[0],edges.shape[1])

	def SetMeshFaces(self,UInteger[:,::1] faces):
		self.thisptr.SetMeshFaces(&faces[0,0],faces.shape[0],faces.shape[1])

	def ScaleMesh(self):
		self.thisptr.ScaleMesh()

	def GetMeshElementType(self):
		cdef string cpp_element_type = self.thisptr.GetMeshElementType()
		cdef bytes py_element_type = cpp_element_type
		return py_element_type

	def SetFeketePoints(self, Real[:,::1] fekete):
		self.thisptr.SetFeketePoints(&fekete[0,0],fekete.shape[0],fekete.shape[1])

	def ReadIGES(self, bytes filename):
		self.thisptr.ReadIGES(<const char*>filename)

	def GetGeomVertices(self):
		self.thisptr.GetGeomVertices()

	def GetGeomEdges(self):
		self.thisptr.GetGeomEdges()

	def GetGeomFaces(self):
		self.thisptr.GetGeomFaces()

	def GetCurvesParameters(self):
		self.thisptr.GetCurvesParameters()

	def GetCurvesLengths(self):
		self.thisptr.GetCurvesLengths()

	def GetGeomPointsOnCorrespondingEdges(self):
		self.thisptr.GetGeomPointsOnCorrespondingEdges()

	def IdentifyCurvesContainingEdges(self):
		self.thisptr.IdentifyCurvesContainingEdges()

	def ProjectMeshOnCurve(self, bytes projection_method):
		self.thisptr.ProjectMeshOnCurve(<const char *> projection_method)
	
	def ProjectMeshOnSurface(self):
		self.thisptr.ProjectMeshOnSurface()

	def RepairDualProjectedParameters(self):
		self.thisptr.RepairDualProjectedParameters()

	def MeshPointInversionCurve(self):
		self.thisptr.MeshPointInversionCurve()

	def MeshPointInversionSurface(self):
		self.thisptr.MeshPointInversionSurface()

	def GetBoundaryPointsOrder(self):
		self.thisptr.GetBoundaryPointsOrder()

	def GetDirichletData(self):
		cdef PassToPython struct_to_python = self.thisptr.GetDirichletData()
		cdef np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
		cdef Integer i 
		for i in range(struct_to_python.nodes_dir_size):
			nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
		cdef np.ndarray displacements_BC = np.zeros((self.thisptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
		for i in range(self.thisptr.ndim*struct_to_python.nodes_dir_size):
			displacements_BC[i] = struct_to_python.displacement_BC_stl[i]

		return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.thisptr.ndim) 


	def __dealloc__(self):
		del self.thisptr




cdef class OCCPluginPy:
	"""
		Cython wrapper for OCCPlugin
	"""

	cdef bytes iges_filename
	cdef UInteger ndim 
	cdef Real scale 
	cdef Real condition 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
	cdef Real[:,:] points 
	cdef UInteger[:,:] elements
	cdef UInteger[:,:] edges 
	cdef UInteger[:,:] faces 
	cdef Real[:,:] boundary_fekete 
	cdef bytes projection_method

	def __init__(self,dimension=2):
		self.ndim = dimension

	def SetMesh(self,np.ndarray[Real, ndim=2, mode="c"] points, np.ndarray[UInteger, ndim=2, mode="c"] elements,
		np.ndarray[UInteger, ndim=2, mode="c"] edges, np.ndarray[UInteger, ndim=2, mode="c"] faces=None):

		self.elements = elements 
		self.points = points
		self.edges = edges 
		self.faces = faces 

		return self

	def SetCADGeometry(self,bytes iges_filename):
		self.iges_filename = iges_filename

	def SetDimension(self,UInteger dim):
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
	np.ndarray[UInteger, ndim=2, mode="c"] elements not None, np.ndarray[UInteger, ndim=2, mode="c"] edges not None, 
	np.ndarray[UInteger, ndim=2, mode="c"] faces not None, Real condition, np.ndarray[Real, ndim=2, mode="c"] boundary_fekete not None,
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

	cdef PassToPython struct_to_python
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
