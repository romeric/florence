#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

from cython import boundscheck, nonecheck, wraparound, profile, double
import numpy as np
cimport numpy as np


cdef class PostMeshCurvePy:
	"""
	Python wrapper for C++ PostMesh 
	"""

	cdef UInteger ndim 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE/CLASS LEVEL
	cdef Real[:,:] boundary_fekete 

	cdef PostMeshCurve *thisptr

	def __cinit__(self, bytes py_element_type, UInteger dimension=2):
		# Convert to cpp string explicitly
		cdef string cpp_element_type = py_element_type
		# self.thisptr = new OCCPlugin()
		self.thisptr = new PostMeshCurve(cpp_element_type,dimension)

	def Init(self):
		self.thisptr.Init()

	def SetScale(self,Real scale):
		self.thisptr.SetScale(scale)

	def SetCondition(self,Real condition):
		self.thisptr.SetCondition(condition)

	def SetProjectionPrecision(self, Real precision):
		self.thisptr.SetProjectionPrecision(precision)

	def SetProjectionCriteria(self, UInteger[:,::1] criteria):
		self.thisptr.SetProjectionCriteria(&criteria[0,0],criteria.shape[0],criteria.shape[1])

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
	
	# def ProjectMeshOnSurface(self):
		# self.thisptr.ProjectMeshOnSurface()

	def RepairDualProjectedParameters(self):
		self.thisptr.RepairDualProjectedParameters()

	def MeshPointInversionCurve(self):
		self.thisptr.MeshPointInversionCurve()

	# def MeshPointInversionSurface(self):
		# self.thisptr.MeshPointInversionSurface()

	def GetBoundaryPointsOrder(self):
		self.thisptr.GetBoundaryPointsOrder()

	@boundscheck(False)
	def GetDirichletData(self):
		cdef: 
			PassToPython struct_to_python = self.thisptr.GetDirichletData()
			np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
			Integer i
			UInteger j 

		for i in range(struct_to_python.nodes_dir_size):
			nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
			
		cdef np.ndarray displacements_BC = np.zeros((self.thisptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
		for j in range(self.thisptr.ndim*struct_to_python.nodes_dir_size):
			displacements_BC[j] = struct_to_python.displacement_BC_stl[j]

		return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.thisptr.ndim) 


	def __dealloc__(self):
		if self.thisptr != NULL:
			del self.thisptr




cdef class PostMeshSurfacePy:
	"""
	Python wrapper for C++ PostMesh 
	"""

	cdef UInteger ndim 
	# USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
	# ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
	cdef Real[:,:] boundary_fekete 

	cdef PostMeshSurface *thisptr

	def __cinit__(self, bytes py_element_type, UInteger dimension=2):
		# Convert to cpp string explicitly
		cdef string cpp_element_type = py_element_type
		self.thisptr = new PostMeshSurface(cpp_element_type,dimension)

	def Init(self):
		self.thisptr.Init()

	def SetScale(self,Real scale):
		self.thisptr.SetScale(scale)

	def SetCondition(self,Real condition):
		self.thisptr.SetCondition(condition)

	def SetProjectionPrecision(self, Real precision):
		self.thisptr.SetProjectionPrecision(precision)

	def SetProjectionCriteria(self, UInteger[:,::1] criteria):
		self.thisptr.SetProjectionCriteria(&criteria[0,0],criteria.shape[0],criteria.shape[1])

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

	def GetSurfacesParameters(self):
		self.thisptr.GetSurfacesParameters()

	def GetGeomPointsOnCorrespondingFaces(self):
		self.thisptr.GetGeomPointsOnCorrespondingFaces()

	def IdentifySurfacesContainingFaces(self):
		self.thisptr.IdentifySurfacesContainingFaces()

	def ProjectMeshOnSurface(self, bytes projection_method):
		self.thisptr.ProjectMeshOnSurface(<const char *> projection_method)

	# def RepairDualProjectedParameters(self):
	# 	self.thisptr.RepairDualProjectedParameters()

	def MeshPointInversionSurface(self):
		self.thisptr.MeshPointInversionSurface()

	# def MeshPointInversionSurface(self):
		# self.thisptr.MeshPointInversionSurface()

	# def GetBoundaryPointsOrder(self):
	# 	self.thisptr.GetBoundaryPointsOrder()

	@boundscheck(False)
	def GetDirichletData(self):
		cdef: 
			PassToPython struct_to_python = self.thisptr.GetDirichletData()
			np.ndarray nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
			Integer i
			UInteger j 

		for i in range(struct_to_python.nodes_dir_size):
			nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
			
		cdef np.ndarray displacements_BC = np.zeros((self.thisptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
		for j in range(self.thisptr.ndim*struct_to_python.nodes_dir_size):
			displacements_BC[j] = struct_to_python.displacement_BC_stl[j]

		return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.thisptr.ndim) 


	def __dealloc__(self):
		if self.thisptr != NULL:
			del self.thisptr			