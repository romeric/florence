#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

from cython import boundscheck, nonecheck, wraparound, profile, double
import numpy as np
cimport numpy as np


cdef class PostMeshCurvePy:
    """
    Python wrapper for C++ PostMeshCurve 
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
        self.ndim = 2

    def Init(self):
        self.thisptr.Init()

    def SetScale(self,Real scale):
        self.thisptr.SetScale(scale)

    def SetCondition(self,Real condition):
        self.thisptr.SetCondition(condition)

    def SetProjectionPrecision(self, Real precision):
        self.thisptr.SetProjectionPrecision(precision)

    def ComputeProjectionCriteria(self):
        self.thisptr.ComputeProjectionCriteria()

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

    def SetNodalSpacing(self, Real[:,::1] fekete):
        self.thisptr.SetNodalSpacing(&fekete[0,0],fekete.shape[0],fekete.shape[1])

    def ReadIGES(self, bytes filename):
        self.thisptr.ReadIGES(<const char*>filename)

    def GetGeomVertices(self):
        self.thisptr.GetGeomVertices()
        cdef vector[Real] geom_points = self.thisptr.ObtainGeomVertices()
        cdef np.ndarray geometry_points = np.array(geom_points,copy=False)
        return geometry_points.reshape(int(geometry_points.shape[0]/self.ndim),self.ndim)

    def GetGeomEdges(self):
        self.thisptr.GetGeomEdges()

    def GetGeomFaces(self):
        self.thisptr.GetGeomFaces()

    def NbPoints(self):
        return self.thisptr.NbPoints()

    def NbCurves(self):
        return self.thisptr.NbCurves()

    def DiscretiseCurves(self,Integer npoints):
        cdef vector[vector[Real]] discretised_points
        discretised_points = self.thisptr.DiscretiseCurves(npoints)
        discretised_points_py = []
        for i in range(len(discretised_points)):
            discretised_points_py.append(np.array(discretised_points[i]).reshape(len(discretised_points[0])/3,3))
        return discretised_points_py

    def GetCurvesParameters(self):
        self.thisptr.GetCurvesParameters()

    def GetCurvesLengths(self):
        self.thisptr.GetCurvesLengths()

    def GetGeomPointsOnCorrespondingEdges(self):
        self.thisptr.GetGeomPointsOnCorrespondingEdges()

    def IdentifyCurvesContainingEdges(self):
        self.thisptr.IdentifyCurvesContainingEdges()

    def ProjectMeshOnCurve(self):
        self.thisptr.ProjectMeshOnCurve()
    
    # def ProjectMeshOnSurface(self):
        # self.thisptr.ProjectMeshOnSurface()

    def RepairDualProjectedParameters(self):
        self.thisptr.RepairDualProjectedParameters()

    def MeshPointInversionCurve(self):
        self.thisptr.MeshPointInversionCurve()

    def MeshPointInversionCurveArcLength(self):
        self.thisptr.MeshPointInversionCurveArcLength()

    def GetBoundaryPointsOrder(self):
        self.thisptr.GetBoundaryPointsOrder()

    def ReturnModifiedMeshPoints(self,Real[:,::1] points):
        self.thisptr.ReturnModifiedMeshPoints(&points[0,0])

    @boundscheck(False)
    def GetDirichletData(self):
        cdef: 
            DirichletData struct_to_python = self.thisptr.GetDirichletData()
            np.ndarray[np.int64_t, ndim=2, mode='c'] nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
            Integer i
            UInteger j 

        for i in range(struct_to_python.nodes_dir_size):
            nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
            
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] displacements_BC = \
            np.zeros((self.thisptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
        for j in range(self.thisptr.ndim*struct_to_python.nodes_dir_size):
            displacements_BC[j] = struct_to_python.displacement_BC_stl[j]

        return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.thisptr.ndim) 


    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr




cdef class PostMeshSurfacePy:
    """
    Python wrapper for C++ PostMeshSurface 
    """

    cdef UInteger ndim 
    # USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
    # ARRAYS CANNOT BE DECLARED AT MODULE LEVEL
    cdef Real[:,:] boundary_fekete 

    cdef PostMeshSurface *thisptr

    def __cinit__(self, bytes py_element_type, UInteger dimension=3):
        # Convert to cpp string explicitly
        cdef string cpp_element_type = py_element_type
        self.thisptr = new PostMeshSurface(cpp_element_type,dimension)
        self.ndim = 3

    def Init(self):
        self.thisptr.Init()

    def SetScale(self,Real scale):
        self.thisptr.SetScale(scale)

    def SetCondition(self,Real condition):
        self.thisptr.SetCondition(condition)

    def SetProjectionPrecision(self, Real precision):
        self.thisptr.SetProjectionPrecision(precision)

    def ComputeProjectionCriteria(self):
        self.thisptr.ComputeProjectionCriteria()

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

    def SetNodalSpacing(self, Real[:,::1] fekete):
        self.thisptr.SetNodalSpacing(&fekete[0,0],fekete.shape[0],fekete.shape[1])

    def ReadIGES(self, bytes filename):
        self.thisptr.ReadIGES(<const char*>filename)

    def GetGeomVertices(self):
        self.thisptr.GetGeomVertices()
        cdef vector[Real] geom_points = self.thisptr.ObtainGeomVertices()
        cdef np.ndarray geometry_points = np.array(geom_points,copy=False)
        return geometry_points.reshape(int(geometry_points.shape[0]/self.ndim),self.ndim)

    def GetGeomEdges(self):
        self.thisptr.GetGeomEdges()

    def GetGeomFaces(self):
        self.thisptr.GetGeomFaces()

    def NbPoints(self):
        return self.thisptr.NbPoints()

    def NbCurves(self):
        return self.thisptr.NbCurves()

    def NbSurfaces(self):
        return self.thisptr.NbSurfaces()

    def GetSurfacesParameters(self):
        self.thisptr.GetSurfacesParameters()

    def GetGeomPointsOnCorrespondingFaces(self):
        self.thisptr.GetGeomPointsOnCorrespondingFaces()

    def IdentifySurfacesContainingFaces(self):
        self.thisptr.IdentifySurfacesContainingFaces()

    def ProjectMeshOnSurface(self):
        self.thisptr.ProjectMeshOnSurface()

    # def RepairDualProjectedParameters(self):
    #   self.thisptr.RepairDualProjectedParameters()

    def MeshPointInversionSurface(self):
        self.thisptr.MeshPointInversionSurface()

    # def GetBoundaryPointsOrder(self):
    #   self.thisptr.GetBoundaryPointsOrder()

    def ReturnModifiedMeshPoints(self,Real[:,::1] points):
        self.thisptr.ReturnModifiedMeshPoints(&points[0,0])

    @boundscheck(False)
    def GetDirichletData(self):
        cdef: 
            DirichletData struct_to_python = self.thisptr.GetDirichletData()
            np.ndarray[np.int64_t, ndim=2, mode='c'] nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
            Integer i
            UInteger j 

        for i in range(struct_to_python.nodes_dir_size):
            nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
            
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] displacements_BC = \
                np.zeros((self.thisptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
        for j in range(self.thisptr.ndim*struct_to_python.nodes_dir_size):
            displacements_BC[j] = struct_to_python.displacement_BC_stl[j]

        return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.thisptr.ndim) 


    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr            