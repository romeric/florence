from cython import double
from libcpp.vector cimport vector
from libcpp.string cimport string


ctypedef long Integer
ctypedef unsigned long UInteger
ctypedef double Real

cdef extern from "PyInterface.hpp":

    struct DirichletData:
        vector[Real] displacement_BC_stl
        vector[Integer] nodes_dir_out_stl
        Integer nodes_dir_size


cdef extern from "PostMeshCurve.hpp":

    cdef cppclass PostMeshCurve:
        PostMeshCurve() except +
        PostMeshCurve(string &element_type, const UInteger &dim) except +
        UInteger ndim
        void Init() except +
        void SetScale(const Real &scale)
        void SetCondition(const Real &condition)
        void SetProjectionPrecision(const Real &precision)
        void SetProjectionCriteria(UInteger *criteria, Integer &rows, Integer &cols)
        void ComputeProjectionCriteria()
        void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
        void SetMeshPoints(Real *arr, Integer &rows, Integer &cols)
        void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
        void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
        void ScaleMesh()
        string GetMeshElementType()
        void SetNodalSpacing(Real *arr, const Integer &rows, const Integer &cols)
        void ReadIGES(const char* filename)
        void GetGeomVertices()
        void GetGeomEdges()
        void GetGeomFaces()
        vector[Real] ObtainGeomVertices()
        Integer NbPoints()
        Integer NbCurves()
        vector[vector[Real]] DiscretiseCurves(Integer npoints) except +
        void GetCurvesParameters()
        void GetCurvesLengths()
        void GetGeomPointsOnCorrespondingEdges()
        void IdentifyCurvesContainingEdges()
        void ProjectMeshOnCurve()
        void RepairDualProjectedParameters()
        void MeshPointInversionCurve()
        void MeshPointInversionCurveArcLength()
        void GetBoundaryPointsOrder()
        void ReturnModifiedMeshPoints(Real *points)
        DirichletData GetDirichletData()


cdef extern from "PostMeshSurface.hpp":

    cdef cppclass PostMeshSurface:
        PostMeshSurface() except +
        PostMeshSurface(string &element_type, const UInteger &dim) except +
        UInteger ndim
        void Init() except +
        void SetScale(Real &scale)
        void SetCondition(Real &condition)
        void SetProjectionPrecision(const Real &precision)
        void SetProjectionCriteria(UInteger *criteria, Integer &rows, Integer &cols)
        void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
        void ComputeProjectionCriteria()
        void SetMeshPoints(Real *arr, Integer &rows, Integer &cols)
        void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
        void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
        void ScaleMesh()
        string GetMeshElementType()
        void SetNodalSpacing(Real *arr, const Integer &rows, const Integer &cols)
        void ReadIGES(const char* filename)
        void GetGeomVertices()
        void GetGeomEdges()
        void GetGeomFaces()
        vector[Real] ObtainGeomVertices()
        Integer NbPoints()
        Integer NbCurves()
        Integer NbSurfaces()
        void GetSurfacesParameters()
        void GetGeomPointsOnCorrespondingFaces()
        void IdentifyRemainingSurfacesByProjection()
        void IdentifySurfacesContainingFacesByPureProjection()
        void IdentifySurfacesContainingFaces()
        void IdentifySurfacesIntersections()
        void SupplySurfacesContainingFaces(const Integer *arr, Integer rows, Integer already_mapped, Integer caller)
        void ProjectMeshOnSurface()
        void RepairDualProjectedParameters()
        void MeshPointInversionSurface(Integer project_on_curves, Integer modify_linear_mesh)
        void MeshPointInversionSurfaceArcLength(Integer project_on_curves, Real OrthTol, Real *FEbases, Integer rows, Integer cols)
        void ReturnModifiedMeshPoints(Real *points)
        vector[vector[Integer]] GetMeshFacesOnPlanarSurfaces()
        vector[Integer] GetDirichletFaces()
        DirichletData GetDirichletData()
        

cdef extern from "PyInterfaceEmulator.hpp": 
    
    DirichletData ComputeDirichleteData (const char* iges_filename, Real scale, 
        Real* points_array, Integer points_rows, Integer points_cols, 
        UInteger* elements_array, const Integer element_rows, const Integer element_cols, 
        UInteger* edges, const Integer edges_rows, const Integer edges_cols,
        UInteger* faces, const Integer faces_rows, const Integer faces_cols, Real condition, 
        Real* boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
        UInteger* criteria, const Integer criteria_rows, const Integer criteria_cols, 
        const Real precision)

    DirichletData ComputeDirichleteData3D (const char* iges_filename, Real scale, 
        Real* points_array, Integer points_rows, Integer points_cols, 
        UInteger* elements_array, const Integer element_rows, const Integer element_cols, 
        UInteger* edges, const Integer edges_rows, const Integer edges_cols,
        UInteger* faces, const Integer faces_rows, const Integer faces_cols, Real condition, 
        Real* boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
        UInteger* criteria, const Integer criteria_rows, const Integer criteria_cols, 
        const Real precision)
