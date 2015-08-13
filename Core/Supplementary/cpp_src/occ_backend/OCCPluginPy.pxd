from cython import double
from libcpp.vector cimport vector
from libcpp.string cimport string


ctypedef long Integer
ctypedef unsigned long UInteger
ctypedef double Real


cdef extern from "OCCPlugin.hpp":

	struct PassToPython:
		vector[Real] displacement_BC_stl
		vector[Integer] nodes_dir_out_stl
		Integer nodes_dir_size
	
	cdef cppclass OCCPlugin:
		OCCPlugin() except +
		OCCPlugin(string &element_type, const UInteger &dim) except +
		UInteger ndim
		void Init(string &element_type, const UInteger &dim)
		void SetScale(Real &scale)
		void SetCondition(Real &condition)
		void SetProjectionCriteria(UInteger *criteria, Integer &rows, Integer &cols)
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

		

cdef extern from "OCCPluginInterface.hpp": 
	
	PassToPython PyCppInterface (const char* iges_filename, Real scale, Real* points_array, Integer points_rows, Integer points_cols, 
		UInteger* elements_array, const Integer element_rows, const Integer element_cols, 
		UInteger* edges, const Integer edges_rows, const Integer edges_cols,
		UInteger* faces, const Integer faces_rows, const Integer faces_cols, Real condition, 
		Real* boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
		UInteger* criteria, const Integer criteria_rows, const Integer criteria_cols, const char* projection_method)