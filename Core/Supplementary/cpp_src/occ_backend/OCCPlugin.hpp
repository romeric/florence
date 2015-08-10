#ifndef OCCPlugin_HPP
#define OCCPlugin_HPP


#include <EIGEN_INC.hpp>
#include <OCC_INC.hpp>
#include <CNPFuncs.hpp>
#include <AuxFuncs.hpp>


// Struct to send to Python
struct PassToPython
{
    std::vector<Real> displacement_BC_stl;
    std::vector<Integer> nodes_dir_out_stl;
    Integer nodes_dir_size;
};


// BASE CLASS FOR OPENCASCADE FRONT-END
class OCCPlugin
{
public:
    // CONSTRUCTOR
    OCCPlugin():  mesh_element_type("tri"), ndim(2) {}

    OCCPlugin(std::string &element_type, const UInteger &dim) : mesh_element_type(element_type), ndim(dim) {
//        this->mesh_element_type = element_type;
//        this->ndim = dim;
        this->condition = 1.0e10;
        this->scale = 1.;
    }

    ~OCCPlugin(){}

    // Public members of OCCPlugin
//    Eigen::Map<Eigen::MatrixI> kkk;

    std::string mesh_element_type;
    UInteger ndim;
    UInteger degree;
    Eigen::MatrixUI mesh_elements;
//    Eigen::Map<Eigen::MatrixI> mesh_elements;
    Eigen::MatrixR mesh_points;
    Eigen::MatrixUI mesh_edges;
    Eigen::MatrixUI mesh_faces;
    TopoDS_Shape imported_shape;
    UInteger no_of_shapes;
    std::vector<gp_Pnt> geometry_points;
    std::vector<Handle_Geom_Curve> geometry_curves;
    std::vector<Handle_Geom_Surface> geometry_surfaces;
    std::vector<Eigen::MatrixR> geometry_points_on_curves;
    std::vector<Eigen::MatrixR> geometry_points_on_surfaces;
    std::vector<UInteger> geometry_curves_types;
    std::vector<UInteger> geometry_surfaces_types;
    std::vector<Handle_Geom_BSplineCurve> geometry_curves_bspline;
    std::vector<Handle_Geom_BSplineSurface> geometry_surfaces_bspline;
    Real condition;
    Real scale;
    Eigen::MatrixR displacements_BC;
    Eigen::MatrixI index_nodes;
    Eigen::MatrixI nodes_dir;
    const char *projection_method;
    Eigen::MatrixR fekete;
    Eigen::MatrixI boundary_points_order;
    Eigen::MatrixI boundary_edges_order;
    Eigen::MatrixR curve_to_parameter_scale_U;
    Eigen::MatrixR curves_parameters;
    Eigen::MatrixR curves_lengths;

    // Public member functions
    void Init(std::string &element_type, const UInteger &ndim);
    void SetScale(Real &scale);
    void SetCondition(Real &condition);
    void SetDimension(const UInteger &dim);
    void SetMeshElementType(std::string &type);
    void SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols);
    void SetMeshPoints(Real *arr, const Integer &rows, const Integer &cols);
    void SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols);
    void SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols);
    void ScaleMesh();
    std::string GetMeshElementType();
    void ReadMeshConnectivityFile(std::string &filename, char delim);
    void ReadMeshCoordinateFile(std::string &filename, char delim);
    void ReadMeshEdgesFile(std::string &filename, char delim);
    void ReadMeshFacesFile(std::string &filename, char delim);
    void InferInterpolationPolynomialDegree();
    static Eigen::MatrixI Read(std::string &filename);
    static Eigen::MatrixUI ReadI(std::string &filename, char delim);
    static Eigen::MatrixR ReadR(std::string &filename, char delim);
    void CheckMesh();
    void SetFeketePoints(Real *arr, const Integer &rows, const Integer &cols);
    void ReadIGES(const char *filename);
    void GetGeomVertices();
    void GetGeomEdges();
    void GetGeomFaces();
    void GetCurvesParameters();
    void GetCurvesLengths();
    void GetGeomPointsOnCorrespondingEdges();
    void IdentifyCurvesContainingEdges();
    void ProjectMeshOnCurve(const char *projection_method);
    void ProjectMeshOnSurface();
    void RepairDualProjectedParameters();
    void MeshPointInversionCurve();
    void MeshPointInversionSurface();
    void GetBoundaryPointsOrder();
    PassToPython GetDirichletData();



private:
    Eigen::MatrixI projection_ID;
    Eigen::MatrixR projection_U;
    Eigen::MatrixR projection_V;
    Eigen::MatrixI sorted_projected_indices;
    Eigen::MatrixI dirichlet_edges;
    Eigen::MatrixI dirichlet_faces;
    std::vector<Integer> listedges;
    std::vector<Integer> listfaces;
    Standard_Integer no_dir_edges;
    Standard_Integer no_dir_faces;
    Eigen::MatrixI unique_edges;
    Eigen::MatrixI unique_faces;
    Eigen::MatrixR u_of_all_fekete_mesh_edges;
    Eigen::MatrixI elements_with_boundary_edges;

    void FindCurvesSequentiallity();
    void ConcatenateSequentialCurves();
    void GetInternalCurveScale();
    void GetInternalSurfaceScales();
    void CurvesToBsplineCurves();
    void SurfacesToBsplineSurfaces();
    Eigen::MatrixR ParametricFeketePoints(Standard_Real &u1, Standard_Real &u2);
    void GetElementsWithBoundaryEdgesTri();
    void EstimatedParameterUOnMesh();
};


#endif // OCCPlugin_HPP

