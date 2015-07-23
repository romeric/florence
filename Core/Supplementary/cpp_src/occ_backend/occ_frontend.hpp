#ifndef OCC_FRONTEND_HPP
#define OCC_FRONTEND_HPP


#include <eigen_inc.hpp>
#include <occ_inc.hpp>
#include <cnp_funcs.hpp>
#include <aux_funcs.hpp>



// OCC_FrontEnd CLASS
class OCC_FrontEnd
{
public:
    // CONSTRUCTOR
    OCC_FrontEnd():  mesh_element_type("tri"), ndim(2) {}

    OCC_FrontEnd(std::string &element_type,Integer &dim) : mesh_element_type(element_type), ndim(dim) {
//        this->mesh_element_type = element_type;
//        this->ndim = dim;
        this->condition = 1.0e10;
        this->scale = 1000.;
    }

    ~OCC_FrontEnd(){}

    // members of OCC_FrontEnd
    std::string mesh_element_type;
    Integer ndim;
//    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_elements;
    Eigen::MatrixI mesh_elements;
    Eigen::MatrixR mesh_points;
    Eigen::MatrixI mesh_edges;
    Eigen::MatrixI mesh_faces;
    TopoDS_Shape imported_shape;
    Standard_Integer no_of_shapes;
    std::vector<gp_Pnt> geometry_points;
    std::vector<Handle_Geom_Curve> geometry_curves;
    std::vector<Handle_Geom_Surface> geometry_surfaces;
    Eigen::MatrixR geometry_points_on_curves;
    Eigen::MatrixR geometry_points_on_surfaces;
    std::vector<Integer> geometry_curves_types;
    std::vector<Integer> geometry_surfaces_types;
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
    Integer degree;
    Eigen::MatrixR curve_to_parameter_scale_U;


    // methods of occ_backend
    void Init(std::string &element_type,Integer &ndim);
    void SetScale(Real &scale);
    void SetCondition(Real &condition);
    void SetDimension(Integer &dim);
    void SetMeshElementType(std::string &type);
    void SetMeshElements(Eigen::MatrixI &arr);
    void SetMeshPoints(Eigen::MatrixR &arr);
    void SetMeshEdges(Eigen::MatrixI &arr);
    void SetMeshFaces(Eigen::MatrixI &arr);
    void ScaleMesh();
    std::string GetMeshElementType();
    void ReadMeshConnectivityFile(std::string &filename, char delim);
    void ReadMeshCoordinateFile(std::string &filename, char delim);
    void ReadMeshEdgesFile(std::string &filename, char delim);
    void ReadMeshFacesFile(std::string &filename, char delim);
    void ReadUniqueEdges(std::string &filename);
    void ReadUniqueFaces(std::string &filename);
    void SetUniqueEdges(Eigen::MatrixI &un_arr);
    void SetUniqueFaces(Eigen::MatrixI &un_arr);
    void InferInterpolationPolynomialDegree();
    static Eigen::MatrixI Read(std::string &filename);
    static Eigen::MatrixI ReadI(std::string &filename, char delim);
    static Eigen::MatrixR ReadR(std::string &filename, char delim);
    void CheckMesh();
    void ReadIGES(const char *filename);
    void GetGeomVertices();
    void GetGeomEdges();
    void GetGeomFaces();
    void GetGeomPointsOnCorrespondingEdges();
    void GetInternalCurveScale();
    void GetInternalSurfaceScales();
    void IdentifyCurvesContainingEdges();
    void ProjectMeshOnCurve(const char *projection_method);
    void ProjectMeshOnCurve_Old(const char *projection_method);
    void ProjectMeshOnSurface();
    void RepairDualProjectedParameters();
    void RepairDualProjectedParameters_Old();
    void CurvesToBsplineCurves();
    void SurfacesToBsplineSurfaces();
    void MeshPointInversionCurve();
    void MeshPointInversionSurface();
    void SetFeketePoints(Eigen::MatrixR &boundary_fekete);
    Eigen::MatrixR ParametricFeketePoints(Handle_Geom_Curve &curve,Standard_Real &u1,Standard_Real &u2);
    void GetElementsWithBoundaryEdgesTri();
    void GetBoundaryPointsOrder();
    void EstimatedParameterUOnMesh();


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
};




#endif // OCC_FRONTEND_HPP

