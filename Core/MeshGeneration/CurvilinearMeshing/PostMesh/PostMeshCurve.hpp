#ifndef PostMeshCurve_HPP
#define PostMeshCurve_HPP

#include <PostMeshBase.hpp>


// BASE CLASS FOR OPENCASCADE FRONT-END
class PostMeshCurve: public PostMeshBase
{
    friend class PostMeshSurface;

public:

    ALWAYS_INLINE PostMeshCurve() : PostMeshBase()
    {
        this->ndim = 2;
        this->mesh_element_type = "tri";
    }

    ALWAYS_INLINE PostMeshCurve(std::string &element_type, const UInteger &dim) \
        : PostMeshBase(element_type,dim){}

    PostMeshCurve(const PostMeshCurve& other) \
    noexcept(std::is_copy_constructible<PostMeshCurve>::value);
    PostMeshCurve& operator=(const PostMeshCurve& other) \
    noexcept(std::is_copy_assignable<PostMeshCurve>::value);
    PostMeshCurve(PostMeshCurve&& other) noexcept;
    PostMeshCurve& operator=(PostMeshCurve&& other) noexcept;
    ALWAYS_INLINE ~PostMeshCurve(){}

    ALWAYS_INLINE void Init()
    {
        this->mesh_element_type = "tri";
        this->ndim = 2;
        this->scale = 1.0;
        this->condition = 1.0e10;
        this->projection_precision = 1.0e-4;
    }

    void InferInterpolationPolynomialDegree();
    void CurvesToBsplineCurves();
    void GetCurvesParameters();
    void GetCurvesLengths();
    void GetGeomPointsOnCorrespondingEdges();
    void IdentifyCurvesContainingEdges();
    void ProjectMeshOnCurve(const char *projection_method);
    void RepairDualProjectedParameters();
    void MeshPointInversionCurve();    
    void GetBoundaryPointsOrder();
    PassToPython GetDirichletData();


    // Public data members of PostMeshCurve
    std::vector<Eigen::MatrixR> geometry_points_on_curves;
    std::vector<Handle_Geom_BSplineCurve> geometry_curves_bspline;
    Eigen::MatrixI boundary_points_order;
    Eigen::MatrixI boundary_edges_order;
    Eigen::MatrixR curve_to_parameter_scale_U;
    Eigen::MatrixR curves_parameters;
    Eigen::MatrixR curves_lengths;


protected:
    void FindCurvesSequentiallity();
    void ConcatenateSequentialCurves();
    void GetInternalCurveScale();
    Eigen::MatrixR ParametricFeketePoints(Standard_Real &u1, Standard_Real &u2);
    void GetElementsWithBoundaryEdgesTri();
    void EstimatedParameterUOnMesh();


    Eigen::MatrixR projection_U;
    Eigen::MatrixI sorted_projected_indices;
    Eigen::MatrixI dirichlet_edges;
    std::vector<Integer> listedges;
    Standard_Integer no_dir_edges;
    Eigen::MatrixR u_of_all_fekete_mesh_edges;
    Eigen::MatrixI elements_with_boundary_edges;
};


#endif // PostMeshCurve_HPP

