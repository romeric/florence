#ifndef POSTMESHSURFACE_H
#define POSTMESHSURFACE_H

#include <PostMeshBase.hpp>
#include <PostMeshCurve.hpp>

class PostMeshSurface: public PostMeshBase
{

public:
    PostMeshSurface() : PostMeshBase()
    {
        this->ndim = 3;
        this->mesh_element_type = "tet";
    }

    PostMeshSurface(std::string &element_type, const UInteger &dim) : \
        PostMeshBase(element_type,dim){}

    PostMeshSurface(const PostMeshSurface& other) \
    noexcept(std::is_copy_constructible<PostMeshSurface>::value);
    PostMeshSurface& operator=(const PostMeshSurface& other) \
    noexcept(std::is_copy_assignable<PostMeshSurface>::value);
    PostMeshSurface(PostMeshSurface&& other) noexcept;
    PostMeshSurface& operator=(PostMeshSurface&& other) noexcept;
    ~PostMeshSurface(){}

    ALWAYS_INLINE void Init()
    {
        this->ndim = 3;
        this->mesh_element_type = "tet";
        this->scale = 1.0;
        this->condition = 1.0e10;
    }

    void InferInterpolationPolynomialDegree();
    void SurfacesToBsplineSurfaces();
    void GetSurfacesParameters();
    void GetGeomPointsOnCorrespondingFaces();
    void IdentifySurfacesContainingFacesByPureProjection();
    void IdentifyRemainingSurfacesByProjection();
    void IdentifySurfacesContainingFaces();
    void SupplySurfacesContainingFaces(const Integer *arr, Integer rows, Integer already_mapped = 0, Integer caller = 0);
    void IdentifySurfacesIntersections();
    void ProjectMeshOnSurface();
    void RepairDualProjectedParameters();
    void MeshPointInversionCurve(const gp_Pnt &point_in, gp_Pnt &point_out);
    void MeshPointInversionSurface(Integer project_on_curves, Integer modify_linear_mesh = 0);
    void MeshPointInversionSurfaceArcLength(Integer project_on_curves, Real OrthTol, Real *FEbases, Integer rows, Integer cols);
    void GetBoundaryPointsOrder();
    std::vector<std::vector<Integer> > GetMeshFacesOnPlanarSurfaces();
    std::vector<Integer> GetDirichletFaces();


    std::vector<Eigen::MatrixR> geometry_points_on_surfaces;
    std::vector<Handle_Geom_BSplineSurface> geometry_surfaces_bspline;
    Eigen::MatrixI boundary_faces_order;
    Eigen::MatrixR surfaces_Uparameters;
    Eigen::MatrixR surfaces_Vparameters;

protected:

    Eigen::MatrixI projection_ID;
    Eigen::MatrixR projection_U;
    Eigen::MatrixR projection_V;
    Eigen::MatrixI sorted_projected_indicesU;
    Eigen::MatrixI sorted_projected_indicesV;
    Eigen::MatrixI dirichlet_faces;
    std::vector<Integer> listfaces;
    Integer no_dir_faces;
    Eigen::MatrixI elements_with_boundary_faces;
    Eigen::MatrixI curve_surface_projection_flags;

    std::vector<Boolean> FindPlanarSurfaces();
};

#endif // POSTMESHSURFACE_H
