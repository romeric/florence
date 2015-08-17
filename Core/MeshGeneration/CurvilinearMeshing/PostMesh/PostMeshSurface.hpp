#ifndef POSTMESHSURFACE_H
#define POSTMESHSURFACE_H

#include <PostMeshBase.hpp>
#include <PostMeshCurve.hpp>

class PostMeshSurface: public PostMeshBase
{


public:
    PostMeshSurface()
    {
        this->ndim = 3;
        this->mesh_element_type = "tet";
        this->scale = 1.0;
        this->condition = 1.0e10;

        this_curve = std::make_shared<PostMeshCurve>(PostMeshCurve());
    }
    PostMeshSurface(std::string &element_type, const UInteger &dim) : PostMeshBase(element_type,dim){}
    ~PostMeshSurface(){}

    inline void Init();
    void ProjectMeshOnSurface();
    void InferInterpolationPolynomialDegree();
    void SurfacesToBsplineSurfaces();
    void MeshPointInversionSurface();


    std::vector<Eigen::MatrixR> geometry_points_on_surfaces;
    std::vector<Handle_Geom_BSplineSurface> geometry_surfaces_bspline;
    Eigen::MatrixI boundary_edges_order;

protected:
//    PostMeshCurve *this_curve;
    std::shared_ptr<PostMeshCurve> this_curve;

    Eigen::MatrixI projection_ID;
    Eigen::MatrixR projection_V;
    Eigen::MatrixI sorted_projected_indices;
    Eigen::MatrixI dirichlet_faces;
    std::vector<Integer> listfaces;
    Standard_Integer no_dir_faces;
    Eigen::MatrixR v_of_all_fekete_mesh_edges;
    Eigen::MatrixR u_of_all_fekete_mesh_faces;
    Eigen::MatrixR v_of_all_fekete_mesh_faces;
    Eigen::MatrixI elements_with_boundary_faces;

    void GetInternalSurfaceScales();
};

#endif // POSTMESHSURFACE_H
