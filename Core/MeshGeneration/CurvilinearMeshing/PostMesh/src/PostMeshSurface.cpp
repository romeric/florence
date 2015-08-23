
#include <PostMeshSurface.hpp>

PostMeshSurface::PostMeshSurface(const PostMeshSurface& other) : PostMeshBase(other)
{
    // Copy constructor
    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;

    this->geometry_points_on_surfaces = other.geometry_points_on_surfaces;
    this->geometry_surfaces_bspline = other.geometry_surfaces_bspline;
    this->boundary_faces_order = other.boundary_faces_order;
    cout << "copy" << endl;
}

PostMeshSurface& PostMeshSurface::operator=(const PostMeshSurface& other)
{
    // Copy assignment operator
    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;

    this->geometry_points_on_surfaces = other.geometry_points_on_surfaces;
    this->geometry_surfaces_bspline = other.geometry_surfaces_bspline;
    this->boundary_faces_order = other.boundary_faces_order;
    cout << "copy assignment" << endl;

    return *this;
}

PostMeshSurface::PostMeshSurface(PostMeshSurface&& other) : PostMeshBase(std::move(other))
{
    // Move constructor
    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;

    this->geometry_points_on_surfaces = std::move(other.geometry_points_on_surfaces);
    this->geometry_surfaces_bspline = std::move(other.geometry_surfaces_bspline);
    this->boundary_faces_order = std::move(other.boundary_faces_order);
    cout << "move" << endl;
}

PostMeshSurface& PostMeshSurface::operator=(PostMeshSurface&& other)
{
    // Move assignment operator
    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;

    this->geometry_points_on_surfaces = std::move(other.geometry_points_on_surfaces);
    this->geometry_surfaces_bspline = std::move(other.geometry_surfaces_bspline);
    this->boundary_faces_order = std::move(other.boundary_faces_order);
    cout << "move assignment" << endl;

    return *this;
}



void PostMeshSurface::InferInterpolationPolynomialDegree()
{
    //! WORKS ONLY FOR TETS
    for (auto i=0; i<50;++i)
    {
        if ( (i+1)*(i+2)*(i+3)/6 == this->mesh_elements.cols())
        {
            this->degree = i;
            break;
        }
    }
}

void PostMeshSurface::SurfacesToBsplineSurfaces()
{
    //! Converts all the imported surfaces to bspline surfaces : http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
    this->geometry_surfaces_bspline.clear();
    for (unsigned int isurf=0; isurf < this->geometry_surfaces.size(); ++isurf)
    {
        this->geometry_surfaces_bspline.push_back( GeomConvert::SurfaceToBSplineSurface(this->geometry_surfaces[isurf]) );
    }
}

void PostMeshSurface::ProjectMeshOnSurface()
{
    /* Projects all the points on the mesh to the boundary of Geom_Curve */
    this->projection_ID = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim);
    this_curve->projection_U = Eigen::MatrixR::Zero(this->mesh_edges.rows(),this->ndim);
    this->projection_V = Eigen::MatrixR::Zero(this->mesh_edges.rows(),this->ndim);
    this->dirichlet_faces = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim+1);
    this->listfaces.clear();
    int index_face = 0;

    for (Integer iface=0; iface<this->mesh_edges.rows(); ++iface)
    {
        for (UInteger jface=0; jface<this->ndim; ++jface) // linear
        {
            Real x = this->mesh_points(this->mesh_edges(iface,jface),0);
            Real y = this->mesh_points(this->mesh_edges(iface,jface),1);
            Real z = this->mesh_points(this->mesh_edges(iface,jface),2);
            if (sqrt(x*x+y*y+z*z)<this->condition)
            {
                if (jface==0)
                {
                    this->listfaces.push_back(iface);
                    for (UInteger iter=0;iter<ndim;++iter)
                        dirichlet_faces(index_face,iter) = this->mesh_edges(iface,iter);
                    index_face +=1;
                }

                Real min_distance = 1.0e10;
                Real distance = 0.;
                gp_Pnt project_this_point = gp_Pnt(x,y,0.0);
                for (unsigned int kface=0; kface<this->geometry_surfaces.size(); ++kface)
                {
                    GeomAPI_ProjectPointOnSurf proj;
                    proj.Init(project_this_point,this->geometry_surfaces[kface]);
                    distance = proj.LowerDistance();

                    if (distance < min_distance)
                    {
                        // STORE ID OF NURBS
                        this->projection_ID(iface,jface) = kface;
                        proj.LowerDistanceParameters(this_curve->projection_U(iface,jface),this->projection_V(iface,jface)); //DIVIDE BY DIRECTIONAL LENGTH (AREA)?
                        min_distance = distance;
                    }
                }
            }
        }
    }
}

void PostMeshSurface::MeshPointInversionSurface()
{

}

void PostMeshSurface::GetInternalSurfaceScales()
{

}
