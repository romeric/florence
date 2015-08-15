#include "PostMeshSurface.hpp"

PostMeshSurface::PostMeshSurface()
{
    this->ndim = 3;
    this->mesh_element_type = "tet";
    this->scale = 1.0;
    this->condition = 1.0e10;

    this_curve = std::make_shared<PostMeshCurve>(PostMeshCurve());
}

void PostMeshSurface::Init()
{
    this->ndim = 3;
    this->mesh_element_type = "tet";
    this->scale = 1.0;
    this->condition = 1.0e10;

//    this->this_curve = new PostMeshCurve();
    this_curve = std::make_shared<PostMeshCurve>(PostMeshCurve());
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
