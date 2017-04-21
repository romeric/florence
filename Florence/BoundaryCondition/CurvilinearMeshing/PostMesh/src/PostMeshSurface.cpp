
#include <PostMeshSurface.hpp>

PostMeshSurface::PostMeshSurface(const PostMeshSurface& other)
noexcept(std::is_copy_assignable<PostMeshSurface>::value) : PostMeshBase(other)
{
    // COPY CONSTRUCTOR
    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;
    this->geometry_points_on_surfaces = other.geometry_points_on_surfaces;
    this->geometry_surfaces_bspline = other.geometry_surfaces_bspline;
    this->boundary_faces_order = other.boundary_faces_order;
    // REMAINING MEMBERS ARE COPY CONSTRUCTED BY BASE
}

PostMeshSurface& PostMeshSurface::operator=(const PostMeshSurface& other)
noexcept(std::is_copy_assignable<PostMeshSurface>::value)
{
    // COPY ASSIGNMENT OPERATOR
    this->mesh_elements = other.mesh_elements;
    this->mesh_points = other.mesh_points;
    this->mesh_edges = other.mesh_edges;
    this->mesh_faces = other.mesh_faces;
    this->projection_criteria = other.projection_criteria;
    this->degree = other.degree;
    this->imported_shape = other.imported_shape;
    this->no_of_shapes = other.no_of_shapes;
    this->geometry_points = other.geometry_points;
    this->geometry_curves = other.geometry_curves;
    this->geometry_surfaces = other.geometry_surfaces;
    this->geometry_curves_types = other.geometry_curves_types;
    this->geometry_surfaces_types = other.geometry_surfaces_types;
    this->displacements_BC = other.displacements_BC;
    this->index_nodes = other.index_nodes;
    this->nodes_dir = other.nodes_dir;
    this->fekete = other.fekete;

    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;
    this->geometry_points_on_surfaces = other.geometry_points_on_surfaces;
    this->geometry_surfaces_bspline = other.geometry_surfaces_bspline;
    this->boundary_faces_order = other.boundary_faces_order;

    return *this;
}

PostMeshSurface::PostMeshSurface(PostMeshSurface&& other) noexcept \
    : PostMeshBase(std::move(other))
{
    // MOVE CONSTRUCTOR
    this->ndim = other.ndim;
    this->geometry_points_on_surfaces = std::move(other.geometry_points_on_surfaces);
    this->geometry_surfaces_bspline = std::move(other.geometry_surfaces_bspline);
    this->boundary_faces_order = std::move(other.boundary_faces_order);
    // REMAINING MEMBERS ARE MOVE CONSTRUCTED BY BASE
}

PostMeshSurface& PostMeshSurface::operator=(PostMeshSurface&& other) noexcept
{
    // MOVE ASSIGNMENT OPERATOR
    this->mesh_elements = std::move(other.mesh_elements);
    this->mesh_points = std::move(other.mesh_points);
    this->mesh_edges = std::move(other.mesh_edges);
    this->mesh_faces = std::move(other.mesh_faces);
    this->projection_criteria = std::move(other.projection_criteria);
    this->degree = other.degree;
    this->imported_shape = std::move(other.imported_shape);
    this->no_of_shapes = other.no_of_shapes;
    this->geometry_points = std::move(other.geometry_points);
    this->geometry_curves = std::move(other.geometry_curves);
    this->geometry_surfaces = std::move(other.geometry_surfaces);
    this->geometry_curves_types = std::move(other.geometry_curves_types);
    this->geometry_surfaces_types = std::move(other.geometry_surfaces_types);
    this->displacements_BC = std::move(other.displacements_BC);
    this->index_nodes = std::move(other.index_nodes);
    this->nodes_dir = std::move(other.nodes_dir);
    this->fekete = std::move(other.fekete);

    this->ndim = other.ndim;
    this->mesh_element_type = other.mesh_element_type;
    this->geometry_points_on_surfaces = std::move(other.geometry_points_on_surfaces);
    this->geometry_surfaces_bspline = std::move(other.geometry_surfaces_bspline);
    this->boundary_faces_order = std::move(other.boundary_faces_order);

    return *this;
}


void PostMeshSurface::InferInterpolationPolynomialDegree()
{
    //! ONLY FOR TETS
    for (auto i=1; i<50;++i)
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
    //! CONVERST ALL SURFACES TO BSPLINE SURFACES:
    //! http://dev.opencascade.org/doc/refman/html/class_geom_convert.html

    this->geometry_surfaces_bspline.clear();
    for (unsigned int isurf=0; isurf < this->geometry_surfaces.size(); ++isurf)
    {
        this->geometry_surfaces_bspline.push_back(
                    GeomConvert::SurfaceToBSplineSurface(this->geometry_surfaces[isurf]) );
    }
}

void PostMeshSurface::GetSurfacesParameters()
{
    //! COMPUTES SURFACES BOUNDS IN THE PARAMETRIC SPACE
    this->surfaces_Uparameters = Eigen::MatrixR::Zero(this->geometry_surfaces.size(),2);
    this->surfaces_Vparameters = Eigen::MatrixR::Zero(this->geometry_surfaces.size(),2);
    for (UInteger isurface=0; isurface<this->geometry_surfaces.size(); ++isurface)
    {
        Handle_Geom_Surface current_surface = this->geometry_surfaces[isurface];
        Real u1,u2,v1,v2;
        current_surface->Bounds(u1,u2,v1,v2);

        surfaces_Uparameters(isurface,0) = u1;
        surfaces_Uparameters(isurface,1) = u2;

        surfaces_Vparameters(isurface,0) = v1;
        surfaces_Vparameters(isurface,1) = v2;
    }
}

void PostMeshSurface::GetGeomPointsOnCorrespondingFaces()
{
    //! COMPUTE WHICH GEOMETRICAL POINTS LIE ON WHICH GEMOETRICAL SURFACE
    this->geometry_points_on_surfaces.clear();
    for (TopExp_Explorer explorer_face(this->imported_shape,TopAbs_FACE); explorer_face.More(); explorer_face.Next())
    {
        // GET THE FACES
        TopoDS_Face current_face = TopoDS::Face(explorer_face.Current());

        std::vector<Real> current_face_X, current_face_Y, current_face_Z;
        current_face_X.clear(); current_face_Y.clear(); current_face_Z.clear();

        Integer counter=0;
        for (TopExp_Explorer explorer_point(current_face,TopAbs_VERTEX); explorer_point.More(); explorer_point.Next())
        {
            TopoDS_Vertex current_vertex = TopoDS::Vertex(explorer_point.Current());
            gp_Pnt current_vertex_point = BRep_Tool::Pnt(current_vertex);

            current_face_X.push_back(current_vertex_point.X());
            current_face_Y.push_back(current_vertex_point.Y());
            current_face_Z.push_back(current_vertex_point.Z());

            counter++;
        }
        const auto &current_face_X_ = static_cast<Eigen::MatrixR>
                (Eigen::Map<Eigen::MatrixR>(current_face_X.data(),current_face_X.size(),1));
        const auto &current_face_Y_ = static_cast<Eigen::MatrixR>
                (Eigen::Map<Eigen::MatrixR>(current_face_Y.data(),current_face_Y.size(),1));
        const auto &current_face_Z_ = static_cast<Eigen::MatrixR>
                (Eigen::Map<Eigen::MatrixR>(current_face_Z.data(),current_face_Z.size(),1));

        Eigen::MatrixR current_face_coords(current_face_X_.rows(),3*current_face_X_.cols());
        current_face_coords << current_face_X_, current_face_Y_, current_face_Z_;
        this->geometry_points_on_surfaces.push_back(current_face_coords);
    }
}


void PostMeshSurface::IdentifySurfacesContainingFaces()
{
    //! IDENTIFY GEOMETRICAL SURFACES CONTAINING MESH FACES
    this->dirichlet_faces = Eigen::MatrixI::Ones(this->mesh_faces.rows(),this->ndim+1)*(-1);
    this->listfaces.clear();

    auto index_face = 0;

    // LOOP OVER FACES
    for (auto iface=0; iface<this->mesh_faces.rows(); ++iface)
    {
        // ONLY FOR FACES THAT NEED TO BE PROJECTED
        if (this->projection_criteria(iface)==1)
        {
            // A LIST OF PROJECTION FACES
            this->listfaces.push_back(iface);
            // FILL DIRICHLET DATA
            for (UInteger iter=0;iter<ndim;++iter)
            {
               this->dirichlet_faces(index_face,iter) = this->mesh_faces(iface,iter);
            }

            // GET THE COORDINATES OF THE FACE VERTICES (VERTICES OF TRIANGLE)
            auto x1 = this->mesh_points(this->mesh_faces(iface,0),0);
            auto y1 = this->mesh_points(this->mesh_faces(iface,0),1);
            auto z1 = this->mesh_points(this->mesh_faces(iface,0),2);

            auto x2 = this->mesh_points(this->mesh_faces(iface,1),0);
            auto y2 = this->mesh_points(this->mesh_faces(iface,1),1);
            auto z2 = this->mesh_points(this->mesh_faces(iface,1),2);

            auto x3 = this->mesh_points(this->mesh_faces(iface,2),0);
            auto y3 = this->mesh_points(this->mesh_faces(iface,2),1);
            auto z3 = this->mesh_points(this->mesh_faces(iface,2),2);

            std::vector<Integer> mins_1;
            std::vector<Integer> mins_2;
            std::vector<Integer> mins_3;

            // LOOP OVER SURFACES
            for (UInteger isurface=0; isurface<this->geometry_surfaces.size(); ++isurface)
            {
                // CHECK IF THE MESH POINTS AND GEOMETRY POINTS ARE THE SAME
                for (auto surf_iter=0; surf_iter<geometry_points_on_surfaces[isurface].rows(); ++surf_iter)
                {
                    auto x_surface = this->geometry_points_on_surfaces[isurface](surf_iter,0);
                    auto y_surface = this->geometry_points_on_surfaces[isurface](surf_iter,1);
                    auto z_surface = this->geometry_points_on_surfaces[isurface](surf_iter,2);
                    // CHECK IF THE POINT IS ON THE GEOMETRICAL SURFACE
                    if (std::abs(x_surface-x1) < projection_precision && \
                            std::abs(y_surface-y1) < projection_precision && \
                            std::abs(z_surface-z1) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x1 = x_surface;
                        y1 = y_surface;
                        z1 = z_surface;
                        break;
                    }
                    else if (std::abs(x_surface-x2) < projection_precision && \
                            std::abs(y_surface-y2) < projection_precision && \
                            std::abs(z_surface-z2) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x2 = x_surface;
                        y2 = y_surface;
                        z2 = z_surface;
                        break;
                    }
                    else if (std::abs(x_surface-x3) < projection_precision && \
                            std::abs(y_surface-y3) < projection_precision && \
                            std::abs(z_surface-z3) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x3 = x_surface;
                        y3 = y_surface;
                        z3 = z_surface;
                        break;
                    }
                }


                // VERTEX POINTS
                gp_Pnt vertex_1(x1,y1,z1);
                gp_Pnt vertex_2(x2,y2,z2);
                gp_Pnt vertex_3(x3,y3,z3);

                BRepAdaptor_Surface adapt_surface(this->topo_faces[isurface]);

                try
                {
                    Extrema_ExtPS extrema_1(vertex_1,adapt_surface,this->projection_precision,
                                            this->projection_precision,Extrema_ExtFlag_MIN);
                    Extrema_ExtPS extrema_2(vertex_2,adapt_surface,this->projection_precision,
                                            this->projection_precision,Extrema_ExtFlag_MIN);
                    Extrema_ExtPS extrema_3(vertex_3,adapt_surface,this->projection_precision,
                                            this->projection_precision,Extrema_ExtFlag_MIN);

                    for (auto extrema_iter=1; extrema_iter<=extrema_1.NbExt(); ++extrema_iter)
                    {
                        auto point_1_distance = extrema_1.SquareDistance(extrema_iter);
                        if (point_1_distance/this->scale < this->projection_precision)
                        {
                            mins_1.push_back(isurface);
                        }
                    }
                    for (auto extrema_iter=1; extrema_iter<=extrema_2.NbExt(); ++extrema_iter)
                    {
                        auto point_2_distance = extrema_2.SquareDistance(extrema_iter);
                        if (point_2_distance/this->scale < this->projection_precision)
                        {
                            mins_2.push_back(isurface);
                        }
                    }
                    for (auto extrema_iter=1; extrema_iter<=extrema_3.NbExt(); ++extrema_iter)
                    {
                        auto point_3_distance = extrema_3.SquareDistance(extrema_iter);
                        if (point_3_distance/this->scale < this->projection_precision)
                        {
                            mins_3.push_back(isurface);
                        }
                    }
                }
                catch (StdFail_NotDone)
                {

                }
            }

            // FIND IF ALL THREE NODES OF THE MESH CAN BE ON ONE SURFACE
            auto surface_to_project_to = cnp::intersect(mins_1,mins_2,mins_3);
            if (surface_to_project_to.size() == 1)
            {
                this->dirichlet_faces(index_face,3) = surface_to_project_to[0];
            }
            else if (surface_to_project_to.size() > 1)
            {
//                 warn("There is more than one surface to project the mesh face", iface, "to");
            }
            else if (surface_to_project_to.empty())
            {
//                 warn("Could not identify a common surface between three nodes of the mesh face", iface);
            }
            index_face +=1;
        }
    }

    // REDUCE THE MATRIX TO GET DIRICHLET FACES
    auto arr_rows = cnp::arange(static_cast<Integer>(index_face));
    auto arr_cols = cnp::arange(static_cast<Integer>(ndim)+1);
    this->dirichlet_faces = cnp::take(this->dirichlet_faces,arr_rows,arr_cols);

    this->IdentifyRemainingSurfacesByProjection();
}

void PostMeshSurface::IdentifyRemainingSurfacesByProjection()
{
    //! IDENTIFY GEOMETRICAL SURFACES CONTAINING MESH FACES
    this->projection_ID = Eigen::MatrixI::Zero(this->dirichlet_faces.rows(),7);

    // LOOP OVER DIRCHLET FACES
    for (auto idir=0; idir<this->dirichlet_faces.rows(); ++idir)
    {
        // MESH FACES THAT COULD NOT BE DETERMINED
        if (this->dirichlet_faces(idir,3)==-1)
        {
            // PROJECT IT OVER ALL SURFACES
            auto min_mid_distance = 1.0e20;
            auto mid_distance = 1.0e10;

            auto min_edge_1_distance = 1.0e20;
            auto edge_1_distance = 1.0e10;

            auto min_edge_2_distance = 1.0e20;
            auto edge_2_distance = 1.0e10;

            auto min_edge_3_distance = 1.0e20;
            auto edge_3_distance = 1.0e10;

            auto min_point_1_distance = 1.0e20;
            auto point_1_distance = 1.0e10;

            auto min_point_2_distance = 1.0e20;
            auto point_2_distance = 1.0e10;

            auto min_point_3_distance = 1.0e20;
            auto point_3_distance = 1.0e10;

            // GET THE COORDINATES OF THE FACE VERTICES (VERTICES OF TRIANGLE)
            auto x1 = this->mesh_points(this->mesh_faces(this->listfaces[idir],0),0);
            auto y1 = this->mesh_points(this->mesh_faces(this->listfaces[idir],0),1);
            auto z1 = this->mesh_points(this->mesh_faces(this->listfaces[idir],0),2);

            auto x2 = this->mesh_points(this->mesh_faces(this->listfaces[idir],1),0);
            auto y2 = this->mesh_points(this->mesh_faces(this->listfaces[idir],1),1);
            auto z2 = this->mesh_points(this->mesh_faces(this->listfaces[idir],1),2);

            auto x3 = this->mesh_points(this->mesh_faces(this->listfaces[idir],2),0);
            auto y3 = this->mesh_points(this->mesh_faces(this->listfaces[idir],2),1);
            auto z3 = this->mesh_points(this->mesh_faces(this->listfaces[idir],2),2);

            // LOOP OVER SURFACES
            for (UInteger isurface=0; isurface<this->geometry_surfaces.size(); ++isurface)
            {
                // CHECK IF THE MESH POINTS AND GEOMETRY POINTS ARE THE SAME
                for (auto surf_iter=0; surf_iter<geometry_points_on_surfaces[isurface].rows(); ++surf_iter)
                {
                    auto x_surface = this->geometry_points_on_surfaces[isurface](surf_iter,0);
                    auto y_surface = this->geometry_points_on_surfaces[isurface](surf_iter,1);
                    auto z_surface = this->geometry_points_on_surfaces[isurface](surf_iter,2);
                    // CHECK IF THE POINT IS ON THE GEOMETRICAL SURFACE
                    if (std::abs(x_surface-x1) < projection_precision && \
                            std::abs(y_surface-y1) < projection_precision && \
                            std::abs(z_surface-z1) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x1 = x_surface;
                        y1 = y_surface;
                        z1 = z_surface;
                        break;
                    }
                    else if (std::abs(x_surface-x2) < projection_precision && \
                            std::abs(y_surface-y2) < projection_precision && \
                            std::abs(z_surface-z2) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x2 = x_surface;
                        y2 = y_surface;
                        z2 = z_surface;
                        break;
                    }
                    else if (std::abs(x_surface-x3) < projection_precision && \
                            std::abs(y_surface-y3) < projection_precision && \
                            std::abs(z_surface-z3) < projection_precision )
                    {
                        // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE
                        x3 = x_surface;
                        y3 = y_surface;
                        z3 = z_surface;
                        break;
                    }
                }

                // GET THE MID-POINT OF THE FACE/TRIANGLE
                auto x_avg = ( x1 + x2 + x3 )/3.;
                auto y_avg = ( y1 + y2 + y3 )/3.;
                auto z_avg = ( z1 + z2 + z3 )/3.;

                gp_Pnt middle_point(x_avg,y_avg,z_avg);

                // IN 3D A MID-POINT IS NOT ENOUGH TO DECIDE WHICH MESH FACE IS ON WHICH SURFACE
                // HENCE WE PROJECT THE MIDDLE POINT OF EVERY EDGE, THE REASON BEING THAT IF TWO
                // EDGES OF A FACE IS ON A SURFACE THAN THE FACE IS ON THE SURFACE
                gp_Pnt edge_point_1((x1+x2)/2.,(y1+y2)/2.,(z1+z2)/2.);
                gp_Pnt edge_point_2((x1+x3)/2.,(y1+y3)/2.,(z1+z3)/2.);
                gp_Pnt edge_point_3((x2+x3)/2.,(y2+y3)/2.,(z2+z3)/2.);

                // VERTEX POINTS
                gp_Pnt vertex_point_1(x1,y1,z1);
                gp_Pnt vertex_point_2(x2,y2,z2);
                gp_Pnt vertex_point_3(x3,y3,z3);

                // PROJECT THE NODES ON THE SURFACE AND GET THE NEAREST POINT
                try
                {
                    GeomAPI_ProjectPointOnSurf proj;
                    proj.Init(middle_point,this->geometry_surfaces[isurface]);
                    mid_distance = proj.LowerDistance();

                    // MAKE ONLY ONE OBJECT AND UPDATE IT
                    GeomAPI_ProjectPointOnSurf proj_edges;
                    proj_edges.Init(edge_point_1,this->geometry_surfaces[isurface]);
                    edge_1_distance = proj_edges.LowerDistance();

                    proj_edges.Init(edge_point_2,this->geometry_surfaces[isurface]);
                    edge_2_distance = proj_edges.LowerDistance();

                    proj_edges.Init(edge_point_3,this->geometry_surfaces[isurface]);
                    edge_3_distance = proj_edges.LowerDistance();

                    // MAKE ONLY ONE OBJECT AND UPDATE IT
                    GeomAPI_ProjectPointOnSurf proj_points;
                    proj_points.Init(vertex_point_1,this->geometry_surfaces[isurface]);
                    point_1_distance = proj_points.LowerDistance();

                    proj_points.Init(vertex_point_2,this->geometry_surfaces[isurface]);
                    point_2_distance = proj_points.LowerDistance();

                    proj_points.Init(vertex_point_3,this->geometry_surfaces[isurface]);
                    point_3_distance = proj_points.LowerDistance();
                }
                catch (StdFail_NotDone)
                {
                    // StdFail_NotDone ISSUE - DO NOTHING
                }
                if (mid_distance < min_mid_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,0) = isurface;
                    // RE-ASSIGN
                    min_mid_distance = mid_distance;
                }

                if (edge_1_distance < min_edge_1_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,1) = isurface;
                    // RE-ASSIGN
                    min_edge_1_distance = edge_1_distance;
                }
                if (edge_2_distance < min_edge_2_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,2) = isurface;
                    // RE-ASSIGN
                    min_edge_2_distance = edge_2_distance;
                }
                if (edge_3_distance < min_edge_3_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,3) = isurface;
                    // RE-ASSIGN
                    min_edge_3_distance = edge_3_distance;
                }

                if (point_1_distance < min_point_1_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,4) = isurface;
                    // RE-ASSIGN
                    min_point_1_distance = point_1_distance;
                }
                if (point_2_distance < min_point_2_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,5) = isurface;
                    // RE-ASSIGN
                    min_point_2_distance = point_2_distance;
                }
                if (point_3_distance < min_point_3_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(idir,6) = isurface;
                    // RE-ASSIGN
                    min_point_3_distance = point_3_distance;
                }
            }
        }
    }

    // BASED ON FOUR PROJECTIONS DECIDE WHICH FACE IS ON WHICH SURFACE
    for (auto i=0; i<this->dirichlet_faces.rows(); ++i)
    {
        if (dirichlet_faces(i,3)==-1)
        {
            if (this->projection_ID(i,4)==this->projection_ID(i,5) &&
                    this->projection_ID(i,5)==this->projection_ID(i,6) &&
                    this->projection_ID(i,1)==this->projection_ID(i,2) &&
                    this->projection_ID(i,2)==this->projection_ID(i,3) &&
                    this->projection_ID(i,1) ==this->projection_ID(i,4) )
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,4);
            }
            else if (this->projection_ID(i,4)==this->projection_ID(i,5) &&
                     this->projection_ID(i,5)==this->projection_ID(i,6) &&
                     this->projection_ID(i,1)==this->projection_ID(i,2) &&
                     this->projection_ID(i,2)==this->projection_ID(i,3) &&
                     this->projection_ID(i,1) !=this->projection_ID(i,4) )
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,1);
            }
    //        else if (this->projection_ID(i,4)==this->projection_ID(i,5) &&
    //                this->projection_ID(i,5)==this->projection_ID(i,6) )
    //        {
    //            this->dirichlet_faces(i,3) = this->projection_ID(i,4);
    //        }
            else if (this->projection_ID(i,0)==this->projection_ID(i,1) &&
                    this->projection_ID(i,1)==this->projection_ID(i,2) &&
                    this->projection_ID(i,2)==this->projection_ID(i,3) )
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,0);
            }
            else if (this->projection_ID(i,1)==this->projection_ID(i,2) &&
                     this->projection_ID(i,2)==this->projection_ID(i,3))
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,1);
            }
            else if (this->projection_ID(i,1)==this->projection_ID(i,2) &&
                     this->projection_ID(i,1)!=this->projection_ID(i,3))
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,1);
            }
            else if (this->projection_ID(i,1)==this->projection_ID(i,3) &&
                     this->projection_ID(i,1)!=this->projection_ID(i,2))
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,1);
            }
            else if (this->projection_ID(i,2)==this->projection_ID(i,3) &&
                     this->projection_ID(i,2)!=this->projection_ID(i,1))
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,2);
            }
            else
            {
                this->dirichlet_faces(i,3) = this->projection_ID(i,0);
            }
        }
    }
}

void PostMeshSurface::IdentifySurfacesContainingFacesByPureProjection()
{
    //! IDENTIFY GEOMETRICAL SURFACES CONTAINING MESH FACES
    this->dirichlet_faces = Eigen::MatrixI::Zero(this->mesh_faces.rows(),this->ndim+1);
    this->projection_ID = Eigen::MatrixI::Zero(this->mesh_faces.rows(),4);
    this->listfaces.clear();

    auto index_face = 0;

    // CREATE THE OBJECTS ONLY ONCE
    GeomAPI_ProjectPointOnSurf proj;
    // IN 3D A MID-POINT IS NOT ENOUGH TO DECIDE WHICH MESH FACE IS ON WHICH SURFACE
    // HENCE WE PROJECT THE MIDDLE POINT OF EVERY EDGE, THE REASON BEING THAT IF TWO
    // EDGES OF A FACE IS ON A SURFACE THAN THE FACE IS ON THE SURFACE
    gp_Pnt middle_point, edge_point_1, edge_point_2, edge_point_3;

    // LOOP OVER DIRCHLET FACES
    for (auto iface=0; iface<this->mesh_faces.rows(); ++iface)
    {
        // MESH FACES THAT COULD NOT BE DETERMINED
        if (this->projection_criteria(iface)==1)
        {
            // A LIST OF PROJECTION FACES
            this->listfaces.push_back(iface);
            // FILL DIRICHLET DATA
            for (UInteger iter=0;iter<ndim;++iter)
            {
               this->dirichlet_faces(index_face,iter) = this->mesh_faces(iface,iter);
            }

            // PROJECT IT OVER ALL SURFACES
            auto min_mid_distance = 1.0e20;
            auto mid_distance = 1.0e10;

            auto min_edge_1_distance = 1.0e20;
            auto edge_1_distance = 1.0e10;

            auto min_edge_2_distance = 1.0e20;
            auto edge_2_distance = 1.0e10;

            auto min_edge_3_distance = 1.0e20;
            auto edge_3_distance = 1.0e10;

            // GET THE COORDINATES OF THE FACE VERTICES (VERTICES OF TRIANGLE)
            auto x1 = this->mesh_points(this->mesh_faces(iface,0),0);
            auto y1 = this->mesh_points(this->mesh_faces(iface,0),1);
            auto z1 = this->mesh_points(this->mesh_faces(iface,0),2);

            auto x2 = this->mesh_points(this->mesh_faces(iface,1),0);
            auto y2 = this->mesh_points(this->mesh_faces(iface,1),1);
            auto z2 = this->mesh_points(this->mesh_faces(iface,1),2);

            auto x3 = this->mesh_points(this->mesh_faces(iface,2),0);
            auto y3 = this->mesh_points(this->mesh_faces(iface,2),1);
            auto z3 = this->mesh_points(this->mesh_faces(iface,2),2);

            // GET THE MID-POINT OF THE FACE/TRIANGLE
            auto x_avg = ( x1 + x2 + x3 )/3.;
            auto y_avg = ( y1 + y2 + y3 )/3.;
            auto z_avg = ( z1 + z2 + z3 )/3.;

            // UPDATE POINTS
            middle_point.SetX(x_avg); middle_point.SetY(y_avg); middle_point.SetZ(z_avg);

            edge_point_1.SetX((x1+x2)/2.); edge_point_1.SetY((y1+y2)/2.); edge_point_1.SetZ((z1+z2)/2.);
            edge_point_2.SetX((x1+x3)/2.); edge_point_2.SetY((y1+y3)/2.); edge_point_2.SetZ((z1+z3)/2.);
            edge_point_3.SetX((x2+x3)/2.); edge_point_3.SetY((y2+y3)/2.); edge_point_3.SetZ((z2+z3)/2.);

            // LOOP OVER SURFACES
            for (UInteger isurface=0; isurface<this->geometry_surfaces.size(); ++isurface)
            {
                // PROJECT THE NODES ON THE SURFACE AND GET THE NEAREST POINT
                try
                {
                    proj.Init(middle_point,this->geometry_surfaces[isurface]);
                    mid_distance = proj.LowerDistance();

                    proj.Init(edge_point_1,this->geometry_surfaces[isurface]);
                    edge_1_distance = proj.LowerDistance();

                    proj.Init(edge_point_2,this->geometry_surfaces[isurface]);
                    edge_2_distance = proj.LowerDistance();

                    proj.Init(edge_point_3,this->geometry_surfaces[isurface]);
                    edge_3_distance = proj.LowerDistance();
                }
                catch (StdFail_NotDone)
                {
                    // StdFail_NotDone ISSUE - DO NOTHING
                }
                if (mid_distance < min_mid_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(index_face,0) = isurface;
                    // RE-ASSIGN
                    min_mid_distance = mid_distance;
                }

                if (edge_1_distance < min_edge_1_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(index_face,1) = isurface;
                    // RE-ASSIGN
                    min_edge_1_distance = edge_1_distance;
                }
                if (edge_2_distance < min_edge_2_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(index_face,2) = isurface;
                    // RE-ASSIGN
                    min_edge_2_distance = edge_2_distance;
                }
                if (edge_3_distance < min_edge_3_distance)
                {
                    // STORE ID OF SURFACES
                    this->projection_ID(index_face,3) = isurface;
                    // RE-ASSIGN
                    min_edge_3_distance = edge_3_distance;
                }
            }
            index_face +=1;
        }
    }

    // REDUCE THE MATRIX TO GET DIRICHLET FACES
    auto arr_rows = cnp::arange(static_cast<Integer>(index_face));
    auto arr_cols = cnp::arange(static_cast<Integer>(ndim)+1);
    this->dirichlet_faces = cnp::take(this->dirichlet_faces,arr_rows,arr_cols);
    arr_cols = cnp::arange(static_cast<Integer>(4));
    this->projection_ID = cnp::take(this->projection_ID,arr_rows,arr_cols);

    // BASED ON FOUR PROJECTIONS DECIDE WHICH FACE IS ON WHICH SURFACE
    for (auto i=0; i<this->dirichlet_faces.rows(); ++i)
    {
        if (this->projection_ID(i,0)==this->projection_ID(i,1) &&
                this->projection_ID(i,1)==this->projection_ID(i,2) &&
                this->projection_ID(i,2)==this->projection_ID(i,3) )
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,0);
        }
        else if (this->projection_ID(i,1)==this->projection_ID(i,2) &&
                 this->projection_ID(i,2)==this->projection_ID(i,3))
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,1);
        }
        else if (this->projection_ID(i,1)==this->projection_ID(i,2) &&
                 this->projection_ID(i,1)!=this->projection_ID(i,3))
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,1);
        }
        else if (this->projection_ID(i,1)==this->projection_ID(i,3) &&
                 this->projection_ID(i,1)!=this->projection_ID(i,2))
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,1);
        }
        else if (this->projection_ID(i,2)==this->projection_ID(i,3) &&
                 this->projection_ID(i,2)!=this->projection_ID(i,1))
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,2);
        }
        else
        {
            this->dirichlet_faces(i,3) = this->projection_ID(i,0);
        }
    }
}

void PostMeshSurface::SupplySurfacesContainingFaces(const Integer *arr, Integer rows, Integer already_mapped, Integer caller)
{
    //! IN SOME EXTREME CASES OPENCASCADE MIGHT FAIL TO IDENTIFY THE RIGHT GEOMETRICAL
    //! SURFACE TO PROJECT THE MESH FACE TO. IN SUCH CASES IT IS CONVENIENT TO SUPPLY
    //! THIS INFORMATION FROM AN EXTERNAL MESH GENERATOR (E.G. GID, GMSH ETC).
    //! NOTE THAT THIS FUNCTION ASSUMES THAT THE EXTERNAL FACE-T0-SURFACE CORRESPONDENCE
    //! SUPPLIED, IS MORE ACCURATE THAN THE INTERNAL (OPENCASADE ONE). HOWEVER, IT IS
    //! POSSIBLE THAT THE SURFACE NUMBERING OF EXTERNALLY PROVIDED SOFTWARE AND OPENCASCADE
    //! ARE DIFFERENT. FOR SUCH CASES SUPPLY ALREADY_MAPPED FLAG AS ZERO

    Eigen::MatrixI dirichlet_faces_ext = Eigen::MatrixI::Ones(rows,this->ndim+1)*(-1);
    this->listfaces.clear();
    auto index_face = 0;
    // LOOP OVER FACES
    for (auto iface=0; iface<this->mesh_faces.rows(); ++iface)
    {
        // ONLY FOR FACES THAT NEED TO BE PROJECTED
        if (this->projection_criteria(iface)==1)
        {
            // A LIST OF PROJECTION FACES
            this->listfaces.push_back(iface);
            // FILL DIRICHLET DATA
            for (UInteger iter=0; iter<this->ndim; ++iter)
            {
               dirichlet_faces_ext(index_face,iter) = this->mesh_faces(iface,iter);
            }
            if (already_mapped == 1)
            {
                dirichlet_faces_ext(index_face,3) = arr[iface];
            }
            index_face++;
        }
    }

    if (already_mapped == 1)
    {
        // ALREADY-MAPPED ESSENTIALLY MEANS THAT THE EXTERNAL GEOMETRICAL SURFACE
        // NUMBERING IS THE SAME AS OPENCASCADE'S
        // REDUCE THE MATRIX TO GET DIRICHLET FACES
        auto arr_rows = cnp::arange(static_cast<Integer>(index_face));
        auto arr_cols = cnp::arange(static_cast<Integer>(ndim)+1);
        this->dirichlet_faces = cnp::take(dirichlet_faces_ext,arr_rows,arr_cols);
        return;
    }

    // IF THE EXTERNAL AND OPENCASCADE GEOMETRICAL SURFACES ARE NUMBERED DIFFERENTLY
    // THEN WE NEED A MAPPING, WHICH REQUIRES IDENTIFICATION OF SURFACES BY OPENCASCADE
    if (caller == 0)
    {
        // CALLER = 0 CORRESPONDS TO MORE EXPENSIVE FUNCTION FOR SURFACE IDENTIFICATION
        this->IdentifySurfacesContainingFaces();
    }
    else if (caller == 1)
    {
        // CALLER = 1 CORRESPONDS TO THE CHEAPER FUNCTION FOR SURFACE IDENTIFICATION
        this->IdentifySurfacesContainingFacesByPureProjection();
    }

    const Eigen::MatrixI surface_flags_ext = Eigen::Map<const Eigen::MatrixI>(arr,rows,1);
    std::vector<Integer> unique_surface_flags_ext;
    std::tie(unique_surface_flags_ext,std::ignore) = cnp::unique(surface_flags_ext);

    for (auto &k: unique_surface_flags_ext)
    {
        Eigen::MatrixUI urows;
        std::tie(urows,std::ignore) = cnp::where_eq(surface_flags_ext,k);
        Eigen::MatrixUI col(1,1); col(0,0) = 3;

        //! FIND THE ROWS OF INTERNAL BASED ON EXTRENAL. GET ITEMFREQ OF THESE ROWS
        //! GET THE MAXIMUM OCCURENCE FROM ITEMFREQ AND CHANGE THE REMAINING
        //! INTERNAL ROWS BASED ON THAT
        Eigen::MatrixI flags_int = cnp::take(this->dirichlet_faces,urows,col);
        // Eigen::MatrixI flags_ext = cnp::take(dirichlet_faces_ext,urows,col);
        auto freqs_int = cnp::itemfreq(flags_int);
        if (freqs_int.rows()==1)
        {
            cnp::put(dirichlet_faces_ext,flags_int(0,0),urows,col);
        }
        else
        {
            Eigen::MatrixI col_1_flags_int = freqs_int.col(1);
            auto nmax = col_1_flags_int.maxCoeff();
            Eigen::MatrixUI encounters;
            std::tie(encounters,std::ignore) = cnp::where_eq(col_1_flags_int,nmax);
            if (encounters.rows()==1)
            {
                cnp::put(dirichlet_faces_ext,freqs_int(encounters(nmax,0),0),urows,col);
            }
            else
            {
                warn("Surface detection failed. Mesh faces are getting projected to incorrect surfaces");
            }
        }
    }

    this->dirichlet_faces = dirichlet_faces_ext;
}

void PostMeshSurface::IdentifySurfacesIntersections()
{
    //! THIS METHOD CHECKS IF THERE ARE EDGES WHICH ARE SHARED BETWEEN TWO FACES
    //! WITH EACH OF THOSE FACES BEING PROJECTED ON TO DIFFERENT SURFACES

    // BUILD EDGES FIRST
    Eigen::MatrixI edges(this->dirichlet_faces.rows()*3,3);
    for (auto i=0; i < this->dirichlet_faces.rows(); ++i)
    {
        edges(i,0) = this->dirichlet_faces(i,0);
        edges(i,1) = this->dirichlet_faces(i,1);
        edges(i,2) = this->dirichlet_faces(i,3);

        edges(i+this->dirichlet_faces.rows(),0) = this->dirichlet_faces(i,0);
        edges(i+this->dirichlet_faces.rows(),1) = this->dirichlet_faces(i,2);
        edges(i+this->dirichlet_faces.rows(),2) = this->dirichlet_faces(i,3);

        edges(i+2*this->dirichlet_faces.rows(),0) = this->dirichlet_faces(i,1);
        edges(i+2*this->dirichlet_faces.rows(),1) = this->dirichlet_faces(i,2);
        edges(i+2*this->dirichlet_faces.rows(),2) = this->dirichlet_faces(i,3);
    }

    Eigen::MatrixI edges_only = edges.block(0,0,edges.rows(),2);
    cnp::sort_rows(edges_only);
    edges.block(0,0,edges.rows(),2) = edges_only;

    Eigen::MatrixI faces_with_curve_projection_edges(3*this->dirichlet_faces.rows(),2);
    auto counter = 0;
    for (auto i=0; i<edges.rows();++i)
    {
        for (auto j=i+1; j<edges.rows();++j)
        {
            if ( i!=j && (edges(i,0)==edges(j,0) && edges(i,1)==edges(j,1)) )
            {
                if (edges(i,2) != edges(j,2))
                {
                    faces_with_curve_projection_edges(2*counter,0) = i % this->dirichlet_faces.rows();
                    faces_with_curve_projection_edges(2*counter,1) = i / this->dirichlet_faces.rows();

                    faces_with_curve_projection_edges(2*counter+1,0) = j % this->dirichlet_faces.rows();
                    faces_with_curve_projection_edges(2*counter+1,1) = j / this->dirichlet_faces.rows();
                    counter++;
                }
            }
        }
    }

    auto arr_row = cnp::arange(2*counter);
    auto arr_col = cnp::arange(2);
    faces_with_curve_projection_edges = cnp::take(faces_with_curve_projection_edges,arr_row,arr_col);

    this->GetBoundaryPointsOrder();

    this->curve_surface_projection_flags.setZero(this->dirichlet_faces.rows(),this->mesh_faces.cols());
    for (auto i=0; i<faces_with_curve_projection_edges.rows(); ++i)
    {
        auto current_edge = faces_with_curve_projection_edges(i,1);
        auto edge_order = this->boundary_faces_order.row(current_edge);
        for (auto j=1; j< edge_order.cols()-1; ++j)
        {
            this->curve_surface_projection_flags(faces_with_curve_projection_edges(i,0),edge_order[j]) = 1;
        }
    }
}

void PostMeshSurface::ProjectMeshOnSurface()
{
    // CONVENIENCE FUNCTION FOR SIMILARITY WITH 2D (USEFUL FOR REPAIRING DUAL IMAGES)
    this->InferInterpolationPolynomialDegree();
    this->projection_U = Eigen::MatrixR::Zero(this->dirichlet_faces.rows(),this->ndim);
    this->projection_V = Eigen::MatrixR::Zero(this->dirichlet_faces.rows(),this->ndim);

    // LOOP OVER EDGES
    for (auto idir=0; idir<this->dirichlet_faces.rows(); ++idir)
    {
        // LOOP OVER THE THREE VERTICES OF THE FACE
        for (UInteger inode=0; inode<this->ndim; ++inode)
        {
            // PROJECTION PARAMETERS
            Real parameterU, parameterV;
            // GET THE COORDINATES OF THE NODE
            auto x = this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),0);
            auto y = this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),1);
            auto z = this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),2);

            // GET THE SURFACE THAT THIS FACE HAS TO BE PROJECTED TO
            auto isurface = this->dirichlet_faces(idir,3);
            Handle_Geom_Surface current_surface = this->geometry_surfaces[isurface];
            // GET THE COORDINATES OF SURFACE'S THREE VERTICES
            Real x_surface, y_surface, z_surface;
            for (auto surf_iter=0; surf_iter<geometry_points_on_surfaces[isurface].rows(); ++surf_iter)
            {
                x_surface = this->geometry_points_on_surfaces[isurface](surf_iter,0);
                y_surface = this->geometry_points_on_surfaces[isurface](surf_iter,1);
                z_surface = this->geometry_points_on_surfaces[isurface](surf_iter,2);
                // CHECK IF THE POINT IS ON THE GEOMETRICAL SURFACE
                if (std::abs(x_surface-x) < projection_precision && \
                        std::abs(y_surface-y) < projection_precision && \
                        std::abs(z_surface-z) < projection_precision )
                {
                    // PROJECT THE SURFACE VERTEX INSTEAD OF THE FACE NODE (THIS IS NECESSARY TO ENSURE SUCCESSFUL PROJECTION)
                    x = x_surface;
                    y = y_surface;
                    z = z_surface;
                }
            }

            auto xEq = gp_Pnt(x,y,z);
            try
            {
                // GET THE NODE THAT HAS TO BE PROJECTED TO THE CURVE
                auto node_to_be_projected = gp_Pnt(x,y,z);
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                GeomAPI_ProjectPointOnSurf proj;
                proj.Init(node_to_be_projected,current_surface);
                proj.LowerDistanceParameters(parameterU,parameterV);
                current_surface->D0(parameterU,parameterV,xEq);
            }
            catch (StdFail_NotDone)
            {
                warn("The face node was not projected on to the right surface. Surface number: ",isurface);
            }

            // UPDATE THE MESH POINTS TO CONFORM TO CAD GEOMETRY - NOT TO SCALE
            this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),0) = xEq.X();
            this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),1) = xEq.Y();
            this->mesh_points(this->mesh_faces(this->listfaces[idir],inode),2) = xEq.Z();

            // STORE PROJECTION POINT PARAMETER ON THE SURFACE (NORMALISED)
            this->projection_U(idir,inode) = parameterU;
            this->projection_V(idir,inode) = parameterV;
        }
    }
}

void PostMeshSurface::RepairDualProjectedParameters()
{
    // SORT PROJECTED PARAMETERS OF EACH EDGE - MUST INITIALISE SORT INDICES
    this->sorted_projected_indicesU.setZero(this->projection_U.rows(),this->projection_U.cols());
    this->sorted_projected_indicesV.setZero(this->projection_V.rows(),this->projection_V.cols());

    cnp::sort_rows(this->projection_U,this->sorted_projected_indicesU);
    cnp::sort_rows(this->projection_V,this->sorted_projected_indicesV);


    auto tol = 1.0e-6;
    for (auto idir=0; idir < this->dirichlet_faces.rows(); ++idir)
    {
        auto id_surface = static_cast<Integer>(this->dirichlet_faces(idir,3));
        Handle_Geom_Surface current_surface = this->geometry_surfaces[id_surface];

        if (current_surface->IsUClosed())
        {
            Real u1,u2;
            u1 = this->projection_U.row(idir).minCoeff();
            u2 = this->projection_U.row(idir).maxCoeff();
            Real umin, umax, vmin, vmax;
            current_surface->Bounds(umin,umax,vmin,vmax);
            // CHECK BOUNDS TO MAKE SURE
            if (vmax < vmin)
            {
                std::swap(vmin,vmax);
            }

            if ( std::abs(u1 - umin) < tol )
            {
                // NODE IS AT THE BEGINNGING OF PARAMETRIC ISO-LINE
                auto L0C = std::abs(u2 - umin);
                auto LC1 = std::abs(umax - u2);
                if (L0C > LC1)
                {
                    if (std::abs(this->projection_U(idir,0) - this->projection_U(idir,1)) < tol)
                    {
                        this->projection_U(idir,1) = umax;
                    }
                    this->projection_U(idir,0) = umax;
                }
            }
        }

        if ( current_surface->IsVClosed())
        {
            Real v1,v2;
            v1 = this->projection_V.row(idir).minCoeff();
            v2 = this->projection_V.row(idir).maxCoeff();
            Real umin, umax, vmin, vmax;
            current_surface->Bounds(umin,umax,vmin,vmax);
            // CHECK BOUNDS TO MAKE SURE
            if (vmax < vmin)
            {
                std::swap(vmin,vmax);
            }
            if ( std::abs(v1 - umin) < tol )
            {
                // NODE IS AT THE BEGINNGING OF PARAMETRIC ISO-LINE
                auto L0C = std::abs(v2 - vmin);
                auto LC1 = std::abs(vmax - v2);
                if (L0C > LC1)
                {
                    if (std::abs(this->projection_V(idir,0) - this->projection_V(idir,1)) < tol)
                    {
                        this->projection_V(idir,1) = vmax;
                    }
                    this->projection_V(idir,0) = vmax;
                }
            }
        }
    }

    // SORT BACK
    cnp::sort_back_rows(this->projection_U,this->sorted_projected_indicesU);
    cnp::sort_back_rows(this->projection_V,this->sorted_projected_indicesV);
}

void PostMeshSurface::MeshPointInversionCurve(const gp_Pnt &point_in, gp_Pnt &point_out)
{
    auto min_distance = 1.0e20;
    auto distance = 1.0e10;
    for (UInteger i=0; i<this->geometry_curves.size(); ++i)
    {
        auto current_curve = this->geometry_curves[i];
        auto current_curve_type = this->geometry_curves_types[i];
        if (current_curve_type != GeomAbs_OtherCurve)
        {
            try
            {
    //            Real uEq;
    //            // TRY NEWTON-RAPSHON PROJECTION METHOD WITH LOWER PRECISION
    //            ShapeAnalysis_Curve proj_Newton;
    //            auto Newton_precision = projection_precision < 1.0e-05 ? 1.0e-5: projection_precision;
    //            proj_Newton.Project(current_curve,point_in,Newton_precision,point_out,uEq,True);

                GeomAPI_ProjectPointOnCurve proj;
                proj.Init(point_in,current_curve);
                distance = proj.LowerDistance();
                if (distance < min_distance)
                {
                    auto uEq = proj.LowerDistanceParameter();
                    current_curve->D0(uEq,point_out);
                    min_distance = distance;
                }
            }
            catch (StdFail_NotDone)
            {
                // DO NOTHING
            }
        }
    }
}

void PostMeshSurface::MeshPointInversionSurface(Integer project_on_curves, Integer modify_linear_mesh)
{
    this->no_dir_faces = this->dirichlet_faces.rows();
    auto no_face_nodes = this->mesh_faces.cols();
    Eigen::MatrixI arr_row = Eigen::Map<Eigen::Matrix<Integer,Eigen::Dynamic,1> >
            (this->listfaces.data(),this->listfaces.size());
    auto arr_col = cnp::arange(no_face_nodes);
    this->nodes_dir = cnp::take(this->mesh_faces,arr_row,arr_col);
    this->nodes_dir = cnp::ravel(this->nodes_dir);
    this->index_nodes = cnp::arange((Integer)no_face_nodes);
    this->displacements_BC = Eigen::MatrixR::Zero(this->no_dir_faces*no_face_nodes,this->ndim);

    if (this->curve_surface_projection_flags.rows() != this->dirichlet_faces.rows())
    {
        this->curve_surface_projection_flags.setZero(this->dirichlet_faces.rows(),this->mesh_faces.cols());
    }

    auto starter = 3;
    if (modify_linear_mesh==1)
    {
        starter = 0;
    }


    for (auto idir=0; idir< this->no_dir_faces; ++idir)
    {
        auto id_surface = static_cast<Integer>(this->dirichlet_faces(idir,3));
        Handle_Geom_Surface current_surface = this->geometry_surfaces[id_surface];

        for (auto j=starter; j<no_face_nodes;++j)
        {
            auto x = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),0);
            auto y = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),1);
            auto z = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),2);

            // LOOP OVER ALL GEOMETRY POINTS AND IF POSSIBLE PICK THOSE INSTEAD
            for (auto &k : this->geometry_points)
            {
                if ( (std::abs(k.X() - x ) < this->projection_precision) && \
                     (std::abs(k.Y() - y ) < this->projection_precision) && \
                     (std::abs(k.Z() - z ) < this->projection_precision) )
                {
                    x = k.X(); y = k.Y(); z = k.Z();
                    break;
                }
            }

            Real uEq,vEq;
            // COORDINDATES OF PROJECTED NODE
            auto xEq = gp_Pnt(x,y,z);
            auto gp_pnt_old = (this->mesh_points.row(this->nodes_dir(this->index_nodes( j ))).array()/this->scale);

            // MAKE THE POINT
            auto point_to_be_projected = gp_Pnt(x,y,z);

            // CHECK IF THE POINT IS SUPPOSED TO BE PROEJECTED TO A CURVE
            if (this->curve_surface_projection_flags(idir,j) == 1 && project_on_curves == 1)
            {
                this->MeshPointInversionCurve(point_to_be_projected,xEq);
            }
            else
            {
                try
                {
                    GeomAPI_ProjectPointOnSurf proj;
                    proj.Init(point_to_be_projected,current_surface,1e-06,Extrema_ExtAlgo_Grad);
                    proj.LowerDistanceParameters(uEq,vEq);
                    current_surface->D0(uEq,vEq,xEq);
                }
                catch (StdFail_NotDone)
                {
                    warn("Could not project node to the right surface. "
                         "Surface ID is:", id_surface, "  Surface type is:", this->geometry_surfaces_types[id_surface],
                         "  Node number is:", this->mesh_faces(this->listfaces[idir],j));
                }
            }

            if (j<static_cast<decltype(j)>(this->ndim))
            {
                // FOR VERTEX NODES KEEP THE DISPLACEMENT ZERO
                this->displacements_BC(this->index_nodes(j),0) = 0.;
                this->displacements_BC(this->index_nodes(j),1) = 0.;
                this->displacements_BC(this->index_nodes(j),2) = 0.;
                // BUT UPDATE THE MESH POINTS TO CONFORM TO CAD GEOMETRY - NOT TO SCALE
                this->mesh_points(this->mesh_faces(this->listfaces[idir],j),0) = xEq.X();
                this->mesh_points(this->mesh_faces(this->listfaces[idir],j),1) = xEq.Y();
                this->mesh_points(this->mesh_faces(this->listfaces[idir],j),2) = xEq.Z();
            }
            else
            {
                // FOR NON-VERTEX NODES GET THE REQUIRED DISPLACEMENT
                this->displacements_BC(this->index_nodes(j),0) = (xEq.X()/this->scale - gp_pnt_old(0));
                this->displacements_BC(this->index_nodes(j),1) = (xEq.Y()/this->scale - gp_pnt_old(1));
                this->displacements_BC(this->index_nodes(j),2) = (xEq.Z()/this->scale - gp_pnt_old(2));
            }
//            print(proj.NearestPoint().X()/this->scale, xEq.X()/this->scale,proj.Point(proj.NbPoints()).X()/this->scale);
//            print(proj.NearestPoint().X()/this->scale ,proj.Point(proj.NbPoints()).X()/this->scale);
//            print(gp_pnt_old(0),gp_pnt_old(1),gp_pnt_old(2),"   ",xEq.X()/this->scale,xEq.Y()/this->scale,xEq.Z()/this->scale);
//            print(gp_pnt_old(0),gp_pnt_old(1),gp_pnt_old(2));

        }
        this->index_nodes = ((this->index_nodes).array()+no_face_nodes).eval().matrix();
    }
//    print(this->dirichlet_faces);
//    print(this->displacements_BC);
//    print(displacements_BC.maxCoeff());
//    exit (EXIT_FAILURE);
}

void PostMeshSurface::MeshPointInversionSurfaceArcLength(Integer project_on_curves, Real OrthTol, Real *FEbases, Integer rows, Integer cols)
{
    if (project_on_curves==1)
    {
        warn("Projection on curves is not implemented in the 3D arc-length based projection. This will be ignored");
    }
    Eigen::MatrixR FEBases = Eigen::Map<Eigen::MatrixR>(FEbases,rows,cols);

    this->no_dir_faces = this->dirichlet_faces.rows();
    auto no_face_nodes = this->mesh_faces.cols();
    Eigen::MatrixI arr_row = Eigen::Map<Eigen::Matrix<Integer,Eigen::Dynamic,1> >
            (this->listfaces.data(),this->listfaces.size());
    auto arr_col = cnp::arange(no_face_nodes);
    this->nodes_dir = cnp::take(this->mesh_faces,arr_row,arr_col);
    this->nodes_dir = cnp::ravel(this->nodes_dir);
    this->index_nodes = cnp::arange((Integer)no_face_nodes);
    this->displacements_BC = Eigen::MatrixR::Zero(this->no_dir_faces*no_face_nodes,this->ndim);

    for (auto idir=0; idir< this->no_dir_faces; ++idir)
    {
        auto id_surface = static_cast<Integer>(this->dirichlet_faces(idir,3));
        Handle_Geom_Surface current_surface = this->geometry_surfaces[id_surface];

        Real u1 = this->projection_U(idir,0);
        Real u2 = this->projection_U(idir,1);
        Real u3 = this->projection_U(idir,2);

        Real v1 = this->projection_V(idir,0);
        Real v2 = this->projection_V(idir,1);
        Real v3 = this->projection_V(idir,2);

        Eigen::MatrixR face_vertices(3,2);
        face_vertices << u1,v1,u2,v2,u3,v3;

        Eigen::MatrixR parametric_surface = FEBases.transpose()*face_vertices;
        parametric_surface.block(0,0,3,2) = face_vertices; // NOT NECESSARY

        for (auto j=0; j<no_face_nodes;++j)
        {
            if (j<static_cast<decltype(j)>(this->ndim))
            {
                // FOR VERTEX NODES KEEP THE DISPLACEMENT ZERO
                this->displacements_BC(this->index_nodes(j),0) = 0.;
                this->displacements_BC(this->index_nodes(j),1) = 0.;
                this->displacements_BC(this->index_nodes(j),2) = 0.;
            }
            else
            {
                auto gp_pnt_old = this->mesh_points.row(this->mesh_faces(this->listfaces[idir],j))/this->scale;
                auto xEq = gp_Pnt(gp_pnt_old(0),gp_pnt_old(1),gp_pnt_old(2));
                current_surface->D0(parametric_surface(j,0),parametric_surface(j,1),xEq);

                // TRY PROJECTION AS WELL TO RESOLVE FOR INCORRECT NODES
                auto xEq_Orthogonal = gp_Pnt(gp_pnt_old(0)*this->scale,gp_pnt_old(1)*this->scale,gp_pnt_old(2)*this->scale);

                try
                {
                    GeomAPI_ProjectPointOnSurf proj;
                    proj.Init(xEq_Orthogonal,current_surface);
                    Real ux, vx;
                    proj.LowerDistanceParameters(ux,vx);
                    current_surface->D0(ux,vx,xEq_Orthogonal);
                }
                catch (StdFail_NotDone)
                {
                    warn("Could not project node to the right surface. "
                         "Surface ID is:", id_surface, "  Surface type is:", this->geometry_surfaces_types[id_surface],
                         "  Node number is:", this->mesh_faces(this->listfaces[idir],j));
                }

                auto Xdisp_arc = xEq.X()/this->scale - gp_pnt_old(0);
                auto Ydisp_arc = xEq.Y()/this->scale - gp_pnt_old(1);
                auto Zdisp_arc = xEq.Z()/this->scale - gp_pnt_old(2);

                auto Xdisp_orth = xEq_Orthogonal.X()/this->scale - gp_pnt_old(0);
                auto Ydisp_orth = xEq_Orthogonal.Y()/this->scale - gp_pnt_old(1);
                auto Zdisp_orth = xEq_Orthogonal.Z()/this->scale - gp_pnt_old(2);

                if ( std::abs(Xdisp_arc - Xdisp_orth)/std::abs(Xdisp_orth) > OrthTol)
                {
                    Xdisp_arc = Xdisp_orth;
                }
                if ( std::abs(Ydisp_arc - Ydisp_orth)/std::abs(Ydisp_orth) > OrthTol)
                {
                    Ydisp_arc = Ydisp_orth;
                }
                if ( std::abs(Zdisp_arc - Zdisp_orth)/std::abs(Zdisp_orth) > OrthTol)
                {
                    Zdisp_arc = Zdisp_orth;
                }

                // FOR NON-VERTEX NODES GET THE REQUIRED DISPLACEMENT
                this->displacements_BC(this->index_nodes(j),0) = Xdisp_arc;
                this->displacements_BC(this->index_nodes(j),1) = Ydisp_arc;
                this->displacements_BC(this->index_nodes(j),2) = Zdisp_arc;

//                this->displacements_BC(this->index_nodes(j),0) = Xdisp_orth;
//                this->displacements_BC(this->index_nodes(j),1) = Ydisp_orth;
//                this->displacements_BC(this->index_nodes(j),2) = Zdisp_orth;

//                print(gp_pnt_old(0),gp_pnt_old(1),gp_pnt_old(2));
            }
        }
        this->index_nodes = ((this->index_nodes).array()+no_face_nodes).eval().matrix();
    }
//    print(this->displacements_BC);
//    print(this->displacements_BC.maxCoeff());
}

std::vector<Boolean> PostMeshSurface::FindPlanarSurfaces()
{
    std::vector<Boolean> isplanar;
    for (auto &current_surface: this->geometry_surfaces)
    {
        GeomLib_IsPlanarSurface surface_eval(current_surface,1.0e-7);
        isplanar.push_back(surface_eval.IsPlanar());
    }
    return isplanar;
}

std::vector<std::vector<Integer> > PostMeshSurface::GetMeshFacesOnPlanarSurfaces()
{
    std::vector<Boolean> isplanar = this->FindPlanarSurfaces();
    std::vector<std::vector<Integer> > planar_faces(2);
    for (auto idir=0; idir< this->dirichlet_faces.rows(); ++idir)
    {
        if (isplanar[this->dirichlet_faces(idir,3)] == True)
        {
            planar_faces[0].push_back(this->listfaces[idir]);
            planar_faces[1].push_back(this->dirichlet_faces(idir,3));
        }
    }
    return planar_faces;
}

void PostMeshSurface::GetBoundaryPointsOrder()
{
    std::vector<Integer> edge0, edge1, edge2;
    this->InferInterpolationPolynomialDegree();
    auto C = this->degree - 1;
    for (UInteger i=0; i<C; ++i)
    {
        edge0.push_back(i+3);
        edge1.push_back(C+3 +i*(C+1) -i*(i-1)/2 );
        edge2.push_back(2*C+3 +i*C -i*(i-1)/2 );
    }
    edge0.insert(edge0.begin(),0);
    edge0.push_back(1);

    edge1.insert(edge1.begin(),0);
    edge1.push_back(2);

    edge2.insert(edge2.begin(),1);
    edge2.push_back(2);

    std::vector<std::vector<Integer>> edges_order{edge0,edge1,edge2};
    this->boundary_faces_order = cnp::toEigen(edges_order);
}

std::vector<Integer> PostMeshSurface::GetDirichletFaces()
{
    //! RETURNS MESH FACES THAT NEED TO BE PROJECTED TO CAD SURFACES AND A FLAG
    //! (LAST COLUMN) WHICH STATING WHICH FACE BELONGS TO WHICH SURFACE
    std::vector<Integer> dirichlet_faces_stl;
    dirichlet_faces_stl.assign(this->dirichlet_faces.data(),
                               this->dirichlet_faces.data()+this->dirichlet_faces.rows()*this->dirichlet_faces.cols());
    return dirichlet_faces_stl;
}
