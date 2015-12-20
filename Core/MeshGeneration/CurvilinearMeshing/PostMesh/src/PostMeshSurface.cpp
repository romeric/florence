
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
    // Copy assignment operator
    this->mesh_elements = other.mesh_elements;
    this->mesh_points = other.mesh_points;
    this->mesh_edges = other.mesh_edges;
    this->mesh_faces = other.mesh_faces;
    this->projection_criteria = other.projection_criteria;
    this->degree = degree;
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
    this->degree = degree;
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
    //! WORKS ONLY FOR TETS
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
    //! Converts all the imported surfaces to bspline surfaces : http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
    this->geometry_surfaces_bspline.clear();
    for (unsigned int isurf=0; isurf < this->geometry_surfaces.size(); ++isurf)
    {
        this->geometry_surfaces_bspline.push_back( GeomConvert::SurfaceToBSplineSurface(this->geometry_surfaces[isurf]) );
    }
}

void PostMeshSurface::GetSurfacesParameters()
{
    //! Gets first and last curve parameters and checks if they are consecutive
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
    this->dirichlet_faces = Eigen::MatrixI::Zero(this->mesh_faces.rows(),this->ndim+1);
    this->listfaces.clear();
//    this->InferInterpolationPolynomialDegree();

    auto index_face = 0;

    // LOOP OVER FACES
//#pragma omp parallel for
    for (auto iface=0; iface<this->mesh_faces.rows(); ++iface)
    {
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

        // GET THE MIDDLE POINT OF THE FACE/TRIANGLE
        auto x_avg = ( x1 + x2 + x3 )/3.;
        auto y_avg = ( y1 + y2 + y3 )/3.;
        auto z_avg = ( z1 + z2 + z3 )/3.;

//        if (std::sqrt(x_avg*x_avg+y_avg*y_avg+z_avg*z_avg)< this->condition)
        if (this->projection_criteria(iface)==1)
        {
            this->listfaces.push_back(iface);
            for (UInteger iter=0;iter<ndim;++iter)
            {
               dirichlet_faces(index_face,iter) = this->mesh_faces(iface,iter);
            }

            // PROJECT IT OVER ALL CURVES
            auto min_mid_distance = 1.0e20;
            auto mid_distance = 1.0e10;
            gp_Pnt middle_point(x_avg,y_avg,z_avg);

            // LOOP OVER CURVES
//#pragma omp critical
            for (UInteger isurface=0; isurface<this->geometry_surfaces.size(); ++isurface)
            {
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                try
                {
                    GeomAPI_ProjectPointOnSurf proj;
                    proj.Init(middle_point,this->geometry_surfaces[isurface]);
                    mid_distance = proj.LowerDistance();
                }
                catch (StdFail_NotDone)
                {
                    // StdFail_NotDone ISSUE - DO NOTHING
                }
                if (mid_distance < min_mid_distance)
                {
                    // STORE ID OF CURVES
//                    this->dirichlet_faces(index_face,2) = isurface; // <--THIS
                    this->dirichlet_faces(index_face,3) = isurface; // <--THIS
                    // RE-ASSIGN
                    min_mid_distance = mid_distance;
                }
            }
            index_face +=1;
        }
    }

    // REDUCE THE MATRIX TO GET DIRICHLET FACES
    auto arr_rows = cnp::arange(index_face);
    auto arr_cols = cnp::arange(ndim+1);
    this->dirichlet_faces = cnp::take(this->dirichlet_faces,arr_rows,arr_cols);
//    print(dirichlet_faces);
//    print(dirichlet_faces.rows());
//    exit(EXIT_FAILURE);
}

void PostMeshSurface::ProjectMeshOnSurface()
{
    this->InferInterpolationPolynomialDegree();
    this->projection_U = Eigen::MatrixR::Zero(this->dirichlet_faces.rows(),this->ndim);
    this->projection_V = Eigen::MatrixR::Zero(this->dirichlet_faces.rows(),this->ndim);

    // LOOP OVER EDGES
    for (auto iface=0; iface<this->dirichlet_faces.rows(); ++iface)
    {
        // LOOP OVER THE THREE VERTICES OF THE FACE
        for (UInteger inode=0; inode<this->ndim; ++inode)
        {
            // PROJECTION PARAMETERS
            Real parameterU, parameterV;
            // GET THE COORDINATES OF THE NODE
            auto x = this->mesh_points(this->dirichlet_faces(iface,inode),0);
            auto y = this->mesh_points(this->dirichlet_faces(iface,inode),1);
            auto z = this->mesh_points(this->dirichlet_faces(iface,inode),2);
            // GET THE SURFACE THAT THIS FACE HAS TO BE PROJECTED TO
            auto isurface = this->dirichlet_faces(iface,3);
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

            try
            {
                // GET THE NODE THAT HAS TO BE PROJECTED TO THE CURVE
                auto node_to_be_projected = gp_Pnt(x,y,z);
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                GeomAPI_ProjectPointOnSurf proj;
                proj.Init(node_to_be_projected,this->geometry_surfaces[isurface]);
                proj.LowerDistanceParameters(parameterU,parameterV);
            }
            catch (StdFail_NotDone)
            {
                warn("The face node was not projected on to the right surface. Surface number: ",isurface);
            }

            // STORE PROJECTION POINT PARAMETER ON THE SURFACE (NORMALISED)
            this->projection_U(iface,inode) = parameterU;
            this->projection_V(iface,inode) = parameterV;
        }
    }

    // SORT PROJECTED PARAMETERS OF EACH EDGE - MUST INITIALISE SORT INDICES
    this->sorted_projected_indices.setZero(this->projection_U.rows(),this->projection_U.cols());

//    print(this->projection_U);
//    print(this->projection_V);
    cnp::sort_rows(this->projection_U,this->sorted_projected_indices);
    cnp::sort_rows(this->projection_V,this->sorted_projected_indices);

//    for (auto &k: this->geometry_points){
//        print("[",k.X(),",",k.Y(),",",k.Z(),"],");
//    }

//    exit(EXIT_FAILURE);
}

void PostMeshSurface::MeshPointInversionSurface()
{
    this->no_dir_faces = this->dirichlet_faces.rows();
    auto no_face_nodes = this->mesh_faces.cols();
    Eigen::MatrixI arr_row = Eigen::Map<Eigen::Matrix<Integer,Eigen::Dynamic,1> >
            (this->listfaces.data(),this->listfaces.size());
    auto arr_col = cnp::arange(0,no_face_nodes);
    this->nodes_dir = cnp::take(this->mesh_faces,arr_row,arr_col);
    this->nodes_dir = cnp::ravel(this->nodes_dir);
    this->index_nodes = cnp::arange(no_face_nodes);
    this->displacements_BC = Eigen::MatrixR::Zero(this->no_dir_faces*no_face_nodes,this->ndim);


    this->GetSurfacesParameters();

    for (auto idir=0; idir< this->no_dir_faces; ++idir)
    {
        auto id_surface = static_cast<Integer>(this->dirichlet_faces(idir,3));
        Handle_Geom_Surface current_surface = this->geometry_surfaces[id_surface];

        for (auto j=3; j<no_face_nodes;++j)
        {
            auto x = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),0);
            auto y = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),1);
            auto z = this->mesh_points(this->mesh_faces(this->listfaces[idir],j),2);


            // LOOP OVER ALL GEOMETRY POINTS AND IF POSSIBLE PICK THOSE INSTEAD
            for (auto &k : geometry_points)
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
            auto point_to_be_projected = gp_Pnt(x,y,z);
            GeomAPI_ProjectPointOnSurf proj;
            proj.Init(point_to_be_projected,current_surface,1e-06);
            proj.LowerDistanceParameters(uEq,vEq);

            auto xEq = gp_Pnt(0.,0.,0.);
            current_surface->D0(uEq,vEq,xEq);

            auto gp_pnt_old = (this->mesh_points.row(this->nodes_dir(this->index_nodes( j ))).array()/this->scale);

            this->displacements_BC(this->index_nodes(j),0) = (xEq.X()/this->scale - gp_pnt_old(0));
            this->displacements_BC(this->index_nodes(j),1) = (xEq.Y()/this->scale - gp_pnt_old(1));
            this->displacements_BC(this->index_nodes(j),2) = (xEq.Z()/this->scale - gp_pnt_old(2));

//            print(proj.NearestPoint().X()/this->scale, xEq.X()/this->scale,proj.Point(proj.NbPoints()).X()/this->scale);
//            print(proj.NearestPoint().X()/this->scale ,proj.Point(proj.NbPoints()).X()/this->scale);
//            print(gp_pnt_old(0),gp_pnt_old(1),gp_pnt_old(2),"   ",xEq.X()/this->scale,xEq.Y()/this->scale,xEq.Z()/this->scale);

        }
        this->index_nodes = ((this->index_nodes).array()+no_face_nodes).eval().matrix();
    }
//    cout << displacements_BC << endl;
//    print (displacements_BC.maxCoeff());
//    exit (EXIT_FAILURE);
}

void PostMeshSurface::GetInternalSurfaceScales()
{

}


PassToPython PostMeshSurface::GetDirichletData()
{
    PassToPython struct_to_python;
    struct_to_python.nodes_dir_size = this->nodes_dir.rows();
    // CONVERT FROM EIGEN TO STL VECTOR
    struct_to_python.nodes_dir_out_stl.assign(this->nodes_dir.data(),this->nodes_dir.data()+struct_to_python.nodes_dir_size);
    struct_to_python.displacement_BC_stl.assign(this->displacements_BC.data(),this->displacements_BC.data()+ \
                                                this->ndim*struct_to_python.nodes_dir_size);

//    cout << this->displacements_BC << endl;

    return struct_to_python;
}
