
#include <PostMeshCurve.hpp>


using namespace std;


PostMeshCurve::PostMeshCurve(const PostMeshCurve& other) \
noexcept(std::is_copy_constructible<PostMeshCurve>::value)
    : PostMeshBase(other)
    {
        // COPY CONSTRUCTOR
        this->ndim = other.ndim;
        this->geometry_points_on_curves = other.geometry_points_on_curves;
        this->geometry_curves_bspline = other.geometry_curves_bspline;
        this->boundary_points_order = other.boundary_points_order;
        this->boundary_edges_order = other.boundary_edges_order;
        this->curve_to_parameter_scale_U = other.curve_to_parameter_scale_U;
        this->curves_parameters = other.curves_parameters;
        this->curves_lengths = other.curves_lengths;
        // REMAINING MEMBERS ARE COPY CONSTRUCTED BY BASE
    }

PostMeshCurve& PostMeshCurve::operator=(const PostMeshCurve& other) \
noexcept(std::is_copy_assignable<PostMeshCurve>::value)
    {
        // COPY ASSIGNMENT OPERATOR
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
        this->geometry_points_on_curves = other.geometry_points_on_curves;
        this->geometry_curves_bspline = other.geometry_curves_bspline;
        this->boundary_points_order = other.boundary_points_order;
        this->boundary_edges_order = other.boundary_edges_order;
        this->curve_to_parameter_scale_U = other.curve_to_parameter_scale_U;
        this->curves_parameters = other.curves_parameters;
        this->curves_lengths = other.curves_lengths;

        return *this;
    }

PostMeshCurve::PostMeshCurve(PostMeshCurve&& other) noexcept :
    PostMeshBase(std::move(other))
    {
        // MOVE CONSTRUCTOR
        this->ndim = other.ndim;
        this->geometry_points_on_curves = std::move(other.geometry_points_on_curves);
        this->geometry_curves_bspline = std::move(other.geometry_curves_bspline);
        this->boundary_points_order = std::move(other.boundary_points_order);
        this->boundary_edges_order = std::move(other.boundary_edges_order);
        this->curve_to_parameter_scale_U = std::move(other.curve_to_parameter_scale_U);
        this->curves_parameters = std::move(other.curves_parameters);
        this->curves_lengths = std::move(other.curves_lengths);
        // REMAINING MEMBERS ARE MOVE CONSTRUCTED BY BASE
    }

PostMeshCurve& PostMeshCurve::operator=(PostMeshCurve&& other) noexcept
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
        this->geometry_points_on_curves = std::move(other.geometry_points_on_curves);
        this->geometry_curves_bspline = std::move(other.geometry_curves_bspline);
        this->boundary_points_order = std::move(other.boundary_points_order);
        this->boundary_edges_order = std::move(other.boundary_edges_order);
        this->curve_to_parameter_scale_U = std::move(other.curve_to_parameter_scale_U);
        this->curves_parameters = std::move(other.curves_parameters);
        this->curves_lengths = std::move(other.curves_lengths);

        return *this;
    }

void PostMeshCurve::InferInterpolationPolynomialDegree()
{
    //! WORKS ONLY FOR TRIS
    for (auto i=1; i<50;++i)
    {
        if ( (i+1)*(i+2)/2 == this->mesh_elements.cols())
        {
            this->degree = i;
            break;
        }
    }
}

void PostMeshCurve::GetCurvesParameters()
{
    //! Gets first and last curve parameters and checks if they are consecutive
    this->curves_parameters = Eigen::MatrixR::Zero(this->geometry_curves.size(),2);
    for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        Handle_Geom_Curve current_curve = this->geometry_curves[icurve];
        curves_parameters(icurve,0) = current_curve->FirstParameter();
        curves_parameters(icurve,1) = current_curve->LastParameter();
    }
}

void PostMeshCurve::GetCurvesLengths()
{
    //! Computes length of all curves
    this->curves_lengths = Eigen::MatrixR::Zero(this->geometry_curves.size(),1);
    for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        Handle_Geom_Curve current_curve = this->geometry_curves[icurve];
        curves_lengths(icurve) = cnp::length(current_curve,1);
    }
}

void PostMeshCurve::FindCurvesSequentiallity()
{
    //! CHECKS IF THE PARAMETRIC REPRESENTATION OF CURVES ARE SEQUENTIAL, I.E.
    //! IF THE LastParameter OF ONE CURVE IS EQUAL TO THE FirstParameter OF THE SECOND CURVE.
    //! GENERALLY TRUE FOR BSPLINE CURVES. NOT TESTED
    auto Standard_Tolerance = 1.0e-14;
    std::vector<Integer> consecutive_curves; consecutive_curves.clear();
    Eigen::MatrixI curves_sequentiallity = -Eigen::MatrixI::Ones(this->geometry_curves.size(),3);
    this->GetCurvesParameters();

    for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        curves_sequentiallity(icurve,0)=icurve;
        Real current_last_parameter = this->curves_parameters(icurve,1);
        Real current_first_parameter = this->curves_parameters(icurve,0);
        for (UInteger jcurve=0; jcurve<this->geometry_curves.size() && jcurve!=icurve; ++jcurve)
        {
           if ((this->curves_parameters(jcurve,0)-current_last_parameter)<1e-13)
           {
               gp_Pnt pnt_1, pnt_2, pnt_3, pnt_4;
               this->geometry_curves[icurve]->D0(current_first_parameter,pnt_1);
               this->geometry_curves[jcurve]->D0(this->curves_parameters(jcurve,1),pnt_2);
               if ( abs(pnt_1.X()-pnt_2.X())< Standard_Tolerance && \
                    abs(pnt_1.Y()-pnt_2.Y())<Standard_Tolerance && \
                    abs(pnt_1.Z()-pnt_2.Z())<Standard_Tolerance )
               {
                   consecutive_curves.push_back(jcurve);
               }

               this->geometry_curves[icurve]->D0(current_last_parameter,pnt_3);
               this->geometry_curves[jcurve]->D0(this->curves_parameters(jcurve,0),pnt_4);
               if ( abs(pnt_3.X()-pnt_4.X())< Standard_Tolerance && \
                    abs(pnt_3.Y()-pnt_4.Y())<Standard_Tolerance && \
                    abs(pnt_3.Z()-pnt_4.Z())<Standard_Tolerance )
               {
                   consecutive_curves.push_back(jcurve);
               }
           }
        }
    }
}

void PostMeshCurve::ConcatenateSequentialCurves()
{
}

void PostMeshCurve::GetGeomPointsOnCorrespondingEdges()
{
    this->geometry_points_on_curves.clear();

    for (TopExp_Explorer explorer_edge(this->imported_shape,TopAbs_EDGE); explorer_edge.More(); explorer_edge.Next())
    {
        // GET THE EDGES
        TopoDS_Edge current_edge = TopoDS::Edge(explorer_edge.Current());

        Eigen::MatrixR current_edge_coords(2,3);
        Integer counter=0;
        for (TopExp_Explorer explorer_point(current_edge,TopAbs_VERTEX); explorer_point.More(); explorer_point.Next())
        {
            TopoDS_Vertex current_vertex = TopoDS::Vertex(explorer_point.Current());
            gp_Pnt current_vertex_point = BRep_Tool::Pnt(current_vertex);

            current_edge_coords(counter,0) = current_vertex_point.X();
            current_edge_coords(counter,1) = current_vertex_point.Y();
            current_edge_coords(counter,2) = current_vertex_point.Z();

            counter++;
        }
        this->geometry_points_on_curves.push_back(current_edge_coords);
    }
}

void PostMeshCurve::GetInternalCurveScale()
{
    this->curve_to_parameter_scale_U = Eigen::MatrixR::Zero(this->geometry_curves.size(),1);
    for (unsigned int icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        Handle_Geom_Curve current_curve = this->geometry_curves[icurve];
        if (this->geometry_curves_types[icurve]!=0)
        {
            //this->curve_to_parameter_scale_U(icurve) = current_curve->LastParameter()/cnp::length(current_curve,1/this->scale);
            this->curve_to_parameter_scale_U(icurve) = std::abs(current_curve->LastParameter() - \
                                                        current_curve->FirstParameter())/cnp::length(current_curve,1/this->scale);
        }
        else
        {
            // IF GEOMETRY TYPE IS A LINE
            this->curve_to_parameter_scale_U(icurve) = 2*current_curve->LastParameter()/cnp::length(current_curve,1/this->scale);
        }
    }
}

void PostMeshCurve::IdentifyCurvesContainingEdges()
{
    this->dirichlet_edges = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim+1);
    this->listedges.clear();
//    this->InferInterpolationPolynomialDegree();
    auto index_edge = 0;

    // LOOP OVER EDGES
    for (auto iedge=0; iedge<this->mesh_edges.rows(); ++iedge)
    {
        // GET THE COORDINATES OF THE TWO END NODES
        auto x1 = this->mesh_points(this->mesh_edges(iedge,0),0);
        auto y1 = this->mesh_points(this->mesh_edges(iedge,0),1);
        auto x2 = this->mesh_points(this->mesh_edges(iedge,1),0);
        auto y2 = this->mesh_points(this->mesh_edges(iedge,1),1);

        // GET THE MIDDLE POINT OF THE EDGE
        auto x_avg = ( x1 + x2 )/2.;
        auto y_avg = ( y1 + y2 )/2.;

        if (this->projection_criteria(iedge)==1)
        {
            this->listedges.push_back(iedge);
            for (UInteger iter=0;iter<ndim;++iter)
            {
               dirichlet_edges(index_edge,iter) = this->mesh_edges(iedge,iter);
            }

            // PROJECT IT OVER ALL CURVES
            auto min_mid_distance = 1.0e20;
            auto mid_distance = 1.0e10;
            gp_Pnt middle_point(x_avg,y_avg,0.0);

            // LOOP OVER CURVES
            for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
            {
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                try
                {
                    GeomAPI_ProjectPointOnCurve proj;
                    proj.Init(middle_point,this->geometry_curves[icurve]);
                    mid_distance = proj.LowerDistance();
                }
                catch (StdFail_NotDone)
                {
                    // StdFail_NotDone ISSUE - DO NOTHING
                }
                if (mid_distance < min_mid_distance)
                {
                    // STORE ID OF CURVES
//                    this->dirichlet_edges(iedge,2) = icurve;
                    this->dirichlet_edges(index_edge,2) = icurve; // <--THIS
//                    this->dirichlet_edges(index_edge,2) = this->geometry_curves_types[icurve]; // CHECK THIS
                    // RE-ASSIGN
                    min_mid_distance = mid_distance;
                }
            }
            index_edge +=1;
        }
    }

    auto arr_rows = cnp::arange(index_edge);
    auto arr_cols = cnp::arange(ndim+1);
    this->dirichlet_edges = cnp::take(this->dirichlet_edges,arr_rows,arr_cols);
//    print(this->dirichlet_edges);
//    print(this->geometry_curves.size());
//    print(this->geometry_curves_types);

//    for (auto & k : this->geometry_points)
//        print(k.X(),k.Y());

//    exit(EXIT_FAILURE);
}

void PostMeshCurve::ProjectMeshOnCurve()
{
    this->InferInterpolationPolynomialDegree();

    this->projection_U = Eigen::MatrixR::Zero(this->dirichlet_edges.rows(),this->ndim);

    // LOOP OVER EDGES
    for (Integer iedge=0; iedge<this->dirichlet_edges.rows(); ++iedge)
    {
        // LOOP OVER THE TWO END NODES OF THIS EDGE (VERTICES OF THE EDGE)
        for (UInteger inode=0; inode<this->ndim; ++inode)
        {
            // PROJECTION PARAMETER
            Real parameterU;
            // GET THE COORDINATES OF THE NODE
            auto x = this->mesh_points(this->dirichlet_edges(iedge,inode),0);
            auto y = this->mesh_points(this->dirichlet_edges(iedge,inode),1);
            // GET THE CURVE THAT THIS EDGE HAS TO BE PROJECTED TO
            UInteger icurve = this->dirichlet_edges(iedge,2);
            // GET THE COORDINATES OF CURVE'S TWO VERTICES
            auto x1_curve = this->geometry_points_on_curves[icurve](0,0);
            auto y1_curve = this->geometry_points_on_curves[icurve](0,1);
            auto x2_curve = this->geometry_points_on_curves[icurve](1,0);
            auto y2_curve = this->geometry_points_on_curves[icurve](1,1);
            // CHECK IF WE ARE ON THE CURVE INTERSECTION
            if (std::abs(x1_curve-x) < projection_precision && \
                    std::abs(y1_curve-y) < projection_precision && \
                    this->geometry_curves_types[icurve]!=0)
            {
                // PROJECT THE CURVE VERTEX INSTEAD OF THE EDGE NODE (THIS IS NECESSARY TO ENSURE SUCCESSFUL PROJECTION)
                x = x1_curve;
                y = y1_curve;
            }
            else if (std::abs(x2_curve-x) < projection_precision && \
                     std::abs(y2_curve-y) < projection_precision && \
                     this->geometry_curves_types[icurve]!=0)
            {
                x = x2_curve;
                y = y2_curve;
            }

            try
            {
                // GET THE NODE THAT HAS TO BE PROJECTED TO THE CURVE
                auto node_to_be_projected = gp_Pnt(x,y,0.0);
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                GeomAPI_ProjectPointOnCurve proj;
                proj.Init(node_to_be_projected,this->geometry_curves[icurve]);
                parameterU = proj.LowerDistanceParameter();
            }
            catch (StdFail_NotDone)
            {
                warn("The edge node was not projected to the right curve. Curve number: ",icurve);
            }

            // GET CURVE LENGTH
            auto curve_length = cnp::length(this->geometry_curves[icurve],1.);

            // STORE PROJECTION POINT PARAMETER ON THE CURVE (NORMALISED)
            //this->projection_U(iedge,inode) = this->scale*parameterU/curve_length; //# THIS
            auto U0 = this->geometry_curves_types[icurve]==0 ? 0. : this->geometry_curves[icurve]->FirstParameter();
            this->projection_U(iedge,inode) =   this->scale*(parameterU-U0)/curve_length;
        }
    }

    // SORT PROJECTED PARAMETERS OF EACH EDGE - MUST INITIALISE SORT INDICES
    this->sorted_projected_indices = Eigen::MatrixI::Zero(this->projection_U.rows(),this->projection_U.cols());

//    println(this->projection_U);
    cnp::sort_rows(this->projection_U,this->sorted_projected_indices);
//    print(this->projection_U);
//    print(sorted_projected_indices);
//    print(this->dirichlet_edges);

//    exit(EXIT_FAILURE);
}

void PostMeshCurve::RepairDualProjectedParameters()
{
    auto lengthTol = 1.0e-10;

    for (auto iedge=0;iedge<this->dirichlet_edges.rows();++iedge)
    {
        auto id_curve = this->dirichlet_edges(iedge,2);
        auto current_curve = this->geometry_curves[id_curve];
        // DUAL PROJECTION HAPPENS FOR CLOSED CURVES
        if (current_curve->IsClosed() || current_curve->IsPeriodic())
        {
            // GET THE CURRENT EDGE
            Eigen::Matrix<Real,1,2> current_edge_U = this->projection_U.row(iedge);
            // SORT IT
            std::sort(current_edge_U.data(),current_edge_U.data()+current_edge_U.cols());
            // GET THE FIRST AND LAST PARAMETERS OF THIS CURVE
            auto u1 = current_curve->FirstParameter()/cnp::length(current_curve,1.0/this->scale);
            auto u2 = current_curve->LastParameter()/cnp::length(current_curve,1.0/this->scale);

            // IF THE U PARAMETER FOR THE FIRST NODE OF THIS EDGE EQUALS TO THE FIRST PARAMETER
            if (std::abs(current_edge_U(0) - u1) < lengthTol )
            {
                 // WE ARE AT THE STARTING PARAMETER - GET THE LENGTH TO uc
                 auto uc = current_edge_U(1);
                 GeomAdaptor_Curve current_adapt_curve(current_curve);
                 auto l0c = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,uc)/this->scale;
                 auto l01 = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,u2)/this->scale;

                 if ( l0c > (l01-l0c))
                 {
                     // FIX THE PARAMETER TO LastParameter
                     this->projection_U(iedge,0) = u2;
                 }
            }
            else if ( abs(current_edge_U(1) - u2) < lengthTol )
            {
                 // WE ARE AT THE END PARAMETER - GET THE LENGTH TO uc
                 auto uc = current_edge_U(0);
                 GeomAdaptor_Curve current_adapt_curve(current_curve);
                 auto lc1 = GCPnts_AbscissaPoint::Length(current_adapt_curve,uc,u2)/this->scale;
                 auto l0c = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,uc)/this->scale;

                 if ( l0c < lc1 )
                 {
                     // FIX THE PARAMETER TO FirstParameter
                     this->projection_U(iedge,1) = u1;
                 }
            }
        }
    }

    // SORT projection_U AGAIN
//    print(this->projection_U);
    cnp::sort_back_rows(this->projection_U,this->sorted_projected_indices);
//    print(sorted_projected_indices);
//    print(this->projection_U);
//    exit(EXIT_FAILURE);

}

void PostMeshCurve::CurvesToBsplineCurves()
{
    /* Converts all the imported curves to bspline curves. Apart from circle all the remaining converted
     * curves will be non-periodic: http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
     */
    this->geometry_curves_bspline.clear();
    for (unsigned int icurve=0; icurve < this->geometry_curves.size(); ++icurve)
    {
        this->geometry_curves_bspline.push_back( GeomConvert::CurveToBSplineCurve(this->geometry_curves[icurve]) );
    }
}

void PostMeshCurve::MeshPointInversionCurve()
{
    this->no_dir_edges = this->dirichlet_edges.rows();
    Integer no_edge_nodes = this->mesh_edges.cols();
    Eigen::MatrixI arr_row = Eigen::Map<Eigen::Matrix<Integer,Eigen::Dynamic,1> >(this->listedges.data(),this->listedges.size());
    auto arr_col = cnp::arange(0,no_edge_nodes);
    this->nodes_dir = cnp::take(this->mesh_edges,arr_row,arr_col);
    this->nodes_dir = cnp::ravel(this->nodes_dir);
    this->index_nodes = cnp::arange(no_edge_nodes);
    this->displacements_BC = Eigen::MatrixR::Zero(this->no_dir_edges*no_edge_nodes,this->ndim);

    // FIND CURVE LENGTH AND LAST PARAMETER SCALE
    this->GetInternalCurveScale();
    this->EstimatedParameterUOnMesh();

//    print(this->u_of_all_fekete_mesh_edges);
//    print(boundary_edges_order);
    this->GetCurvesLengths();
    this->GetCurvesParameters();
//    println(this->curves_parameters);


    for (auto idir=0; idir< this->no_dir_edges; ++idir)
    {
        auto id_curve = this->dirichlet_edges(idir,2);
        Handle_Geom_Curve current_curve = this->geometry_curves[id_curve];
        auto length_current_curve = cnp::length(current_curve,1/this->scale);
        auto internal_scale = 1./this->curve_to_parameter_scale_U(id_curve);
        GeomAdaptor_Curve current_curve_adapt(current_curve);

        for (auto j=0; j<no_edge_nodes;++j)
        {
            auto tol =1e-10;
            auto U0 = this->geometry_curves_types[id_curve]==0 ? 0. : current_curve_adapt.FirstParameter();
            GCPnts_AbscissaPoint inv;
            Real uEq;
            gp_Pnt xEq;

//            inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,
//                                       internal_scale*this->u_of_all_fekete_mesh_edges(idir,
//                                       this->boundary_edges_order(this->listedges[idir],j))*this->scale,0.);

            if ( (this->scale-1.)<1e-14)
            {
                inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,
                                           internal_scale*curves_lengths(id_curve)*this->u_of_all_fekete_mesh_edges(idir,
                                           this->boundary_edges_order(this->listedges[idir],j))*this->scale,U0);

                uEq = inv.Parameter();
                current_curve_adapt.D0(uEq,xEq);
            }
            else
            {
                inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,
                                           internal_scale*this->u_of_all_fekete_mesh_edges(idir,
                                           this->boundary_edges_order(this->listedges[idir],j))*this->scale,U0);

                uEq = inv.Parameter();
                current_curve_adapt.D0(uEq*length_current_curve,xEq);
            }

            Eigen::MatrixR gp_pnt_old = (this->mesh_points.row(this->nodes_dir(this->index_nodes( j ))).array()/this->scale);

            this->displacements_BC(this->index_nodes(j),0) = (xEq.X()/this->scale - gp_pnt_old(0));
            this->displacements_BC(this->index_nodes(j),1) = (xEq.Y()/this->scale - gp_pnt_old(1));
            //this->displacements_BC(this->index_nodes(j),2) = (xEq.Z()/this->scale - gp_pnt_old(2)); // USEFULL FOR 3D CURVES

//            println(internal_scale*this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*this->scale );
//            println(internal_scale*curves_lengths(id_curve)*this->u_of_all_fekete_mesh_edges(idir,this->boundary_edges_order(this->listedges[idir],j))*this->scale);
//            println(curves_lengths.rows(),curves_lengths.cols());
//            println(this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*scale );
//            println( this->boundary_edges_order(this->listedges[idir],j) );
//            print(this->u_of_all_fekete_mesh_edges(idir,this->boundary_edges_order(this->listedges[idir],j))*this->scale);
//            println( internal_scale*this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*this->scale,uEq,id_curve);
//            cout << gp_pnt_old(0) << " " << gp_pnt_old(1) << "] [" << xEq.X()/this->scale << " " << xEq.Y()/this->scale << endl;


        }
        this->index_nodes = ((this->index_nodes).array()+no_edge_nodes).eval().matrix();
//        cout << " " << endl;
    }
//    cout << displacements_BC << endl;
}

Eigen::MatrixR PostMeshCurve::ParametricFeketePoints(Standard_Real &u1,Standard_Real &u2)
{
    Eigen::MatrixR fekete_on_curve;
    fekete_on_curve = (u1 + (u2-u1)/2.0*((this->fekete).array()+1.)).matrix();
    return fekete_on_curve;
}

void PostMeshCurve::GetElementsWithBoundaryEdgesTri()
{
    this->InferInterpolationPolynomialDegree();
    assert (this->degree!=2 && "YOU_TRIED_CALLING_A_TRIANGULAR_METHOD_ON_TETRAHEDRA");
    this->elements_with_boundary_edges = Eigen::MatrixI::Zero(this->mesh_edges.rows(),1);

    for (auto iedge=0; iedge<this->mesh_edges.rows();++iedge)
    {
        std::vector<Integer> all_rows; all_rows.clear();
        for (auto jedge=0; jedge<this->mesh_edges.cols();++jedge)
        {
            Eigen::MatrixI rows; //Eigen::MatrixI cols; not needed
            auto indices = cnp::where_eq(this->mesh_elements,this->mesh_edges(iedge,jedge));
            std::tie(rows,std::ignore) = indices;

            for (auto k=0; k<rows.rows(); ++k)
            {
               all_rows.push_back(rows(k));
            }
        }

        Eigen::MatrixI all_rows_eigen = Eigen::Map<Eigen::MatrixI>(all_rows.data(),all_rows.size(),1);
        for (auto i=0; i<all_rows_eigen.rows(); ++i)
        {
            Eigen::MatrixI rows_2;
            auto indices = cnp::where_eq(all_rows_eigen,all_rows_eigen(i));
            std::tie(rows_2,std::ignore) = indices;
            if (rows_2.rows()==this->mesh_edges.cols())
            {
                this->elements_with_boundary_edges(iedge) = all_rows_eigen(i);
                break;
            }
        }
    }
}

void PostMeshCurve::GetBoundaryPointsOrder()
{
    /*The order of boundary (edge/face) connectivity. Needs the degree to be inferred a priori */
    this->InferInterpolationPolynomialDegree();

    this->boundary_points_order = Eigen::MatrixI::Zero(this->fekete.rows(),this->fekete.cols());
    this->boundary_points_order(1) = this->fekete.rows()-1;
    this->boundary_points_order.block(2,0,fekete.rows()-2,1) = cnp::arange(1,fekete.rows()-1);

    // FOR ALL THE EDGES
    this->boundary_edges_order = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->mesh_edges.cols());
    boundary_edges_order.col(1) = (boundary_edges_order.cols()-1)*Eigen::MatrixI::Ones(boundary_edges_order.rows(),1);

    for (auto iedge=0;iedge<this->mesh_edges.rows();++iedge)
    {
        Eigen::MatrixI current_edge = this->mesh_edges.row(iedge).transpose();
        Eigen::MatrixI all_points_cols = cnp::arange(this->mesh_points.cols());
        Eigen::MatrixR current_edge_coordinates = cnp::take(this->mesh_points,current_edge,all_points_cols);
        current_edge_coordinates = (current_edge_coordinates.array()/1000.).matrix().eval();

        std::vector<Real> norm_rows(this->mesh_edges.cols()-2);
        for (auto j=2; j<current_edge_coordinates.rows(); ++j)
        {
            norm_rows[j-2] = (current_edge_coordinates.row(j).array() - \
                              current_edge_coordinates.row(1).array()).matrix().norm();
        }
        std::vector<Integer> sorted_idx = cnp::argsort(norm_rows);
        Eigen::MatrixI idx = Eigen::Map<Eigen::MatrixI>(sorted_idx.data(),1,sorted_idx.size());
        boundary_edges_order.block(iedge,2,1,sorted_idx.size()) = (idx.array()+1).matrix().reverse();
//            boundary_edges_order.block(iedge,2,1,sorted_idx.size()) = (idx.array()+1).matrix();
    }
}

void PostMeshCurve::EstimatedParameterUOnMesh()
{
    this->no_dir_edges = this->dirichlet_edges.rows();
    this->u_of_all_fekete_mesh_edges = Eigen::MatrixR::Zero(this->no_dir_edges,this->fekete.rows());

    for (auto idir=0; idir< this->no_dir_edges; ++idir)
    {
        Integer id_curve = this->dirichlet_edges(idir,2);
        auto u1 = this->projection_U(idir,0);
        auto u2 = this->projection_U(idir,1);
        Handle_Geom_Curve current_curve = this->geometry_curves[id_curve];
        u_of_all_fekete_mesh_edges.block(idir,0,1,this->fekete.rows()) = ParametricFeketePoints(u1,u2).transpose();
        if (current_curve->IsClosed())
        {
            // FIND THE SCALED LAST PARAMETER
            Real scaled_endU = this->curve_to_parameter_scale_U(id_curve);
            GeomAdaptor_Curve current_curve_adapt(current_curve);
            Real umin = u_of_all_fekete_mesh_edges(idir,0); //abs
            Real umax = u_of_all_fekete_mesh_edges(idir,this->fekete.rows()-1); //abs
            if (umin>umax)
            {
                Real temp = umin;
                umin = umax;
                umax = temp;
            }
            Real current_curve_length = cnp::length(current_curve,1.);
            umin *= current_curve_length/this->scale;
            umax *= current_curve_length/this->scale;
            Real length_right = GCPnts_AbscissaPoint::Length(current_curve_adapt,current_curve->FirstParameter(),umin);
            Real length_left = GCPnts_AbscissaPoint::Length(current_curve_adapt,umax,current_curve->LastParameter());
            Real edge_length = GCPnts_AbscissaPoint::Length(current_curve_adapt,umin,umax);

            if (length_left+length_right < edge_length)
            {
                assert ( (length_left+length_right) - edge_length > 1e-4 && "MESH_TOO_COARSE");
                // MESH BOUNDARY (EDGE/FACE) PASSES THROUGH THE ORIGIN
                umin = umin/current_curve_length*this->scale;
                umax = umax/current_curve_length*this->scale;
                Real trespassed_length = abs(this->scale*current_curve->FirstParameter()/current_curve_length-umin) + \
                        abs(scaled_endU-umax);

                u1 = umin;
                u2 = this->scale*current_curve->FirstParameter()/current_curve_length;
                Eigen::MatrixR right_one = (u1 - (trespassed_length)/2.0*((this->fekete).array()+1.)).matrix();

                u1 = umax;
                u2 = this->scale*current_curve->LastParameter()/current_curve_length;
                Eigen::MatrixR left_one = (u1 + (trespassed_length)/2.0*((this->fekete).array()+1.)).matrix();
                std::vector<Real> dum_1; dum_1.clear();
                for (Integer i=0; i<right_one.rows();++i)
                {
                    if (right_one(i) >= this->scale*current_curve->FirstParameter()/current_curve_length)
                    {
                        dum_1.push_back(right_one(i));
                    }
                }

                for (Integer i=left_one.rows()-1; i>=0;--i)
                {
                    if (left_one(i) < this->scale*current_curve->LastParameter()/current_curve_length)
                    {
                        dum_1.push_back(left_one(i));
                    }

                }
                Eigen::MatrixR dum_1_eigen = Eigen::Map<Eigen::MatrixR>(dum_1.data(),dum_1.size(),1);
                if (u_of_all_fekete_mesh_edges(idir,0) > u_of_all_fekete_mesh_edges(idir,1))
                {
                    dum_1_eigen = dum_1_eigen.reverse().eval();
                }
                this->u_of_all_fekete_mesh_edges.block(idir,0,1,this->fekete.rows()) = dum_1_eigen.transpose();
            }
        }
    }
}

PassToPython PostMeshCurve::GetDirichletData()
{
    PassToPython struct_to_python;
    struct_to_python.nodes_dir_size = this->nodes_dir.rows();
    // CONVERT FROM EIGEN TO STL VECTOR
    struct_to_python.nodes_dir_out_stl.assign(this->nodes_dir.data(),this->nodes_dir.data()+struct_to_python.nodes_dir_size);
    struct_to_python.displacement_BC_stl.assign(this->displacements_BC.data(),this->displacements_BC.data()+ \
                                                this->ndim*struct_to_python.nodes_dir_size);

    return struct_to_python;
}
