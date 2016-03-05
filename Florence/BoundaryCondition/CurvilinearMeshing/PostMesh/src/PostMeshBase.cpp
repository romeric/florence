
#include <PostMeshBase.hpp>

PostMeshBase::PostMeshBase(const PostMeshBase& other) \
    noexcept(std::is_copy_constructible<PostMeshBase>::value): \
    scale(other.scale), condition(other.condition), \
    projection_precision(other.projection_precision)
{
    // COPY CONSTRUCTOR
    this->mesh_element_type = other.mesh_element_type;
    this->ndim = other.ndim;
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
}

PostMeshBase& PostMeshBase::operator=(const PostMeshBase& other) \
noexcept(std::is_copy_assignable<PostMeshBase>::value)
{
    // COPY ASSIGNMENT OPERATOR
    this->scale = other.scale;
    this->condition = other.condition;
    this->projection_precision = other.projection_precision;

    this->mesh_element_type = other.mesh_element_type;
    this->ndim = other.ndim;
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

    return *this;
}

PostMeshBase::PostMeshBase(PostMeshBase&& other) noexcept :  \
    scale(other.scale), condition(other.condition), \
    projection_precision(other.projection_precision)
{
    // MOVE CONSTRUCTOR
    this->mesh_element_type = other.mesh_element_type;
    this->ndim = other.ndim;
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

    //! NB: CHECK THAT YOUR VERSION OF EIGEN SUPPORTS RVALUE REFERENCES
    //! (EIGEN_HAVE_RVALUE_REFERENCES). In PostMesh this is activated by default.
    //! ACTIVATING/DEACTIVATING DOES NOT AFFECT THE CODE IN ANY WAY COMPATIBILITY-WISE
    //! AS THEN THE COPY CONSTRUCTOR WOULD KICK IN INSTEAD OF MOVE CONSTRUCTOR
}

PostMeshBase& PostMeshBase::operator=(PostMeshBase&& other) noexcept
{
    // MOVE ASSIGNMENT OPERATOR
    this->scale = other.scale;
    this->condition = other.condition;
    this->projection_precision = other.projection_precision;

    this->mesh_element_type = other.mesh_element_type;
    this->ndim = other.ndim;
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

    //! NB: CHECK THAT YOUR VERSION OF EIGEN SUPPORTS RVALUE REFERENCES
    //! (EIGEN_HAVE_RVALUE_REFERENCES). In PostMesh this is activated by default.
    //! ACTIVATING/DEACTIVATING DOES NOT AFFECT THE CODE IN ANY WAY COMPATIBILITY-WISE
    //! AS THEN THE COPY CONSTRUCTOR WOULD KICK IN INSTEAD OF MOVE CONSTRUCTOR

    return *this;
}



void PostMeshBase::ReadIGES(const char* filename)
{
    //! IGES FILE READER BASED ON OCC BACKEND
    //! THIS FUNCTION CAN BE EXPANDED FURTHER TO TAKE CURVE/SURFACE CONSISTENY INTO ACCOUNT
    //! http://www.opencascade.org/doc/occt-6.7.0/overview/html/user_guides__iges.html

    IGESControl_Reader reader;
    reader.ReadFile(filename);
    // CHECK FOR IMPORT STATUS
    reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
    reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
    // READ IGES FILE AS-IS
    Interface_Static::SetIVal("read.iges.bspline.continuity",0);
    Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
    if (ic !=0)
    {
        std::cerr << "IGES file was not read as-is. The file was not read/transformed correctly\n";
    }

    // FORCE UNITS (NONE SEEM TO WORK)
    //Interface_Static::SetIVal("xstep.cascade.unit",0);
    //Interface_Static::SetIVal("read.scale.unit",0);

    // IF ALL OKAY, THEN TRANSFER ROOTS
    reader.TransferRoots();

    this->imported_shape  = reader.OneShape();
    this->no_of_shapes = reader.NbShapes();
}

void PostMeshBase::ReadSTEP(const char* filename)
{
    //! IGES FILE READER BASED ON OCC BACKEND
    //! THIS FUNCTION CAN BE EXPANDED FURTHER TO TAKE CURVE/SURFACE CONSISTENY INTO ACCOUNT
    //! http://www.opencascade.org/doc/occt-6.7.0/overview/html/user_guides__iges.html

    STEPControl_Reader reader;
    reader.ReadFile(filename);
    // CHECK FOR IMPORT STATUS
    reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
    reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
    // READ IGES FILE AS-IS
    Interface_Static::SetIVal("read.iges.bspline.continuity",0);
    Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
    if (ic !=0)
    {
        std::cerr << "STEP file was not read as-is. The file was not read/transformed correctly\n";
    }

    // FORCE UNITS (NONE SEEM TO WORK)
    //Interface_Static::SetIVal("xstep.cascade.unit",0);
    //Interface_Static::SetIVal("read.scale.unit",0);

    // IF ALL OKAY, THEN TRANSFER ROOTS
    reader.TransferRoots();

    this->imported_shape  = reader.OneShape();
    this->no_of_shapes = reader.NbShapes();

    //auto edges = reader.GiveList("iges-faces");

}

Eigen::MatrixI PostMeshBase::Read(std::string &filename)
{
    //! Reading 1D integer arrays
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        warn("Unable to read file");
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();

    const Integer rows = arr.size();
    const Integer cols = 1;

    Eigen::MatrixI out_arr = Eigen::MatrixI::Zero(rows,cols);

    for(Integer i=0 ; i<rows;i++)
    {
        out_arr(i) = std::atof(arr[i].c_str());
    }

    // CHECK IF LAST LINE IS READ CORRECTLY
    if ( out_arr(out_arr.rows()-2)==out_arr(out_arr.rows()-1) )
    {
        out_arr = out_arr.block(0,0,out_arr.rows()-1,1).eval();
    }

    return out_arr;
}

Eigen::MatrixUI PostMeshBase::ReadI(std::string &filename, char delim)
{
    /*Reading 2D integer row major arrays */
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        warn("Unable to read file");
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();


    const Integer rows = arr.size();
    const Integer cols = (split(arr[0], delim)).size();


    Eigen::MatrixUI out_arr = Eigen::MatrixUI::Zero(rows,cols);

    for(Integer i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for(Integer j=0 ; j<cols;j++)
        {
            out_arr(i,j) = std::atof(elems[j].c_str());
        }
    }

    // CHECK IF LAST LINE IS READ CORRECTLY
    bool duplicate_rows = False;
    for (Integer j=0; j<cols; ++j)
    {
        if ( out_arr(out_arr.rows()-2,j)==out_arr(out_arr.rows()-1,j) )
        {
            duplicate_rows = True;
        }
        else
        {
            duplicate_rows = False;
        }
    }
    if (duplicate_rows == True)
    {
        out_arr = out_arr.block(0,0,out_arr.rows()-1,out_arr.cols()).eval();
    }

    return out_arr;
}

Eigen::MatrixR PostMeshBase::ReadR(std::string &filename, char delim)
{
    /*Reading 2D floating point row major arrays */
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        warn("Unable to read file");
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();


    const Integer rows = arr.size();
    const Integer cols = (split(arr[0], delim)).size();


    Eigen::MatrixR out_arr = Eigen::MatrixR::Zero(rows,cols);

    for(Integer i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for(Integer j=0 ; j<cols;j++)
        {
            out_arr(i,j) = std::atof(elems[j].c_str());
        }
    }

    // CHECK IF LAST LINE IS READ CORRECTLY
    bool duplicate_rows = False;
    for (Integer j=0; j<cols; ++j)
    {
        if ( out_arr(out_arr.rows()-2,j)==out_arr(out_arr.rows()-1,j) )
        {
            duplicate_rows = True;
        }
        else
        {
            duplicate_rows = False;
        }
    }
    if (duplicate_rows == True)
    {
        out_arr = out_arr.block(0,0,out_arr.rows()-1,out_arr.cols()).eval();
    }

    return out_arr;
}

void PostMeshBase::CheckMesh()
{
    /* CHECKS IF MESH IS IMPORTED CORRECTLY */

    // CHECK FOR DUPLICATED LINE COPIES IN ELEMENTS, POINTS, EDGES AND FACES
    double check_duplicated_rows;
    int flag_p = 0;
    for (int i=this->mesh_points.rows()-2; i<this->mesh_points.rows();i++)
    {
        check_duplicated_rows = (this->mesh_points.row(i) - this->mesh_points.row(i-1)).norm();
        if (std::abs(check_duplicated_rows)  < 1.0e-14)
        {
            flag_p = 1;
        }
    }
    if (flag_p == 1)
    {
        Eigen::MatrixI a_rows = cnp::arange(this->mesh_points.rows()-1);
        Eigen::MatrixI a_cols = cnp::arange(this->mesh_points.cols());
        this->mesh_points = cnp::take(this->mesh_points,a_rows,a_cols);
    }

    // ELEMENTS
    int flag_e = 0;
    for (int i=this->mesh_elements.rows()-2; i<this->mesh_elements.rows();i++)
    {
        check_duplicated_rows = (this->mesh_elements.row(i) - this->mesh_elements.row(i-1)).norm();
        if (std::abs(check_duplicated_rows)  < 1.0e-14)
        {
            flag_e = 1;
        }
    }
    if (flag_e == 1)
    {
        Eigen::MatrixI a_rows = cnp::arange(this->mesh_elements.rows()-1);
        Eigen::MatrixI a_cols = cnp::arange(this->mesh_elements.cols());
        this->mesh_elements = cnp::take(this->mesh_elements,a_rows,a_cols);
    }

    // EDGES
    int flag_ed = 0;
    for (Integer i=this->mesh_edges.rows()-2; i<this->mesh_edges.rows();i++)
    {
        check_duplicated_rows = (this->mesh_edges.row(i) - this->mesh_edges.row(i-1)).norm();
        if (std::abs(check_duplicated_rows)  < 1.0e-14)
        {
            flag_ed = 1;
        }
    }
    if (flag_ed == 1)
    {
        Eigen::MatrixI a_rows = cnp::arange(this->mesh_edges.rows()-1);
        Eigen::MatrixI a_cols = cnp::arange(this->mesh_edges.cols());
        this->mesh_edges = cnp::take(this->mesh_edges,a_rows,a_cols);
    }

    // FACES FOR 3D
    if (this->mesh_faces.cols()!=0)
    {
        Integer flag_f = 0;
        for (Integer i=this->mesh_faces.rows()-2; i<this->mesh_faces.rows();i++)
        {
            check_duplicated_rows = (this->mesh_faces.row(i) - this->mesh_faces.row(i-1)).norm();
            if (std::abs(check_duplicated_rows)  < 1.0e-14)
            {
                flag_f = 1;
            }
        }
        if (flag_f == 1)
        {
            Eigen::MatrixI a_rows = cnp::arange(this->mesh_faces.rows()-1);
            Eigen::MatrixI a_cols = cnp::arange(this->mesh_faces.cols());
            this->mesh_faces = cnp::take(this->mesh_faces,a_rows,a_cols);
        }
    }

    Integer maxelem = this->mesh_elements.maxCoeff();
    Integer maxpoint = this->mesh_points.rows();

    if (maxelem+1 != maxpoint)
    {
        throw std::invalid_argument("Element connectivity and nodal coordinates do not match. This can be "
                                    "caused by giving two files which do not correspond to each other");
    }

    std::cout << "All good with imported mesh. proceeding..." << std::endl;

}

void PostMeshBase::GetGeomVertices()
{
    if (!this->geometry_points.empty())
        return;

    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_VERTEX); explorer.More(); explorer.Next())
    {
        // GET THE VERTICES LYING ON THE IMPORTED TOPOLOGICAL SHAPE
        TopoDS_Vertex current_vertex = TopoDS::Vertex(explorer.Current());
        gp_Pnt current_vertex_point = BRep_Tool::Pnt(current_vertex);

        this->geometry_points.push_back(current_vertex_point);
    }
}

void PostMeshBase::GetGeomEdges()
{
    //!  ITERATE OVER TopoDS_Shape AND EXTRACT ALL THE EDGES. CONVERT THE EDGES TO Geom_Curve AND
    //! GET THEIR HANDLES

    if (!this->geometry_curves.empty())
        return;

    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_EDGE); explorer.More(); explorer.Next())
    {
        // GET THE EDGES
        TopoDS_Edge current_edge = TopoDS::Edge(explorer.Current());
        // CONVERT THEM TO GEOM_CURVE
        Real first, last;
        Handle_Geom_Curve curve = BRep_Tool::Curve(current_edge,first,last);
        // STORE HANDLE IN THE CONTAINER
        this->geometry_curves.push_back(curve);

        // TO GET TYPE OF THE CURVE - UNLIKE GeomAdaptor_Curve, BRepAdaptor_Curve
        // DOES NOT THROW BUT RETURNS GeomAbs_OtherCurve FOR UNKNOWN TYPE OF CURVES
        BRepAdaptor_Curve adapt_curve(current_edge);
        // STORE TYPE OF CURVE (CURVE TYPES ARE DEFINED IN OCC_INC.hpp)
        this->geometry_curves_types.push_back(adapt_curve.GetType());
    }
}

void PostMeshBase::GetGeomFaces()
{
    //!  ITERATE OVER TopoDS_Shape AND EXTRACT ALL THE EDGES. CONVERT THE EDGES TO Geom_Surface AND
    //! GET THEIR HANDLES

    if (!this->geometry_surfaces.empty() && !this->topo_faces.empty())
        return;

    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_FACE); explorer.More(); explorer.Next())
    {

        // GET THE FACES
        TopoDS_Face current_face = TopoDS::Face(explorer.Current());
        // STORE
        this->topo_faces.push_back(current_face);
        // CONVERT THEM TO GEOM_SURFACE
        Handle_Geom_Surface surface = BRep_Tool::Surface(current_face);
        // STORE HANDLE IN THE CONTAINER
        this->geometry_surfaces.push_back(surface);

        // TO GET TYPE OF THE SURFACE. UNLIKE GeomAdaptor_Surface, BRepAdaptor_Surface
        // DOES NOT THROW BUT RETURNS GeomAbs_OtherSurface FOR UNKNOWN TYPE OF SURFACES
        BRepAdaptor_Surface adapt_surface(current_face);
        // STORE TYPE OF SURFACE (SURFACE TYPES ARE DEFINED IN OCC_INC.hpp)
        this->geometry_surfaces_types.push_back(adapt_surface.GetType());

    }
}

std::vector<Real> PostMeshBase::ObtainGeomVertices()
{
    std::vector<Real> geom_points;
    geom_points.clear();
    // PASS TO CYTHON
    for (UInteger i=0; i<this->geometry_points.size(); ++i)
    {
        geom_points.push_back(this->geometry_points[i].X());
        geom_points.push_back(this->geometry_points[i].Y());
        geom_points.push_back(this->geometry_points[i].Z());
    }

    return geom_points;
}

void PostMeshBase::ComputeProjectionCriteria()
{
    // IF NOT INITIALISED THEN COMPUTE
    assert((this->ndim==2 || this->ndim==3) && "Unknown number of dimensions");

    if (this->projection_criteria.rows()==0)
    {
        this->projection_criteria.setZero(mesh_edges.rows(),mesh_edges.cols());
        for (Integer iedge=0; iedge<mesh_edges.rows(); iedge++)
        {
            // GET THE COORDINATES OF THE TWO END NODES
            auto x1 = this->mesh_points(this->mesh_edges(iedge,0),0);
            auto y1 = this->mesh_points(this->mesh_edges(iedge,0),1);
            auto x2 = this->mesh_points(this->mesh_edges(iedge,1),0);
            auto y2 = this->mesh_points(this->mesh_edges(iedge,1),1);

            // GET THE MIDDLE POINT OF THE EDGE
            auto x_avg = ( x1 + x2 )/2.;
            auto y_avg = ( y1 + y2 )/2.;

            auto cond = std::sqrt(x_avg*x_avg+y_avg*y_avg);

            Real z1,z2,z3,x3,y3,z_avg;
            if (this->ndim==3) {
                z1 = this->mesh_points(this->mesh_edges(iedge,0),2);
                z2 = this->mesh_points(this->mesh_edges(iedge,1),2);

                x3 = this->mesh_points(this->mesh_edges(iedge,2),0);
                y3 = this->mesh_points(this->mesh_edges(iedge,2),1);
                z3 = this->mesh_points(this->mesh_edges(iedge,2),2);

                x_avg = (x1 + x2 + x3 )/3.;
                y_avg = (y1 + y2 + y3 )/3.;
                z_avg = (z1 + z2 + z3 )/3.;

                cond = std::sqrt(x_avg*x_avg+y_avg*y_avg+z_avg*z_avg);
            }

            if (cond<this->condition)
            {
                projection_criteria(iedge)=1;
            }
        }
    }
}

DirichletData PostMeshBase::GetDirichletData()
{
    // OBTAIN DIRICHLET DATA
    DirichletData Dirichlet_data;
    // CONVERT FROM EIGEN TO STL VECTOR
    std::vector<Integer> nodes_Dirichlet_data_stl;
    nodes_Dirichlet_data_stl.assign(this->nodes_dir.data(),this->nodes_dir.data()+this->nodes_dir.rows());
    // FIND UNIQUE VALUES OF DIRICHLET DATA
    // std::vector<UInteger> idx;
    // std::tie(std::ignore,idx) = cnp::unique(nodes_Dirichlet_data_stl);
    std::vector<Integer> idx;
    std::tie(std::ignore,idx) = cnp::unique(nodes_Dirichlet_data_stl,true);

    Dirichlet_data.nodes_dir_out_stl.resize(idx.size());
    Dirichlet_data.displacement_BC_stl.resize(this->ndim*idx.size());
    Dirichlet_data.nodes_dir_size = idx.size();

    for (UInteger i=0; i<idx.size(); ++i)
    {
        Dirichlet_data.nodes_dir_out_stl[i] = nodes_Dirichlet_data_stl[idx[i]];
        for (UInteger j=0; j<this->ndim; ++j)
        {
            Dirichlet_data.displacement_BC_stl[this->ndim*i+j] = this->displacements_BC(idx[i],j);
        }
    }

    return Dirichlet_data;
}
