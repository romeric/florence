
#include <PostMeshBase.hpp>


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

//        Handle_TColStd_HSequenceOfTransient edges = reader.GiveList("iges-faces");

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

//        Handle_TColStd_HSequenceOfTransient edges = reader.GiveList("iges-faces");

}

Eigen::MatrixI PostMeshBase::Read(std::string &filename)
{
    /*Reading 1D integer arrays */
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        cout << "Unable to read file" << endl;
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();

    const int rows = arr.size();
    const int cols = 1;

    Eigen::MatrixI out_arr = Eigen::MatrixI::Zero(rows,cols);
    //Eigen::MatrixBase<Derived> out_arr = Eigen::MatrixBase<Derived>::Zero(rows,cols);

    for(int i=0 ; i<rows;i++)
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
        cout << "Unable to read file" << endl;
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
    bool duplicate_rows = Standard_False;
    for (Integer j=0; j<cols; ++j)
    {
        if ( out_arr(out_arr.rows()-2,j)==out_arr(out_arr.rows()-1,j) )
        {
            duplicate_rows = Standard_True;
        }
        else
        {
            duplicate_rows = Standard_False;
        }
    }
    if (duplicate_rows==Standard_True)
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
        cout << "Unable to read file" << endl;
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
        for(int j=0 ; j<cols;j++)
        {
            out_arr(i,j) = std::atof(elems[j].c_str());
        }
    }

    // CHECK IF LAST LINE IS READ CORRECTLY
    bool duplicate_rows = Standard_False;
    for (Integer j=0; j<cols; ++j)
    {
        if ( out_arr(out_arr.rows()-2,j)==out_arr(out_arr.rows()-1,j) )
        {
            duplicate_rows = Standard_True;
        }
        else
        {
            duplicate_rows = Standard_False;
        }
    }
    if (duplicate_rows==Standard_True)
    {
        out_arr = out_arr.block(0,0,out_arr.rows()-1,out_arr.cols()).eval();
    }

    return out_arr;
}

void PostMeshBase::CheckMesh()
{
    /* Checks if the correct mesh data are imported */

    // check for multiple line copies of elements, points, edges and faces
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
//            int d1=0; int d2 =  this->mesh_points.rows()-1;
        Eigen::MatrixI a_rows = cnp::arange(0,this->mesh_points.rows()-1);
//            Eigen::MatrixI a_rows = cnp::arange(d1,d2);
        Eigen::MatrixI a_cols = cnp::arange(0,this->mesh_points.cols());
        this->mesh_points = cnp::take(this->mesh_points,a_rows,a_cols);
    }

    // elements
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
        Eigen::MatrixI a_rows = cnp::arange(0,this->mesh_elements.rows()-1);
        Eigen::MatrixI a_cols = cnp::arange(0,this->mesh_elements.cols());
        this->mesh_elements = cnp::take(this->mesh_elements,a_rows,a_cols);
    }

    // edges
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
        Eigen::MatrixI a_rows = cnp::arange(0,this->mesh_edges.rows()-1);
        Eigen::MatrixI a_cols = cnp::arange(0,this->mesh_edges.cols());
        this->mesh_edges = cnp::take(this->mesh_edges,a_rows,a_cols);
    }

    // faces for 3D
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
            Eigen::MatrixI a_rows = cnp::arange(0,this->mesh_faces.rows()-1);
            Eigen::MatrixI a_cols = cnp::arange(0,this->mesh_faces.cols());
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
    this->geometry_points.clear();
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
    /*  Iterate over TopoDS_Shape and extract all the edges, convert the edges into Geom_Curve and
        get their handles
    */

    this->geometry_curves.clear();
    this->geometry_curves_types.clear();
//    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_WIRE); explorer.More(); explorer.Next())
    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_EDGE); explorer.More(); explorer.Next())
    {
        // GET THE EDGES
        TopoDS_Edge current_edge = TopoDS::Edge(explorer.Current());
        // CONVERT THEM TO GEOM_CURVE
        Standard_Real first, last;
        Handle_Geom_Curve curve = BRep_Tool::Curve(current_edge,first,last);
        // STORE HANDLE IN THE CONTAINER
        this->geometry_curves.push_back(curve);

        // TO GET TYPE OF THE CURVE, CONVERT THE CURVE TO ADAPTIVE CURVE
        GeomAdaptor_Curve adapt_curve = GeomAdaptor_Curve(curve);
        // STORE TYPE OF CURVE (CURVE TYPES ARE DEFINED IN occ_inc.hpp)
        this->geometry_curves_types.push_back(adapt_curve.GetType());
    }
//    print(this->geometry_curves.size());
//    exit(EXIT_FAILURE);
}

void PostMeshBase::GetGeomFaces()
{
    /*  Iterate over TopoDS_Shape and extract all the faces, convert the faces into Geom_Surface and
        get their handles
    */

    this->geometry_surfaces.clear();
    this->geometry_surfaces_types.clear();

    for (TopExp_Explorer explorer(this->imported_shape,TopAbs_FACE); explorer.More(); explorer.Next())
    {

        // GET THE EDGES
        TopoDS_Face current_face = TopoDS::Face(explorer.Current());
        // CONVERT THEM TO GEOM_CURVE
        Handle_Geom_Surface surface = BRep_Tool::Surface(current_face);
        // STORE HANDLE IN THE CONTAINER
        this->geometry_surfaces.push_back(surface);

        // TO GET TYPE OF THE SURFACE, CONVERT THE SURFACE TO ADAPTIVE SURFACE
        GeomAdaptor_Surface adapt_surface = GeomAdaptor_Surface(surface);
        // STORE TYPE OF SURFACE (SURFACE TYPES ARE DEFINED IN occ_inc.hpp)
        this->geometry_surfaces_types.push_back(adapt_surface.GetType());

    }
}

void PostMeshBase::ComputeProjectionCriteria()
{
    // IF NOT ALLOCATED THEN COMPUTE
    if (this->projection_criteria.rows()==0)
    {
        this->projection_criteria.setZero(mesh_edges.rows(),mesh_edges.cols());
        for (auto iedge=0; iedge<mesh_edges.rows(); iedge++)
        {
            // GET THE COORDINATES OF THE TWO END NODES
            Real x1 = this->mesh_points(this->mesh_edges(iedge,0),0);
            Real y1 = this->mesh_points(this->mesh_edges(iedge,0),1);
            Real x2 = this->mesh_points(this->mesh_edges(iedge,1),0);
            Real y2 = this->mesh_points(this->mesh_edges(iedge,1),1);

            // GET THE MIDDLE POINT OF THE EDGE
            Real x_avg = ( x1 + x2 )/2.;
            Real y_avg = ( y1 + y2 )/2.;
            if (sqrt(x_avg*x_avg+y_avg*y_avg)<this->condition)
            {
                projection_criteria(iedge)=1;
            }
        }
        //this->projection_criteria.setOnes(mesh_edges.rows(),mesh_edges.cols());
    }
}
