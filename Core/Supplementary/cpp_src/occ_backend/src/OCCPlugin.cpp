
#include <OCCPlugin.hpp>

// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;
using namespace std;


// OCCPlugin class definitions
void OCCPlugin::Init(std::string &element_type, const UInteger &ndim)
{
    this->mesh_element_type = element_type;
    this->ndim = ndim;
}

void OCCPlugin::SetScale(Real &scale)
{
    this->scale = scale;
}

void OCCPlugin::SetCondition(Real &condition)
{
    this->condition = condition;
}

void OCCPlugin::SetProjectionCriteria(UInteger *criteria, const Integer &rows, const Integer &cols)
{
    this->projection_criteria = Eigen::Map<Eigen::MatrixUI>(criteria,rows,cols);
}

void OCCPlugin::SetDimension(const UInteger &dim)
{
    this->ndim=dim;
}

void OCCPlugin::SetMeshElementType(std::string &type)
{
    this->mesh_element_type = type;
}

void OCCPlugin::SetMeshElements(UInteger *arr, const Integer &rows, const Integer &cols)
{
    this->mesh_elements = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
}

void OCCPlugin::SetMeshPoints(Real *arr, const Integer &rows, const Integer &cols)
{
    this->mesh_points = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
//    new (&this->mesh_points) Eigen::Map<Eigen::MatrixR> (arr,rows,cols);
//    print (this->mesh_points.rows(),this->mesh_points.cols());
}

void OCCPlugin::SetMeshEdges(UInteger *arr, const Integer &rows, const Integer &cols)
{
    this->mesh_edges = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
}

void OCCPlugin::SetMeshFaces(UInteger *arr, const Integer &rows, const Integer &cols)
{
    this->mesh_faces = Eigen::Map<Eigen::MatrixUI>(arr,rows,cols);
}

void OCCPlugin::ScaleMesh()
{
    this->mesh_points *=this->scale;
}

std::string OCCPlugin::GetMeshElementType()
{
    return this->mesh_element_type;
}

void OCCPlugin::ReadMeshConnectivityFile(std::string &filename, char delim)
{
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

    std::vector<std::string> elems = split(arr[0], delim);
    const int rows = arr.size();
    const int cols = elems.size();
    Eigen::MatrixI arr_read = Eigen::MatrixI::Zero(rows,cols);

    for(int i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for (signed int j=0;j<cols; j++)
        {
            arr_read(i,j) = std::atof(elems[j].c_str());
        }
    }

    this->mesh_elements = arr_read;
}

void OCCPlugin::ReadMeshCoordinateFile(std::string &filename, char delim)
{
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

    std::vector<std::string> elems = split(arr[0], delim);
    const int rows = arr.size();
    const int cols = elems.size();
    Eigen::MatrixR arr_read = Eigen::MatrixR::Zero(rows,cols);

    for(int i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for (signed int j=0;j<cols; j++)
        {
            arr_read(i,j) = std::atof(elems[j].c_str());
        }
    }

    this->mesh_points = arr_read;
}

void OCCPlugin::ReadMeshEdgesFile(std::string &filename, char delim)
{
    std::vector<std::string> arr;
    arr.clear();
    std::string temp;

    std::ifstream datafile;
    datafile.open(filename.c_str());

    if(!datafile)
    {
        std::cout << "Unable to read file" << std::endl;
    }
    while(datafile)
    {
        datafile >> temp;
        temp += "";
        arr.push_back(temp);
    }

    datafile.close();

    std::vector<std::string> elems = split(arr[0], delim);
    const int rows = arr.size();
    const int cols = elems.size();
    Eigen::MatrixI arr_read = Eigen::MatrixI::Zero(rows,cols);

    for(int i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for (signed int j=0;j<cols; j++)
        {
            arr_read(i,j) = std::atof(elems[j].c_str());
        }
    }

    this->mesh_edges = arr_read;
}

void OCCPlugin::ReadMeshFacesFile(std::string &filename, char delim)
{
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

    std::vector<std::string> elems = split(arr[0], delim);
    const int rows = arr.size();
    const int cols = elems.size();
    Eigen::MatrixI arr_read = Eigen::MatrixI::Zero(rows,cols);

    for(int i=0 ; i<rows;i++)
    {
        std::vector<std::string> elems;
        elems = split(arr[i], delim);
        for (signed int j=0;j<cols; j++)
        {
            arr_read(i,j) = std::atof(elems[j].c_str());
        }
    }

    this->mesh_faces = arr_read;
}

void OCCPlugin::InferInterpolationPolynomialDegree()
{
    /* This is restricted to tris and tets*/
    if (this->ndim==2)
    {
        const Integer p = 1;
        for (int i=0; i<50;++i)
        {
            if ( (p+1)*(p+2)/2 == this->mesh_elements.cols())
            {
                this->degree = p;
                break;
            }
        }
    }
    else if (this->ndim==3)
    {
        const Integer p = 1;
        for (int i=0; i<50;++i)
        {
            if ( (p+1)*(p+2)*(p+3)/6 == this->mesh_elements.cols())
            {
                this->degree = p;
                break;
            }
        }
    }
}

Eigen::MatrixI OCCPlugin::Read(std::string &filename)
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

Eigen::MatrixUI OCCPlugin::ReadI(std::string &filename, char delim)
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

Eigen::MatrixR OCCPlugin::ReadR(std::string &filename, char delim)
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

void OCCPlugin::CheckMesh()
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

void OCCPlugin::SetFeketePoints(Real *arr, const Integer &rows, const Integer &cols)
{
    this->fekete = Eigen::Map<Eigen::MatrixR>(arr,rows,cols);
}

void OCCPlugin::ComputeProjectionCriteria()
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

void OCCPlugin::ReadIGES(const char* filename)
{
    /* IGES FILE READER BASED ON OCC BACKEND
     * THIS FUNCTION CAN BE EXPANDED FURTHER TO TAKE CURVE/SURFACE CONSISTENY INTO ACCOUNT
     * http://www.opencascade.org/doc/occt-6.7.0/overview/html/user_guides__iges.html
     */

    IGESControl_Reader reader;
    //IFSelect_ReturnStatus stat  = reader.ReadFile(filename.c_str());
    reader.ReadFile(filename);
    // CHECK FOR IMPORT STATUS
    reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
    reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
    // READ IGES FILE AS-IS
    Interface_Static::SetIVal("read.iges.bspline.continuity",0);
    Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
    if (ic !=0)
    {
        std::cout << "IGES file was not read as-is. The file was not read/transformed correctly";
    }

    //Interface_Static::SetIVal("xstep.cascade.unit",0);
    //Interface_Static::SetIVal("read.scale.unit",0);
    // IF ALL OKAY, THEN TRANSFER ROOTS
    reader.TransferRoots();

    this->imported_shape  = reader.OneShape();
    this->no_of_shapes = reader.NbShapes();

//    print(no_of_shapes);
//    exit(EXIT_FAILURE);

//        Handle_TColStd_HSequenceOfTransient edges = reader.GiveList("iges-faces");


}

void OCCPlugin::GetGeomVertices()
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

void OCCPlugin::GetGeomEdges()
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

void OCCPlugin::GetGeomFaces()
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

void OCCPlugin::GetCurvesParameters()
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

void OCCPlugin::GetCurvesLengths()
{
    //! Computes length of all curves
    this->curves_lengths = Eigen::MatrixR::Zero(this->geometry_curves.size(),1);
    for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        Handle_Geom_Curve current_curve = this->geometry_curves[icurve];
        curves_lengths(icurve) = cnp::length(current_curve,1);
    }
}

void OCCPlugin::FindCurvesSequentiallity()
{
    //! CHECKS IF THE PARAMETRIC REPRESENTATION OF CURVES ARE SEQUENTIAL, I.E.
    //! IF THE LastParameter OF ONE CURVE IS EQUAL TO THE FirstParameter OF THE SECOND CURVE.
    //! GENERALLY TRUE FOR BSPLINE CURVES
    Real Standard_Tolerance = 1.0e-14;
    std::vector<Integer> consecutive_curves; consecutive_curves.clear();
    Eigen::MatrixI curves_sequentiallity = -Eigen::MatrixI::Ones(this->geometry_curves.size(),3);
    this->GetCurvesParameters();
    println(this->curves_parameters);
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
//    print(consecutive_curves);
    println(curves_sequentiallity);
}

void OCCPlugin::ConcatenateSequentialCurves()
{

}

void OCCPlugin::GetGeomPointsOnCorrespondingEdges()
{
    this->geometry_points_on_curves.clear();
//    Real edge_counter=0;
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
//        print(this->geometry_points_on_curves[edge_counter]);
//        edge_counter++;
    }
//    exit(EXIT_FAILURE);
}

void OCCPlugin::GetInternalCurveScale()
{
    this->curve_to_parameter_scale_U = Eigen::MatrixR::Zero(this->geometry_curves.size(),1);
    for (unsigned int icurve=0; icurve<this->geometry_curves.size(); ++icurve)
    {
        Handle_Geom_Curve current_curve = this->geometry_curves[icurve];
        if (this->geometry_curves_types[icurve]!=0)
        {
            //this->curve_to_parameter_scale_U(icurve) = current_curve->LastParameter()/cnp::length(current_curve,1/this->scale);
            this->curve_to_parameter_scale_U(icurve) = abs(current_curve->LastParameter() - \
                                                        current_curve->FirstParameter())/cnp::length(current_curve,1/this->scale);
        }
        else
        {
            // IF GEOMETRY TYPE IS A LINE
            this->curve_to_parameter_scale_U(icurve) = 2*current_curve->LastParameter()/cnp::length(current_curve,1/this->scale);
        }
    }
}

void OCCPlugin::GetInternalSurfaceScales()
{

}

void OCCPlugin::IdentifyCurvesContainingEdges()
{
    this->dirichlet_edges = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim+1);
    //this->dirichlet_edges.block(0,0,mesh_edges.rows(),2) = this->mesh_edges.block(0,0,mesh_edges.rows(),2);
    this->listedges.clear();

//    this->InferInterpolationPolynomialDegree();

    Integer index_edge = 0;

    // LOOP OVER EDGES
    for (Integer iedge=0; iedge<this->mesh_edges.rows(); ++iedge)
    {
        // GET THE COORDINATES OF THE TWO END NODES
        Real x1 = this->mesh_points(this->mesh_edges(iedge,0),0);
        Real y1 = this->mesh_points(this->mesh_edges(iedge,0),1);
        Real x2 = this->mesh_points(this->mesh_edges(iedge,1),0);
        Real y2 = this->mesh_points(this->mesh_edges(iedge,1),1);

        // GET THE MIDDLE POINT OF THE EDGE
        Real x_avg = ( x1 + x2 )/2.;
        Real y_avg = ( y1 + y2 )/2.;
//        print(x1,y1,x2,y2);
//        cout << x_avg << " " << y_avg << endl;

//        if (sqrt(x_avg*x_avg+y_avg*y_avg)<this->condition)
        if (this->projection_criteria(iedge)==1)
        {
//            print (sqrt(x_avg*x_avg+y_avg*y_avg),this->condition);
//            println(x_avg,y_avg);
            this->listedges.push_back(iedge);
            for (UInteger iter=0;iter<ndim;++iter){
               dirichlet_edges(index_edge,iter) = this->mesh_edges(iedge,iter);
                }



            // PROJECT IT OVER ALL CURVES
            Real min_mid_distance = 1.0e20;
            Real mid_distance = 1.0e10;
            gp_Pnt middle_point(x_avg,y_avg,0.0);
//            gp_Pnt middle_point(1000.*x_avg,1000.*y_avg,0.0);

            // LOOP OVER CURVES
            for (UInteger icurve=0; icurve<this->geometry_curves.size(); ++icurve)
            {
//                gp_Pnt dd; this->geometry_curves[icurve]->D0(1.0002,dd);
//                print(dd.X(),dd.Y());
//                println(x_avg,y_avg);

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
//                print(mid_distance,min_mid_distance, icurve);
                if (mid_distance < min_mid_distance)
                {
//                    print(mid_distance,min_mid_distance,icurve);
                    // STORE ID OF CURVES
//                    this->dirichlet_edges(iedge,2) = icurve;
                    this->dirichlet_edges(index_edge,2) = icurve; // <--THIS
//                    this->dirichlet_edges(index_edge,2) = this->geometry_curves_types[icurve]; // CHECK THIS
//                    cout << dirichlet_edges(iedge,2) << endl;
                    // RE-ASSIGN
                    min_mid_distance = mid_distance;
                } //print(this->geometry_curves[icurve]->FirstParameter(),this->geometry_curves[icurve]->LastParameter());
            }
            index_edge +=1;
        }
    }

    Eigen::MatrixI arr_rows = cnp::arange(index_edge);
    Eigen::MatrixI arr_cols = cnp::arange(ndim+1);
    this->dirichlet_edges = cnp::take(this->dirichlet_edges,arr_rows,arr_cols);
//    print(this->dirichlet_edges);
//    print(this->geometry_curves.size());
//    print(this->geometry_curves_types);

//    gp_Trsf transformation;
//    gp_Pnt pnt1();
//    Handle_Geom_Curve cr0 = this->geometry_curves[0];
//    gp_Pnt fp = cr0->Value(cr0->FirstParameter());
//    gp_Pnt lp = cr0->Value(cr0->LastParameter());
//    print (fp.X(),fp.Y(),fp.Z());
//    print (lp.X(),lp.Y(),lp.Z());
//    print (cr0->FirstParameter(),cr0->LastParameter());

//    gp_Trsf transformation;
//    transformation.SetTranslation(fp,lp);
////    Real u = cr0->ParametricTransformation(transformation);
////    print (u);
//    Real u = cr0->TransformedParameter(6,transformation);
//    print (u);

//    this->GetCurvesParameters();
//    this->GetCurvesLengths();
//    this->FindCurvesSequentiallity();
//    exit(EXIT_FAILURE);
}

void OCCPlugin::ProjectMeshOnCurve(const char *projection_method)
{
//    Real precision_tolerance = 1.0e-2;
    Real precision_tolerance = 1.0e-4;
    this->projection_method = projection_method;
//    this->InferInterpolationPolynomialDegree();

    this->projection_ID = Eigen::MatrixI::Zero(this->dirichlet_edges.rows(),this->ndim);
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
            Real x = this->mesh_points(this->dirichlet_edges(iedge,inode),0);
            Real y = this->mesh_points(this->dirichlet_edges(iedge,inode),1);
            // GET THE CURVE THAT THIS EDGE HAS TO BE PROJECTED TO
            UInteger icurve = this->dirichlet_edges(iedge,2);
            // GET THE COORDINATES OF CURVE'S TWO VERTICES
            Real x1_curve = this->geometry_points_on_curves[icurve](0,0);
            Real y1_curve = this->geometry_points_on_curves[icurve](0,1);
            Real x2_curve = this->geometry_points_on_curves[icurve](1,0);
            Real y2_curve = this->geometry_points_on_curves[icurve](1,1);
            // CHECK IF WE ARE ON THE CURVE INTERSECTION
            if (abs(x1_curve-x) < precision_tolerance && abs(y1_curve-y) < precision_tolerance && this->geometry_curves_types[icurve]!=0)
            {
                // PROJECT THE CURVE VERTEX INSTEAD OF THE EDGE NODE (THIS IS NECESSARY TO ENSURE SUCCESSFUL PROJECTION)
                x = x1_curve;
                y = y1_curve;
            }
            else if (abs(x2_curve-x) < precision_tolerance && abs(y2_curve-y) < precision_tolerance && this->geometry_curves_types[icurve]!=0)
            {
                x = x2_curve;
                y = y2_curve;
            }

            try
            {
                // GET THE NODE THAT HAS TO BE PROJECTED TO THE CURVE
                gp_Pnt node_to_be_projected = gp_Pnt(x,y,0.0);
                // PROJECT THE NODES ON THE CURVE AND GET THE PARAMETER U
                GeomAPI_ProjectPointOnCurve proj;
                proj.Init(node_to_be_projected,this->geometry_curves[icurve]);
                parameterU = proj.LowerDistanceParameter();
            }
            catch (StdFail_NotDone)
            {
                std::cerr << "The edge node was not projected to the right curve. Curve number: " << " " << icurve << std::endl;
//                print(x,y,iedge);
//                print(x,y,x2_curve,y2_curve);
            }

            // GET CURVE LENGTH
            Real curve_length = cnp::length(this->geometry_curves[icurve],1.);

//            print(x_curve-x,y_curve-y);
//            print(curve_point_coords);
//            print(this->geometry_points_on_curves[icurve].block(0,0,2,2));
//            print(this->geometry_points_on_curves[icurve]);

//            println(x,y,x_curve,y_curve,icurve);
            // STORE ID OF NURBS
            this->projection_ID(iedge,inode) = icurve;
            // STORE PROJECTION POINT PARAMETER ON THE CURVE (NORMALISED)
            //this->projection_U(iedge,inode) = this->scale*parameterU/curve_length; //# THIS
            Real U0 = this->geometry_curves_types[icurve]==0 ? 0. : this->geometry_curves[icurve]->FirstParameter();
            this->projection_U(iedge,inode) =   this->scale*(parameterU-U0)/curve_length;
//            println(parameterU/curve_length);
//            println (parameterU-this->geometry_curves[icurve]->FirstParameter());
        }
    }

    // SORT PROJECTED PARAMETERS OF EACH EDGE - MUST INITIALISE SORT INDICES
    this->sorted_projected_indices = Eigen::MatrixI::Zero(this->projection_U.rows(),this->projection_U.cols());

//    println(this->projection_U);
    cnp::sort_rows(this->projection_U,this->sorted_projected_indices);
//    print(this->projection_U);
//    print(sorted_projected_indices);
//    print(this->dirichlet_edges);
//    print(this->geometry_curves_types);

//    Handle_Geom_Curve dd =  this->geometry_curves[1];
//    gp_Pnt gg;
//    dd->D0(dd->LastParameter()/2.,gg);
//    print(gg.X(),gg.Y());
//    print (mesh_points.block(0,0,8,2));

//    this->CurvesToBsplineCurves();
////    Handle_Geom_BSplineCurve current_bspline_curve = this->geometry_curves_bspline[0];
//    Handle_Geom_BSplineCurve curve_2 = this->geometry_curves_bspline[2];
////    println(current_bspline_curve->FirstParameter(),current_bspline_curve->LastParameter());
//    println(curve_2->FirstParameter(),curve_2->LastParameter());

//    GeomConvert_CompCurveToBSplineCurve xx = GeomConvert_CompCurveToBSplineCurve(this->geometry_curves_bspline[0]);
////    GeomConvert_CompCurveToBSplineCurve xx = GeomConvert_CompCurveToBSplineCurve(this->geometry_curves_bspline[3],Convert_RationalC1);
////    this->geometry_curves_bspline[0]->Reverse();
////    Handle_Geom_BSplineCurve cc = geometry_curves_bspline[1];
////    cc->Reverse();
////    xx.Add(cc,1e-8);
//    xx.Add(this->geometry_curves_bspline[1],1e-8);
//    xx.Add(this->geometry_curves_bspline[2],1e-8);
//    xx.Add(this->geometry_curves_bspline[3],1e-8);
//    Handle_Geom_BSplineCurve xx2 = xx.BSplineCurve();
//    print(xx2->FirstParameter(),xx2->LastParameter());
//    gp_Pnt p1,p2;
//    p1 = xx2->Value(xx2->FirstParameter());
//    p2 = xx2->Value(xx2->LastParameter());
//    println(p1.X(),p1.Y(),p2.X(),p2.Y());
//    println(cnp::length(xx2,1));
//    println (xx2->IsClosed());
//    println(xx2->IsPeriodic());

//    this->GetCurvesParameters();
//    this->GetCurvesLengths();
//    println(curves_parameters);
//    println(curves_lengths);
//    this->FindCurvesSequentiallity();

//    xx.Add(this->geometry_curves_bspline[0],this->geometry_curves_bspline[1],Standard_False,Standard_True,0);
//    xx.Add(,)
//    println()
//    Hermit::Solution()

//    for (int i; i<(int)geometry_points.size(); ++i)
//        print(geometry_points[i].X(),geometry_points[i].Y(),geometry_points[i].Z());

//    exit(EXIT_FAILURE);
}

void OCCPlugin::ProjectMeshOnSurface()
{
    /* Projects all the points on the mesh to the boundary of Geom_Curve */
    this->projection_ID = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim);
    this->projection_U = Eigen::MatrixR::Zero(this->mesh_edges.rows(),this->ndim);
    this->projection_V = Eigen::MatrixR::Zero(this->mesh_edges.rows(),this->ndim);
    this->dirichlet_faces = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->ndim+1);
    this->listfaces.clear();
    int index_face = 0;

    for (Integer iface=0; iface<this->mesh_edges.rows(); ++iface)
    {
        for (UInteger jface=0; jface<this->ndim; ++jface) // linear
        {
            Standard_Real x = this->mesh_points(this->mesh_edges(iface,jface),0);
            Standard_Real y = this->mesh_points(this->mesh_edges(iface,jface),1);
            Standard_Real z = this->mesh_points(this->mesh_edges(iface,jface),2);
            if (sqrt(x*x+y*y+z*z)<this->condition)
            {
                if (jface==0)
                {
                    this->listfaces.push_back(iface);
                    for (UInteger iter=0;iter<ndim;++iter)
                        dirichlet_faces(index_face,iter) = this->mesh_edges(iface,iter);
                    index_face +=1;
                }

                Standard_Real min_distance = 1.0e10;
                Standard_Real distance = 0.;
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
                        proj.LowerDistanceParameters(this->projection_U(iface,jface),this->projection_V(iface,jface)); //DIVIDE BY DIRECTIONAL LENGTH (AREA)?
                        min_distance = distance;
                    }
                }
            }
        }
    }
}

void OCCPlugin::RepairDualProjectedParameters()
{
    Real lengthTol = 1.0e-10;

    //#pragma omp parallel for
    for (Integer iedge=0;iedge<this->dirichlet_edges.rows();++iedge)
    {
        Integer id_curve = this->dirichlet_edges(iedge,2);
        Handle_Geom_Curve current_curve = this->geometry_curves[id_curve];
        // DUAL PROJECTION HAPPENS FOR CLOSED CURVES
        if (current_curve->IsClosed() || current_curve->IsPeriodic())
        {
            // GET THE CURRENT EDGE
            Eigen::Matrix<Real,1,2> current_edge_U = this->projection_U.row(iedge);
            // SORT IT
            std::sort(current_edge_U.data(),current_edge_U.data()+current_edge_U.cols());
            // GET THE FIRST AND LAST PARAMETERS OF THIS CURVE
            Real u1 = current_curve->FirstParameter()/cnp::length(current_curve,1.0/this->scale);
            Real u2 = current_curve->LastParameter()/cnp::length(current_curve,1.0/this->scale);

    //        Real u1 = this->geometry_curves[id_curve]->FirstParameter()/cnp::length(this->geometry_curves[id_curve],1.);
    //        Real u2 = this->geometry_curves[id_curve]->LastParameter()/cnp::length(this->geometry_curves[id_curve],1.);
    //        cout << u1 << " "<< u2<< endl;
    //        cout << current_edge_U << endl;
    //        println(current_edge_U,u1,u2);
            // IF THE U PARAMETER FOR THE FIRST NODE OF THIS EDGE EQUALS TO THE FIRST PARAMETER
            if ( abs(current_edge_U(0) - u1) < lengthTol )
            {
                 // WE ARE AT THE STARTING PARAMETER - GET THE LENGTH TO uc
                 Real uc = current_edge_U(1);
    //             print(u1,uc,u2);
                 GeomAdaptor_Curve current_adapt_curve(current_curve);
                 Real l0c = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,uc)/this->scale;
                 Real l01 = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,u2)/this->scale;

    //             Real l0c = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,uc);
    //             Real l01 = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,u2);
    //             print(l0c,l01);
    //             Real ld = GCPnts_AbscissaPoint::Length(current_adapt_curve,0,0.5)/this->scale;
    //             print(ld,l0c,l01);
                 //if ( (l0c-u1) > (l01-l0c))
                 if ( l0c > (l01-l0c))
                 {
                     // FIX THE PARAMETER TO LastParameter
                     this->projection_U(iedge,0) = u2;
                 }
                 //cout << l0c << " " <<  l01 << endl<<endl;
            }
            else if ( abs(current_edge_U(1) - u2) < lengthTol )
            {
                 // WE ARE AT THE END PARAMETER - GET THE LENGTH TO uc
                 Real uc = current_edge_U(0);
                 GeomAdaptor_Curve current_adapt_curve(current_curve);
                 Real lc1 = GCPnts_AbscissaPoint::Length(current_adapt_curve,uc,u2)/this->scale;
                 Real l0c = GCPnts_AbscissaPoint::Length(current_adapt_curve,u1,uc)/this->scale;

                 //if ( (l0c-u1) > (l01-l0c))
                 if ( l0c < lc1 )
                 {
                     // FIX THE PARAMETER TO FirstParameter
                     this->projection_U(iedge,1) = u1;
                 }
                 //cout << l0c << " " <<  lc1 << endl<<endl;
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

void OCCPlugin::CurvesToBsplineCurves()
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

void OCCPlugin::SurfacesToBsplineSurfaces()
{
    /* Converts all the imported surfaces to bspline surfaces : http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
     */
    this->geometry_surfaces_bspline.clear();
    for (unsigned int isurf=0; isurf < this->geometry_surfaces.size(); ++isurf)
    {
        this->geometry_surfaces_bspline.push_back( GeomConvert::SurfaceToBSplineSurface(this->geometry_surfaces[isurf]) );
    }
}

void OCCPlugin::MeshPointInversionCurve()
{
    this->no_dir_edges = this->dirichlet_edges.rows();
    Integer no_edge_nodes = this->mesh_edges.cols();
    Eigen::MatrixI arr_row = Eigen::Map<Eigen::Matrix<Integer,Eigen::Dynamic,1> >(this->listedges.data(),this->listedges.size());
//    Eigen::MatrixI arr_row = cnp::arange(this->dirichlet_edges.rows());
    Eigen::MatrixI arr_col = cnp::arange(0,no_edge_nodes);
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


    for (Integer idir=0; idir< this->no_dir_edges; ++idir)
    {
        Integer id_curve = this->dirichlet_edges(idir,2);
        Handle_Geom_Curve current_curve = this->geometry_curves[id_curve];
        Real length_current_curve = cnp::length(current_curve,1/this->scale);
        Real internal_scale = 1./this->curve_to_parameter_scale_U(id_curve);
//        cout << internal_scale << endl;
//    print(this->geometry_curves_types[id_curve]);
//        print(cnp::length(current_curve),current_curve->FirstParameter(),current_curve->LastParameter());

        GeomAdaptor_Curve current_curve_adapt(current_curve);
//        println(current_curve_adapt.FirstParameter(),current_curve->FirstParameter());

        for (Integer j=0; j<no_edge_nodes;++j)
        {
            Real tol =1e-10;
            Real U0 = this->geometry_curves_types[id_curve]==0 ? 0. : current_curve_adapt.FirstParameter();
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


//            current_curve_adapt.D0(uEq*length_current_curve,xEq);
//            current_curve_adapt.D0(uEq,xEq);
//            current_curve_adapt.D0(uEq*length_current_curve+U0,xEq);
//            println(uEq,uEq*length_current_curve,U0);
//            println(length_current_curve);

            Eigen::MatrixR gp_pnt_old = (this->mesh_points.row(this->nodes_dir(this->index_nodes( j ))).array()/this->scale);

            this->displacements_BC(this->index_nodes(j),0) = (xEq.X()/this->scale - gp_pnt_old(0));
            this->displacements_BC(this->index_nodes(j),1) = (xEq.Y()/this->scale - gp_pnt_old(1));
            //this->displacements_BC(this->index_nodes(j),2) = (xEq.Z()/this->scale - gp_pnt_old(2)); // USEFULL FOR 3D CURVES

//            println( internal_scale*this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*this->scale );
//            println(internal_scale*curves_lengths(id_curve)*this->u_of_all_fekete_mesh_edges(idir,this->boundary_edges_order(this->listedges[idir],j))*this->scale);
//            println( internal_scale*this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*this->scale,
//                     internal_scale*curves_lengths(id_curve)*this->u_of_all_fekete_mesh_edges(idir,this->boundary_edges_order(this->listedges[idir],j))*this->scale);
//            println(curves_lengths.rows(),curves_lengths.cols());
//            println(this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*scale );
//            println( this->boundary_edges_order(this->listedges[idir],j) );
//           print(this->u_of_all_fekete_mesh_edges(idir,this->boundary_edges_order(this->listedges[idir],j))*this->scale);
//            println(uEq);
//            println(U0);
//            println( internal_scale*this->u_of_all_fekete_mesh_edges(idir, this->boundary_edges_order(this->listedges[idir],j))*this->scale,uEq,id_curve);
//            print(current_curve_adapt.GetType());
//            print(length_current_curve);
//            cout << gp_pnt_old(0) << " " << gp_pnt_old(1) << endl;
//            cout << xEq.X()/this->scale << " " << xEq.Y()/this->scale << endl;
//            cout << gp_pnt_old(0) << " " << gp_pnt_old(1) << "] [" << xEq.X()/this->scale << " " << xEq.Y()/this->scale << endl;

//            Real xx = internal_scale*curves_lengths(id_curve)*this->u_of_all_fekete_mesh_edges(idir,
//                                                  this->boundary_edges_order(this->listedges[idir],j))*this->scale;
//            GCPnts_AbscissaPoint inv1 = GCPnts_AbscissaPoint(tol,current_curve_adapt,xx,current_curve->FirstParameter());
//            GCPnts_AbscissaPoint inv1 = GCPnts_AbscissaPoint(tol,current_curve_adapt,11.7758/2.,current_curve->FirstParameter());
//            Real uEq1 = inv1.Parameter();
//            println(uEq1);
//            println(xx,uEq1);
//            print(cnp::length(current_curve,1.0));
//            println(curves_parameters)
//            gp_Pnt x2 = current_curve->Value(0.);
//            gp_Pnt x1 = current_curve->Value(current_curve->FirstParameter());
//            println(x1.X(),x1.Y(),x2.X(),x2.Y(),current_curve->FirstParameter(),id_curve);



        }
        this->index_nodes = ((this->index_nodes).array()+no_edge_nodes).eval().matrix();
//        cout << " " << endl;
    }
//    cout << displacements_BC << endl;
//    print (this->ndim);

}

void OCCPlugin::MeshPointInversionSurface()
{

}

Eigen::MatrixR OCCPlugin::ParametricFeketePoints(Standard_Real &u1,Standard_Real &u2)
{
    Eigen::MatrixR fekete_on_curve;
    fekete_on_curve = (u1 + (u2-u1)/2.0*((this->fekete).array()+1.)).matrix();
    return fekete_on_curve;
}

void OCCPlugin::GetElementsWithBoundaryEdgesTri()
{
    this->InferInterpolationPolynomialDegree();
    assert (this->degree!=2 && "YOU_TRIED_CALLING_A_TRIANGULAR_METHOD_ON_TETRAHEDRA");
    this->elements_with_boundary_edges = Eigen::MatrixI::Zero(this->mesh_edges.rows(),1);

    for (Integer iedge=0; iedge<this->mesh_edges.rows();++iedge)
    {
        std::vector<Integer> all_rows; all_rows.clear();
        for (Integer jedge=0; jedge<this->mesh_edges.cols();++jedge)
        {
            Eigen::MatrixI rows; //Eigen::MatrixI cols; not needed
            auto indices = cnp::where_eq(this->mesh_elements,this->mesh_edges(iedge,jedge));
            std::tie(rows,std::ignore) = indices;

            for (Integer k=0; k<rows.rows(); ++k)
            {
               all_rows.push_back(rows(k));
            }
        }

        Eigen::MatrixI all_rows_eigen = Eigen::Map<Eigen::MatrixI>(all_rows.data(),all_rows.size(),1);
        for (Integer i=0; i<all_rows_eigen.rows(); ++i)
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
//        print (elements_with_boundary_edges);
//        print (this->degree);
}

void OCCPlugin::GetBoundaryPointsOrder()
{
    /*The order of boundary (edge/face) connectivity. Needs the degree to be inferred a priori */
    if (this->ndim ==2)
    {
//        OLD
//        this->boundary_points_order = Eigen::MatrixI::Zero(this->fekete.rows(),this->fekete.cols());
//        this->boundary_points_order(0) = this->fekete.rows()-1;
//        this->boundary_points_order.block(2,0,fekete.rows()-2,1) = cnp::arange(1,fekete.rows()-1).reverse();

        this->boundary_points_order = Eigen::MatrixI::Zero(this->fekete.rows(),this->fekete.cols());
        this->boundary_points_order(1) = this->fekete.rows()-1;
        this->boundary_points_order.block(2,0,fekete.rows()-2,1) = cnp::arange(1,fekete.rows()-1);

        // FOR ALL THE EDGES
        this->boundary_edges_order = Eigen::MatrixI::Zero(this->mesh_edges.rows(),this->mesh_edges.cols());
        boundary_edges_order.col(1) = (boundary_edges_order.cols()-1)*Eigen::MatrixI::Ones(boundary_edges_order.rows(),1);

        for (Integer iedge=0;iedge<this->mesh_edges.rows();++iedge)
        {
            Eigen::MatrixI current_edge = this->mesh_edges.row(iedge).transpose();
            Eigen::MatrixI all_points_cols = cnp::arange(this->mesh_points.cols());
            Eigen::MatrixR current_edge_coordinates = cnp::take(this->mesh_points,current_edge,all_points_cols);
            current_edge_coordinates = (current_edge_coordinates.array()/1000.).matrix().eval();

            std::vector<Real> norm_rows(this->mesh_edges.cols()-2);
//            for (Integer j=1; j<current_edge_coordinates.rows()-1; ++j)
            for (Integer j=2; j<current_edge_coordinates.rows(); ++j)
            {
//                norm_rows[j-1] = (current_edge_coordinates.row(j).array() - current_edge_coordinates.row(current_edge_coordinates.rows()-1).array()).matrix().norm();
                norm_rows[j-2] = (current_edge_coordinates.row(j).array() - \
                                  current_edge_coordinates.row(1).array()).matrix().norm();
            }
            std::vector<Integer> sorted_idx = cnp::argsort(norm_rows);
            Eigen::MatrixI idx = Eigen::Map<Eigen::MatrixI>(sorted_idx.data(),1,sorted_idx.size());
            boundary_edges_order.block(iedge,2,1,sorted_idx.size()) = (idx.array()+1).matrix().reverse();
//            boundary_edges_order.block(iedge,2,1,sorted_idx.size()) = (idx.array()+1).matrix();


        }
//        print (boundary_edges_order.block(18,0,5,5));
    }
    // CHECK THIS
    else if (this->ndim==3)
    {
        assert (this->degree > 1 || this->degree <100);
        Integer nsize = (this->degree+1)*(this->degree+2)/2;
        this->boundary_points_order = Eigen::MatrixI::Zero(nsize,1);
        if (this->degree==1)
        {
              this->boundary_points_order << 0,1,2;
        }
        else if (this->degree==2)
        {
            this->boundary_points_order << 0,3,1,5,2,4;
        }
        else if (this->degree==3)
        {
            this->boundary_points_order << 0,3,4,1,7,9,2,8,5;
        }
        else if (this->degree==4)
        {
            this->boundary_points_order << 0,3,4,5,1,9,12,14,2,13,10,6;
        }
    }
}

void OCCPlugin::EstimatedParameterUOnMesh()
{
//    this->no_dir_edges = this->listedges.size();
    this->no_dir_edges = this->dirichlet_edges.rows();
    this->u_of_all_fekete_mesh_edges = Eigen::MatrixR::Zero(this->no_dir_edges,this->fekete.rows());

//    print(this->projection_U);
    for (Integer idir=0; idir< this->no_dir_edges; ++idir)
    {
        Integer id_curve = this->dirichlet_edges(idir,2);
//        Real u1 = this->projection_U(this->listedges[idir],0);
//        Real u2 = this->projection_U(this->listedges[idir],1);
        Real u1 = this->projection_U(idir,0);
        Real u2 = this->projection_U(idir,1);
        Handle_Geom_Curve current_curve = this->geometry_curves[id_curve];
        u_of_all_fekete_mesh_edges.block(idir,0,1,this->fekete.rows()) = ParametricFeketePoints(u1,u2).transpose();
//        print(u_of_all_fekete_mesh_edges);
        if (current_curve->IsClosed())
        {

            // FIND THE SCALED LAST PARAMETER
            Real scaled_endU = this->curve_to_parameter_scale_U(id_curve);
            GeomAdaptor_Curve current_curve_adapt(current_curve);
            Real umin = u_of_all_fekete_mesh_edges(idir,0); //abs
            Real umax = u_of_all_fekete_mesh_edges(idir,this->fekete.rows()-1); //abs
    //        cout << umin << " " << umax << endl;
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
    //        cout << current_curve_length << endl;
//            cout << length_right << " "<< length_left << "  "<< edge_length << endl;
    //        cout << umin << " " << umax << endl;
    //        cout << current_curve->FirstParameter() << " " << current_curve->LastParameter() << " "  << umin << " " << umax << " " << id_curve << endl;
    //        cout << umin << " "<< umax << " "<< current_curve->LastParameter() << endl;
    //        print (length_left+length_right , edge_length);
            if (length_left+length_right < edge_length)
            {
    //            assert ( (length_left+length_right) - edge_length > 1e-4 && "TOO_COARSE_MESH");
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
    //            cout << trespassed_length << endl;
    //            cout << right_one.transpose() << " " << left_one.transpose() << endl;
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
    //            print (dum_1_eigen);
                this->u_of_all_fekete_mesh_edges.block(idir,0,1,this->fekete.rows()) = dum_1_eigen.transpose();
            }
        }
    }
//    print (boundary_edges_order);
//    cout << this->u_of_all_fekete_mesh_edges << endl;
//    exit (EXIT_FAILURE);
}

PassToPython OCCPlugin::GetDirichletData()
{
    PassToPython struct_to_python;
    struct_to_python.nodes_dir_size = this->nodes_dir.rows();
    // CONVERT FROM EIGEN TO STL VECTOR
    struct_to_python.nodes_dir_out_stl.assign(this->nodes_dir.data(),this->nodes_dir.data()+struct_to_python.nodes_dir_size);
    struct_to_python.displacement_BC_stl.assign(this->displacements_BC.data(),this->displacements_BC.data()+ \
                                                this->ndim*struct_to_python.nodes_dir_size);

    return struct_to_python;
}
