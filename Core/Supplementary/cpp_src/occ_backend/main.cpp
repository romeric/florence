

//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <occ_frontend.hpp>


using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;



// SUBCLASS GEOM_CURVE CLASS
class PyGeom_Curve: Geom_Curve
{
public:
    // CONSTRUCTOR
    PyGeom_Curve(){}
    ~PyGeom_Curve(){}

    //CONVERT FROM Geom_Curve
    inline Standard_Real length(Standard_Real scale=0.001)
    {
        // GET LENGTH OF THE CURVE
        GeomAdaptor_Curve current_curve(this);
        Standard_Real curve_length = GCPnts_AbscissaPoint::Length(current_curve);
        // CHANGE THE SCALE TO 1. IF NEEDED
        return scale*curve_length;
    }

};
// end of PyGeom_Curve




// OCC_Backend CLASS
class OCC_FrontEnd
{
private:
    Eigen::MatrixXi projection_ID;
    Eigen::MatrixXdr projection_U;
    Eigen::MatrixXdr projection_V;
    Eigen::MatrixXi dirichlet_edges;
    Eigen::MatrixXi dirichlet_faces;
    std::vector<int> listedges;
    std::vector<int> listfaces;
    Standard_Integer no_dir_edges;
    Standard_Integer no_dir_faces;
    Eigen::MatrixXi unique_edges;
    Eigen::MatrixXi unique_faces;
    Eigen::MatrixXd fekete_1d;
    Eigen::MatrixXi boundary_points_order;
public:
    // CONSTRUCTOR
    OCC_FrontEnd() {
    }
    OCC_FrontEnd(std::string &element_type,int &ndim){
        this->mesh_element_type = element_type;
        this->ndim = ndim;
        this->condition = 0.;
        this->scale = 1000.;
    }

    // members of occ_ backend
    std::string mesh_element_type;
    int ndim;
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_elements;
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_points;
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_edges;
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> mesh_faces;
    TopoDS_Shape imported_shape;
    Standard_Integer no_of_shapes;
    std::vector<Handle_Geom_Curve> geometry_edges;
    std::vector<Handle_Geom_Surface> geometry_faces;
    std::vector<Handle_Geom_BSplineCurve> geometry_edges_bspline;
    std::vector<Handle_Geom_BSplineSurface> geometry_faces_bspline;
    Standard_Real condition;
    Standard_Real scale;
    Eigen::MatrixXd displacements_BC;
    Eigen::MatrixXi index_nodes;
    std::string projection_methoed;


    // methods of occ_backend
    void Init(std::string &element_type,int &ndim)
    {
        this->mesh_element_type = element_type;
        this->ndim = ndim;
    }

    void SetCondition(Standard_Real &condition)
    {
        this->condition = condition;
    }

    void SetDimension(int &ndim)
    {
        this->ndim=ndim;
    }

    void SetElementType(std::string &type)
    {
        this->mesh_element_type = type;
    }

    void SetElements(Eigen::MatrixXi &arr)
    {
        this->mesh_elements=arr;
    }

    void SetPoints(Eigen::MatrixXd &arr)
    {
        this->mesh_points=arr;
    }

    void SetEdges(Eigen::MatrixXi &arr)
    {
        this->mesh_edges=arr;
    }

    void SetFaces(Eigen::MatrixXi &arr)
    {
        this->mesh_faces=arr;
    }

    std::string GetElementType()
    {
        return this->mesh_element_type;
    }

    void ReadMeshConnectivityFile(string &filename, char delim)
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
        Eigen::MatrixXi arr_read = Eigen::MatrixXi::Zero(rows,cols);

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

    void ReadMeshCoordinateFile(std::string &filename, char delim)
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
        Eigen::MatrixXd arr_read = Eigen::MatrixXd::Zero(rows,cols);

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

    void ReadMeshEdgesFile(std::string &filename, char delim)
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
        Eigen::MatrixXi arr_read = Eigen::MatrixXi::Zero(rows,cols);

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

    void ReadMeshFacesFile(std::string &filename, char delim)
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
        Eigen::MatrixXi arr_read = Eigen::MatrixXi::Zero(rows,cols);

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

    void ReadUniqueEdges(std::string &filename)
    {
        this->unique_edges = this->Read(filename);
    }

    void ReadUniqueFaces(std::string &filename)
    {
        this->unique_faces = this->Read(filename);
    }

    Eigen::MatrixXi Read(std::string &filename)
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

        const int rows = arr.size();
        const int cols = 1;

        Eigen::MatrixXi out_arr = Eigen::MatrixXi::Zero(rows,cols);
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

    void CheckMesh()
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
            Eigen::MatrixXi a_rows = cnp::arange(0,this->mesh_points.rows()-1);
//            Eigen::MatrixXi a_rows = cnp::arange(d1,d2);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->mesh_points.cols());
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
            Eigen::MatrixXi a_rows = cnp::arange(0,this->mesh_elements.rows()-1);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->mesh_elements.cols());
            this->mesh_elements = cnp::take(this->mesh_elements,a_rows,a_cols);
        }

        // edges
        int flag_ed = 0;
        for (int i=this->mesh_edges.rows()-2; i<this->mesh_edges.rows();i++)
        {
            check_duplicated_rows = (this->mesh_edges.row(i) - this->mesh_edges.row(i-1)).norm();
            if (std::abs(check_duplicated_rows)  < 1.0e-14)
            {
                flag_ed = 1;
            }
        }
        if (flag_ed == 1)
        {
            Eigen::MatrixXi a_rows = cnp::arange(0,this->mesh_edges.rows()-1);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->mesh_edges.cols());
            this->mesh_edges = cnp::take(this->mesh_edges,a_rows,a_cols);
        }

        // faces for 3D
        if (this->mesh_faces.cols()!=0)
        {
            int flag_f = 0;
            for (int i=this->mesh_faces.rows()-2; i<this->mesh_faces.rows();i++)
            {
                check_duplicated_rows = (this->mesh_faces.row(i) - this->mesh_faces.row(i-1)).norm();
                if (std::abs(check_duplicated_rows)  < 1.0e-14)
                {
                    flag_f = 1;
                }
            }
            if (flag_f == 1)
            {
                Eigen::MatrixXi a_rows = cnp::arange(0,this->mesh_faces.rows()-1);
                Eigen::MatrixXi a_cols = cnp::arange(0,this->mesh_faces.cols());
                this->mesh_faces = cnp::take(this->mesh_faces,a_rows,a_cols);
            }
        }

        int maxelem = this->mesh_elements.maxCoeff();
        int maxpoint = this->mesh_points.rows();

        if (maxelem+1 != maxpoint)
        {
            throw std::invalid_argument("Element connectivity and nodal coordinates do not match. This can be "
                                        "caused by giving two files which do not correspond to each other");
        }

        std::cout << "All good with imported mesh. proceeding..." << std::endl;

    }

    void ReadIGES(std::string & filename)
    {
        /* IGES FILE READER BASED ON OCC BACKEND
         * THIS FUNCTION CAN BE EXPANDED FURTHER TO TAKE CURVE/SURFACE CONSISTENY INTO ACCOUNT
         * http://www.opencascade.org/doc/occt-6.7.0/overview/html/user_guides__iges.html
         */

        IGESControl_Reader reader;
        IFSelect_ReturnStatus stat  = reader.ReadFile(filename.c_str());
        // CHECK FOR IMPORT STATUS
        reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
        reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
        // READ IGES FILE AS-IS
        Interface_Static::SetIVal("read.iges.bspline.continuity",0);
        Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
        if (ic !=0)
        {
            cout << "IGES file was not read as-is. The file was not read/transformed correctly";
        }

        //Interface_Static::SetIVal("xstep.cascade.unit",0);
        //Interface_Static::SetIVal("read.scale.unit",0);
        // IF ALL OKAY, THEN TRANSFER ROOTS
        reader.TransferRoots();

        this->imported_shape  = reader.OneShape();
        this->no_of_shapes = reader.NbShapes();

//        Handle_TColStd_HSequenceOfTransient edges = reader.GiveList("iges-faces");


    }

    void GetGeomEdges()
    {
        /*  Iterate over TopoDS_Shape and extract all the edges, convert the edges into Geom_Curve and
            get their handles
        */

        this->geometry_edges.clear();

        for (TopExp_Explorer explorer(this->imported_shape,TopAbs_EDGE); explorer.More(); explorer.Next())
        {
            // GET THE EDGES
            TopoDS_Edge current_edge = TopoDS::Edge(explorer.Current());
            // CONVERT THEM TO GEOM_CURVE
            Standard_Real first, last;
            Handle_Geom_Curve curve = BRep_Tool::Curve(current_edge,first,last);
            // STORE HANDLE IN THE CONTAINER
            this->geometry_edges.push_back(curve);
        }
    }

    void GetGeomFaces()
    {
        /*  Iterate over TopoDS_Shape and extract all the faces, convert the faces into Geom_Surface and
            get their handles
        */

        this->geometry_faces.clear();

        for (TopExp_Explorer explorer(this->imported_shape,TopAbs_FACE); explorer.More(); explorer.Next())
        {

            // GET THE EDGES
            TopoDS_Face current_face = TopoDS::Face(explorer.Current());
            // CONVERT THEM TO GEOM_CURVE
            Handle_Geom_Surface surface = BRep_Tool::Surface(current_face);
            // STORE HANDLE IN THE CONTAINER
            this->geometry_faces.push_back(surface);

        }
    }

    void ProjectMeshOnCurve(std::string &method)
    {
        //void ProjectMeshOnCurve(std::string &ProjectionAlgorithm ="Geom_Curve")
        /* Projects all the points on the mesh to the boundary of Geom_Curve */

        this->projection_methoed = method;

        this->projection_ID = Eigen::MatrixXi::Zero(this->mesh_edges.rows(),this->ndim);
        this->projection_U = Eigen::MatrixXd::Zero(this->mesh_edges.rows(),this->ndim);
        this->dirichlet_edges = Eigen::MatrixXi::Zero(this->mesh_edges.rows(),this->ndim+1);

        this->listedges.clear();
        int index_edge = 0;

        for (int iedge=0; iedge<this->mesh_edges.rows(); ++iedge)
        {
//            for (int jedge=0; jedge<this->mesh_edges.cols(); ++jedge)
            Standard_Boolean flag = Standard_True;
            for (int jedge=0; jedge<this->ndim; ++jedge) // linear
            {
                Standard_Real x = this->mesh_points(this->mesh_edges(iedge,jedge),0);
                Standard_Real y = this->mesh_points(this->mesh_edges(iedge,jedge),1);
                if (sqrt(x*x+y*y)<this->condition)
                {
                    if (flag == Standard_True)
                    {
                        this->listedges.push_back(iedge);
                        for (int iter=0;iter<ndim;++iter){
                            dirichlet_edges(index_edge,iter) = this->mesh_edges(iedge,iter);
                        }
                        index_edge +=1;
                        flag = Standard_False;
                    }

                    Standard_Real min_distance = 1.0e10;
                    Standard_Real distance = 0.;
                    gp_Pnt project_this_point = gp_Pnt(x,y,0.0);
                    for (unsigned int kedge=0; kedge<this->geometry_edges.size(); ++kedge)
                    {
                        Standard_Real parameterU;
                        if (this->projection_methoed.compare("Newton")==0)
                        {
                            gp_Pnt proj;
                            Standard_Real prec = 0;

                            ShapeAnalysis_Curve proj_curve;
//                            distance = proj_curve.Project(this->geometry_edges[kedge],project_this_point,prec,proj,parameterU);
                            GeomAdaptor_Curve curve_adapt(this->geometry_edges[kedge]);
                            distance = proj_curve.NextProject((double)jedge,curve_adapt,project_this_point,prec,proj,parameterU);
                        }
                        else if (this->projection_methoed.compare("bisection")==0)
                        {
                            GeomAPI_ProjectPointOnCurve proj;
                            proj.Init(project_this_point,this->geometry_edges[kedge]);
                            distance = proj.LowerDistance();
                            parameterU = proj.LowerDistanceParameter();
                        }

                        // GET LENGTH OF THE CURVE
                        GeomAdaptor_Curve current_curve(this->geometry_edges[kedge]);
                        Standard_Real curve_length = GCPnts_AbscissaPoint::Length(current_curve);

                        if (distance < min_distance)
                        {
                            // STORE ID OF NURBS
                            this->projection_ID(iedge,jedge) = kedge;
                            // STORE PROJECTION POINT PARAMETER ON THE CURVE (NORMALISED)
                            this->projection_U(iedge,jedge) = this->scale*parameterU/curve_length;
                            min_distance = distance;
                        }

//                        cout << this->projection_U(iedge,jedge) <<  " "<< curve_length <<  endl;
//                        cout << proj.LowerDistance() << " " << proj.NearestPoint().X() << " " << proj.NearestPoint().Y() << endl;
//                        cout << this->projection_U(iedge,jedge) << " [" << proj.NearestPoint().X() << " " << proj.NearestPoint().Y() << "]" << endl;
//                        cout << proj.LowerDistanceParameter() << endl;
//                        cout << sqrt(x*x+y*y) << " "<< this->condition <<  endl;
//                        cout << "["<< proj.NearestPoint().X() << " " << proj.NearestPoint().Y() << "] [" << x << " "<< y <<"]" << endl;



                        // COMPUTE THE PARAMETER OF A POINT (Ui) AT A GIVEN DISTANCE FROM U0
                        //GCPnts_AbscissaPoint calc( curve, 500.0, 0. );
                        //if (calc.IsDone())
                          //  cout << calc.Parameter() << endl;

//                        Handle_Geom_Curve current_curve = this->geometry_edges[kedge];
//                        Standard_Real umin = current_curve->FirstParameter();
//                        Standard_Real umax = current_curve->LastParameter();
//                        Standard_Real len2 = GCPnts_AbscissaPoint::Length(curve,umin,umax); // IMPORTANT
//                        cout << len << " " << " " << len2 << " " << umin <<" "<< umax << endl;

//                        gp_Ax1 ax = gp_Ax1();
//                        Geom_Line line = Geom_Line(ax);
//                        GeomAdaptor_Curve lline(line);
//                        cout << proj.Parameter(1)/2./3.14159 << "  " << proj.Parameter(2)/2./3.14159 << endl;
//                        cout << proj.LowerDistance() <<  "   "<< distance2 << endl;
//                        cout << param << " "<< proj.LowerDistanceParameter() << endl;
//                        cout << " [" << proj.NearestPoint().X() << " " << proj.NearestPoint().Y() << "]  [" << p2.X() << " " << p2.Y()<< "]"  << endl;
                    }
                }
            }
        }
//        print(dirichlet_edges);
        Eigen::MatrixXi arr_rows = cnp::arange(index_edge);
        Eigen::MatrixXi arr_cols = cnp::arange(ndim+1);
        //cout << this->dirichlet_edges << endl << endl;
        this->dirichlet_edges = cnp::take(this->dirichlet_edges,arr_rows,arr_cols);
        //cout << this->dirichlet_edges << endl;
        // SORT PROJECTED PARAMETERS ROW BY ROW
        //cout << this->projection_U<< endl<<endl;
        cnp::sort_rows(this->projection_U);
        //cout << this->projection_U<< endl<<endl;
    }

    void ProjectMeshOnSurface()
    {
        /* Projects all the points on the mesh to the boundary of Geom_Curve */
        this->projection_ID = Eigen::MatrixXi::Zero(this->mesh_edges.rows(),this->ndim);
        this->projection_U = Eigen::MatrixXd::Zero(this->mesh_edges.rows(),this->ndim);
        this->projection_V = Eigen::MatrixXd::Zero(this->mesh_edges.rows(),this->ndim);
        this->dirichlet_faces = Eigen::MatrixXi::Zero(this->mesh_edges.rows(),this->ndim+1);
        this->listfaces.clear();
        int index_face = 0;

        for (int iface=0; iface<this->mesh_edges.rows(); ++iface)
        {
            for (int jface=0; jface<this->ndim; ++jface) // linear
            {
                Standard_Real x = this->mesh_points(this->mesh_edges(iface,jface),0);
                Standard_Real y = this->mesh_points(this->mesh_edges(iface,jface),1);
                Standard_Real z = this->mesh_points(this->mesh_edges(iface,jface),2);
                if (sqrt(x*x+y*y+z*z)<this->condition)
                {
                    if (jface==0)
                    {
                        this->listfaces.push_back(iface);
                        for (int iter=0;iter<ndim;++iter)
                            dirichlet_faces(index_face,iter) = this->mesh_edges(iface,iter);
                        index_face +=1;
                    }

                    Standard_Real min_distance = 1.0e10;
                    Standard_Real distance = 0.;
                    gp_Pnt project_this_point = gp_Pnt(x,y,0.0);
                    for (unsigned int kface=0; kface<this->geometry_faces.size(); ++kface)
                    {
                        GeomAPI_ProjectPointOnSurf proj;
                        proj.Init(project_this_point,this->geometry_faces[kface]);
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

    void RepairDualProjectedParameters()
    {
        // GET MIN AND MAX PARAMETERS OF EACH CURVE
        Standard_Integer no_of_curves = this->geometry_edges.size();
        Eigen::MatrixXd u_min = Eigen::MatrixXd::Zero(no_of_curves,1);
        Eigen::MatrixXd u_max = Eigen::MatrixXd::Zero(no_of_curves,1);

        u_min = (u_min.array() + 1.0e10).matrix();
        u_max = (u_max.array() - 1.0e10).matrix();

        for (unsigned int iedge=0; iedge< this->listedges.size(); ++iedge)
        {
            Standard_Real u1 = this->projection_U(iedge,0);
            Standard_Real u2 = this->projection_U(iedge,1);
            Standard_Integer ID_nurbs = this->projection_ID(iedge,1);
            Eigen::Vector3d dum_1; dum_1 << u_min(ID_nurbs), u1, u2;
            u_min(ID_nurbs) = dum_1.minCoeff();
            Eigen::Vector3d dum_2; dum_2 << u_max(ID_nurbs), u1, u2;
            u_max(ID_nurbs) = dum_2.maxCoeff();
        }

        Standard_Real lengthTol = 1.0e-10;
        for (unsigned int iedge=0;iedge<this->listedges.size();++iedge)
        {
            int current_dir_edge = this->listedges[iedge];
            int id_curve = this->dirichlet_edges(iedge,2); // CHECK the row-counter later
            // GET THE CURRENT EDGE
            Eigen::Matrix<double,1,2> current_edge_U = this->projection_U.row(current_dir_edge);
            // SORT IT
            //cout << "before" << this->projection_U.row(current_dir_edge) << endl;
            std::sort(current_edge_U.data(),current_edge_U.data()+current_edge_U.cols());
            //cout << "after" << this->projection_U.row(current_dir_edge)  << endl;
            Standard_Real u1 = this->geometry_edges[id_curve]->FirstParameter()/cnp::length(this->geometry_edges[id_curve]);
            Standard_Real u2 = this->geometry_edges[id_curve]->LastParameter()/cnp::length(this->geometry_edges[id_curve]);
            //cout << u1 << " "<< u2<< endl;
            if ( abs(current_edge_U(0) - u1) < lengthTol )
            {
                 // WE ARE AT THE STARTING PARAMETER - GET THE LENGTH TO uc
                 Standard_Real uc = current_edge_U(1);
                 GeomAdaptor_Curve dum_curve(this->geometry_edges[id_curve]);
                 Standard_Real l0c = GCPnts_AbscissaPoint::Length(dum_curve,u1,uc)/this->scale;
                 Standard_Real l01 = GCPnts_AbscissaPoint::Length(dum_curve,u1,u2)/this->scale;

                 //if ( (l0c-u1) > (l01-l0c))
                 if ( l0c > (l01-l0c))
                 {
                     // FIX THE PARAMETER TO LastParameter
                     this->projection_U(current_dir_edge,0)=u2;
                 }
                 //cout << l0c << " " <<  l01 << endl<<endl;
            }
            else if ( abs(current_edge_U(1) - u2) < lengthTol )
            {
                 // WE ARE AT THE STARTING PARAMETER - GET THE LENGTH TO uc
                 Standard_Real uc = current_edge_U(0);
                 GeomAdaptor_Curve dum_curve(this->geometry_edges[id_curve]);
                 Standard_Real lc1 = GCPnts_AbscissaPoint::Length(dum_curve,uc,u2)/this->scale;
                 Standard_Real l0c = GCPnts_AbscissaPoint::Length(dum_curve,u1,uc)/this->scale;

                 //if ( (l0c-u1) > (l01-l0c))
                 if ( l0c < lc1 )
                 {
                     // FIX THE PARAMETER TO FirstParameter
                     this->projection_U(current_dir_edge,1)=u1;
                 }
                 //cout << l0c << " " <<  lc1 << endl<<endl;
            }

        }
        // SORT projection_U AGAIN
        cnp::sort_rows(this->projection_U);
        //cout << this->projection_U<< endl<<endl;

    }

    void RepairDualProjectedParameters_Old()
    {
        //cout << this->projection_U << endl;
        int no_of_curves = this->geometry_edges.size();
        Eigen::MatrixXd u_min = Eigen::MatrixXd::Zero(no_of_curves,1); //u_min += 1.0e10;
        Eigen::MatrixXd u_max = Eigen::MatrixXd::Zero(no_of_curves,1); //u_max += -1.0e10;
        Eigen::MatrixXd L_min = Eigen::MatrixXd::Zero(no_of_curves,1);
        Eigen::MatrixXd L_max = Eigen::MatrixXd::Zero(no_of_curves,1);

        u_min = (u_min.array() + 1.0e10).matrix();
        u_max = (u_max.array() - 1.0e10).matrix();

        for (unsigned int iedge=0; iedge< this->listedges.size(); ++iedge)
        {
            Standard_Real u1 = this->projection_U(iedge,0);
            Standard_Real u2 = this->projection_U(iedge,1);
            Standard_Integer ID_nurbs = this->projection_ID(iedge,1);
            Eigen::Vector3d dum_1; dum_1 << u_min(ID_nurbs), u1, u2;
            u_min(ID_nurbs) = dum_1.minCoeff();
            Eigen::Vector3d dum_2; dum_2 << u_max(ID_nurbs), u1, u2;
            u_max(ID_nurbs) = dum_2.maxCoeff();
        }
        cout << u_min << " " <<  u_max << endl;

        Standard_Real lengthTol = 1.0e-10;
        for (unsigned int icurve=0; icurve < this->geometry_edges.size(); ++icurve)
        {
            Handle_Geom_Curve current_curve = this->geometry_edges[icurve];
            Standard_Real u1 = current_curve->FirstParameter(); // THIS SHOULD BE FIRST ELEMENT OF THE KNOT VECTOR
            Standard_Real u2 = u_min(icurve);
            if (u2 > current_curve->FirstParameter())
            {
                u2 = current_curve->FirstParameter();
            }
            if (abs(u1-u2) < lengthTol)
            {
                L_min(icurve) = 0;
            }
            else
            {
                GeomAdaptor_Curve dum_curve(current_curve);
                L_min(icurve) = GCPnts_AbscissaPoint::Length(dum_curve,u1,u2);
            }


            u1 = u_max(icurve);
            if (u1< current_curve->LastParameter())
            {
                u1 = current_curve->LastParameter();
            }
            u2 = current_curve->LastParameter(); // THIS IS FIRST ELEMENT OF THE KNOT VECTOR
            if (abs(u1-u2) < lengthTol)
            {
                L_max(icurve) = 0;
            }
            else
            {
                GeomAdaptor_Curve dum_curve(current_curve);
                L_max(icurve) = GCPnts_AbscissaPoint::Length(dum_curve,u1,u2);
            }
        }
        cout << L_min << " " << L_max << endl;


        Eigen::MatrixXd correctMaxMin = -Eigen::MatrixXd::Ones(this->geometry_edges.size(),1);
        for (unsigned icurve=0; icurve<this->geometry_edges.size(); ++icurve)
        {
            if (L_min(icurve) < L_max(icurve))
            {
                correctMaxMin(icurve) = 0;
            }
            else
            {
                correctMaxMin(icurve) = 1;
            }
        }

        int index_edge = 0;
//        cout << Eigen::Map<Eigen::VectorXi>(this->listedges.data(),this->listedges.size()) << endl;
        for (unsigned int iedge=0; iedge< this->listedges.size(); ++iedge)
        {
            Standard_Real u1 = this->projection_U(iedge,0);
            Standard_Real u2 = this->projection_U(iedge,1);
            Standard_Integer ID_curve = this->dirichlet_edges(index_edge,2);
            index_edge +=1;
            if (correctMaxMin(ID_curve)==0)
            {
                // CORRECT MIN
                if ( (u1>u2) && abs(u2 - u_min(ID_curve)) < 1e-10)
                {
                    // PROBLEM WITH PERIODIC
                    this->projection_U(iedge,1) = this->geometry_edges[ID_curve]->LastParameter();  // THIS IS FIRST ELEMENT OF THE KNOT VECTOR
                }
                else if ( (u1<u2) && abs(u1 - u_min(ID_curve)) < 1e-10)
                {
                    // PROBLEM WITH PERIODIC
                    this->projection_U(iedge,0) = this->geometry_edges[ID_curve]->FirstParameter();  // THIS IS FIRST ELEMENT OF THE KNOT VECTOR
                }
            }

            else
            {
                // CORRECT MAX
                if ( (u1>u2) && abs(u1 - u_max(ID_curve)) < 1e-10)
                {
                    // PROBLEM WITH PERIODIC
                    this->projection_U(iedge,0) = this->geometry_edges[ID_curve]->FirstParameter();  // THIS IS FIRST ELEMENT OF THE KNOT VECTOR
                }
                else if ( (u1<u2) && abs(u2 - u_max(ID_curve)) < 1e-10)
                {
                    // PROBLEM WITH PERIODIC
                    this->projection_U(iedge,1) = this->geometry_edges[ID_curve]->LastParameter();  // THIS IS FIRST ELEMENT OF THE KNOT VECTOR
                }
            }
        } //cout << this->projection_U << endl;
    }

    void CurvesToBsplineCurves()
    {
        /* Converts all the imported curves to bspline curves. Apart from circle all the remaining converted
         * curves will be non-periodic: http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
         */
        this->geometry_edges_bspline.clear();
        for (unsigned int icurve=0; icurve < this->geometry_edges.size(); ++icurve)
        {
            this->geometry_edges_bspline.push_back( GeomConvert::CurveToBSplineCurve(this->geometry_edges[icurve]) );
        }
    }

    void SurfacesToBsplineSurfaces()
    {
        /* Converts all the imported surfaces to bspline surfaces : http://dev.opencascade.org/doc/refman/html/class_geom_convert.html
         */
        this->geometry_faces_bspline.clear();
        for (unsigned int isurf=0; isurf < this->geometry_faces.size(); ++isurf)
        {
            this->geometry_faces_bspline.push_back( GeomConvert::SurfaceToBSplineSurface(this->geometry_faces[isurf]) );
        }
    }

    void MeshPointInversionCurve()
    {
        this->no_dir_edges = this->listedges.size();
        Standard_Real no_edge_nodes = this->mesh_edges.cols();
        Eigen::MatrixXi arr_row = Eigen::Map<Eigen::VectorXi>(this->listedges.data(),this->listedges.size());
        Eigen::MatrixXi arr_col = cnp::arange(0,no_edge_nodes);
        Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> nodes_dir = cnp::take(this->mesh_edges,arr_row,arr_col);
        nodes_dir = cnp::ravel(nodes_dir);
        this->index_nodes = cnp::arange(no_edge_nodes);
        this->displacements_BC = Eigen::MatrixXd::Zero(this->no_dir_edges*no_edge_nodes,2);

        this->FeketePoints1D();

        for (int idir=0; idir< this->no_dir_edges; ++idir)
        {
            int id_curve = this->dirichlet_edges(idir,2);
            Standard_Real u1 = this->projection_U(this->listedges[idir],0);
            Standard_Real u2 = this->projection_U(this->listedges[idir],1);
            Handle_Geom_Curve current_curve = this->geometry_edges[id_curve];
            Eigen::MatrixXd fekete_1d_curve = FeketePointsOnCurve(current_curve,u1,u2);
            Standard_Real length_current_curve = cnp::length(current_curve);

            GeomAdaptor_Curve current_curve_adapt(current_curve);
            for (int j=0; j<no_edge_nodes;++j)
            {
//                int i = no_edge_nodes - j-1;
                Standard_Real tol =1e-08;
//                GCPnts_AbscissaPoint inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,fekete_1d_curve(j)*this->scale,0.);
//                GCPnts_AbscissaPoint inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,fekete_1d_curve(this->boundary_points_order(i))*this->scale,0.);
//                cout << fekete_1d_curve(this->boundary_points_order(j)) << " " << fekete_1d_curve(j)  << endl;
                GCPnts_AbscissaPoint inv = GCPnts_AbscissaPoint(tol,current_curve_adapt,fekete_1d_curve( this->boundary_points_order(j) )*this->scale,0.);
                Standard_Real uEq = inv.Parameter();
                gp_Pnt xEq;
                current_curve_adapt.D0(uEq*length_current_curve,xEq);
//                cout << uEq << endl;
//                cout << xEq.X()/this->scale << " " << xEq.Y()/this->scale << endl;
                Eigen::MatrixXd gp_pnt_old = (this->mesh_points.row(nodes_dir(this->index_nodes( j  ))).array()/this->scale);
//                cout << "[" << gp_pnt_old(0) << "," << gp_pnt_old(1)  << "] ["<< xEq.X()/this->scale << " " << xEq.Y()/this->scale << "]" << endl;
                //cout << gp_pnt_old(0) << "," << gp_pnt_old(1)  << endl;
                this->displacements_BC(this->index_nodes(j),0) = (xEq.X()/this->scale - gp_pnt_old(0));
                this->displacements_BC(this->index_nodes(j),1) = (xEq.Y()/this->scale - gp_pnt_old(1));
//                cout << this->index_nodes(j) << endl;

            }
//            cout << this->index_nodes << endl<<endl;
            this->index_nodes = ((this->index_nodes).array()+no_edge_nodes).eval().matrix();
//            cout << " " << endl;
        }

        cout << displacements_BC << endl;
        // FIND NORMALISED FEKETE POINTS ON THE CURVE [0,1] NOT [0,LENGTH_CURVE]


    }

    void MeshPointInversionSurface()
    {

    }

    void FeketePoints1D()
    {
        const int p = this->mesh_edges.cols()-1;
        Eigen::Matrix<double,3,1> dum;

        if (this->ndim==2)
        {
            if (p==2)
            {
                dum << -1., 0., 1.;
            }
            else if (p==3)
            {
                dum << -1.,-0.447213595499957983,0.447213595499957928,1.;
            }
            else if (p==4)
            {
                dum <<-1.,-0.654653670707977198,0.,0.654653670707977198,1.;
            }
        }

        this->fekete_1d = dum;
        this->boundary_points_order = Eigen::MatrixXi::Zero(this->fekete_1d.rows(),this->fekete_1d.cols());
//        this->boundary_points_order(1) = this->fekete_1d.rows()-1;
        this->boundary_points_order(0) = this->fekete_1d.rows()-1;
        this->boundary_points_order.block(2,0,fekete_1d.rows()-2,1) = cnp::arange(1,fekete_1d.rows()-1);
        //exit (EXIT_FAILURE);
    }

    Eigen::MatrixXd FeketePointsOnCurve(Handle_Geom_Curve &curve,Standard_Real &u1,Standard_Real &u2)
    {
//        Standard_Real u1 = curve->FirstParameter();
//        Standard_Real u2 = curve->LastParameter();
//        this->FeketePoints1D();
        Eigen::MatrixXd fekete_1d_curve;
//        std::string type_d = "normalised";
//        if (std::strcmp(type_p,type_d)==0 )
//        {
//            fekete_1d_curve = (((u2-u1)/2.)*this->fekete_1d).eval();
//        }
//        fekete_1d_curve = ((1.0/2.0)*((this->fekete_1d).array()+1.).matrix()).eval();
        fekete_1d_curve = (u1 + (u2-u1)/2.0*((this->fekete_1d).array()+1.)).matrix();
        return fekete_1d_curve;
    }
};




int main()
{

    clock_t begin = clock();


    string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
    string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
    string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
    string unique_edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";


    std::string element_type = "tri";
    int ndim = 2;
    OCC_FrontEnd MainDataCpp = OCC_FrontEnd(element_type,ndim);

    // READ ELEMENT CONNECTIVITY, NODAL COORDINATES, EDGES & FACES
    MainDataCpp.SetElementType(element_type);
    MainDataCpp.ReadMeshConnectivityFile(elem_file,',');
    MainDataCpp.ReadMeshCoordinateFile(point_file,',');
    MainDataCpp.ReadMeshEdgesFile(edge_file,',');
    MainDataCpp.ReadUniqueEdges(unique_edge_file);
    MainDataCpp.CheckMesh();

    Standard_Real condition = 2000.;
//    Standard_Real condition = 2.;
    MainDataCpp.SetCondition(condition);
    MainDataCpp.mesh_points *= 1000.0;

    // READ THE GEOMETRY FROM THE IGES FILE
    //std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs";
//    std::string filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
    std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";

//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
    MainDataCpp.ReadIGES(filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    MainDataCpp.GetGeomEdges();
    MainDataCpp.GetGeomFaces();
//    if (MainDataCpp.mesh_element_type.compare("tet") == 0)
//    {
//        MainDataCpp.GetGeomFaces();
//    }

//    cout << MainDataCpp.geometry_edges.size() << " " << MainDataCpp.geometry_faces.size() << endl;

    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
//    std::string method = "Newton";
    std::string method = "bisection";
    MainDataCpp.ProjectMeshOnCurve(method);
    // CONVERT CURVES TO BSPLINE CURVES
    MainDataCpp.CurvesToBsplineCurves();
    // FIX IAMGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    //MainDataCpp.RepairDualProjectedParameters_Old();
    MainDataCpp.RepairDualProjectedParameters();

    // PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    MainDataCpp.MeshPointInversionCurve();

//    enum colors {a=10,b=20,c=30};
//    colors cc = a;
//    colors::a;
//    cout << cc << endl;

//    Geom_Curve xff;
//    Handle_Geom_Curve yy = xff.getHandle();



    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "Total time elapsed was " << elapsed_secs << " seconds" << endl;

    return 0;
}

