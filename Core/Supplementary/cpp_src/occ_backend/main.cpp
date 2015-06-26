#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <stdexcept>
#include <cstdarg>
#include <cmath>
#include <ctime>
//#include <tuple>

#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>



#include <IGESControl_Reader.hxx>
#include <XSControl_Reader.hxx>
#include <Interface_Static.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Shape.hxx>
#include <TopoDS_Edge.hxx>
#include <Geom_BSplineCurve.hxx>
#include <GeomConvert.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <TopExp_Explorer.hxx>
#include <BRep_Tool.hxx>
#include <BRepBuilderAPI_NurbsConvert.hxx>
#include <gp_Pnt2d.hxx>
#include <Geom2dAPI_ProjectPointOnCurve.hxx>
#include <Geom2d_BSplineCurve.hxx>
#include <Geom_TrimmedCurve.hxx>
#include <Geom2d_TrimmedCurve.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TColgp_Array1OfPnt2d.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <Geom_Circle.hxx>
#include <gp.hxx>
#include <gp_Circ.hxx>
#include <GeomAPI_ProjectPointOnCurve.hxx>


using namespace std;

// auxilary functions
// split strings
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}
std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

// print functions
template<typename Derived>
void print(Eigen::MatrixBase<Derived> &A)
{
    int rows = A.rows();
    int cols = A.cols();
    for(int i=0 ; i<rows;i++)
    {
        for (int j=0;j<cols; j++)
        {
            cout << A(i,j) << " ";
        }
        cout << endl;
    }
}
template<typename Derived>
void print(Derived A)
{
    std::cout << A << std::endl;
}
void print(const char* fmt...)
{
    va_list args;
    va_start(args, fmt);

    while (*fmt != '\0') {
        if (*fmt == 'd') {
            int i = va_arg(args, int);
            std::cout << i << ", ";
        } else if (*fmt == 'c') {
            // note automatic conversion to integral type
            int c = va_arg(args, int);
            std::cout << static_cast<char>(c) << ", ";
        } else if (*fmt == 'f') {
            double d = va_arg(args, double);
            std::cout << d << ", ";
        }
        ++fmt;
    }

    va_end(args);
    std::cout << std::endl;
}
// end of print functions




// a list of numpy inspired functions, put inside a namespace just for convenience
namespace cpp_numpy {

Eigen::MatrixXi arange(int a, int b)
{
    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
Eigen::MatrixXi arange(int &a, int &b)
{
    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
Eigen::MatrixXi take(Eigen::MatrixXi& arr, Eigen::MatrixXi &arr_row, Eigen::MatrixXi &arr_col)
{

    // the template version does not work
    //template<typename Derived>
    //Eigen::MatrixBase<Derived> take(Eigen::MatrixBase<Derived>& arr, Eigen::MatrixXi &arr_row, Eigen::MatrixXi &arr_col)
    //template<typename DerivedA, typename DerivedB>
    //Eigen::MatrixBase<DerivedA> take(Eigen::MatrixBase<DerivedA>& arr, Eigen::MatrixBase<DerivedB> &arr_row, Eigen::MatrixBase<DerivedB> &arr_col)

//    Eigen::MatrixBase<Derived> arr_reduced = Eigen::MatrixBase<Derived>::Zero(arr_row.rows(),arr_col.rows());
//    Eigen::MatrixBase<DerivedA> arr_reduced = Eigen::MatrixBase<DerivedA>::Zero(arr_row.rows(),arr_col.rows());
//    Eigen::MatrixBase<DerivedA> arr_reduced = Eigen::MatrixBase<DerivedA>::Random(arr_row.rows(),arr_col.rows());
//    Eigen::MatrixBase<DerivedA> arr_reduced(arr_row.rows(),arr_col.rows());
//    Eigen::MatrixXi arr_reduced(arr_row.rows(),arr_col.rows());
    Eigen::MatrixXi arr_reduced = Eigen::MatrixXi::Zero(arr_row.rows(),arr_col.rows());

    for (int i=0; i<arr_row.rows();i++)
    {
        for (int j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}
Eigen::MatrixXd take(Eigen::MatrixXd& arr, Eigen::MatrixXi &arr_row, Eigen::MatrixXi &arr_col)
{
    Eigen::MatrixXd arr_reduced = Eigen::MatrixXd::Zero(arr_row.rows(),arr_col.rows());

    for (int i=0; i<arr_row.rows();i++)
    {
        for (int j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

} // end of namespace

// a syntactic sugar equivalent to import to numpy as np
namespace cnp =cpp_numpy;



// occ_backend class
class occ_backend
{
public:
    // constructor
    occ_backend() {
      this->element_type ="tri";
    }

    // members of occ_ backend
    std::string element_type;// = "tri";
    Eigen::MatrixXi elements;
    Eigen::MatrixXd points;
    Eigen::MatrixXi edges;
    Eigen::MatrixXi faces;

    // methods of occ_backend
    void set_element_type(std::string &type)
    {
        this->element_type = type;
    }
    void set_elements(Eigen::MatrixXi &arr)
    {
        this->elements=arr;
    }
    void set_points(Eigen::MatrixXd &arr)
    {
        this->points=arr;
    }
    void set_edges(Eigen::MatrixXi &arr)
    {
        this->edges=arr;
    }
    void set_faces(Eigen::MatrixXi &arr)
    {
        this->faces=arr;
    }
    void read_mesh_elements(string &filename, char delim)
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

        this->elements = arr_read;
    }
    void read_mesh_points(std::string &filename, char delim)
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

        this->points = arr_read;
    }
    void read_mesh_edges(std::string &filename, char delim)
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

        this->edges = arr_read;
    }
    void read_mesh_faces(std::string &filename, char delim)
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

        this->faces = arr_read;
    }
    template<typename Derived> Eigen::MatrixBase<Derived> read_template(std::string &filename, char delim)
    {
        /* should work if you move it to the header file */
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

        Eigen::MatrixBase<Derived> out_arr = Eigen::MatrixBase<Derived>::Zero(rows,cols);

        for(int i=0 ; i<rows;i++)
        {
            elems = split(arr[0], delim);
            for (unsigned int j=0;j<cols; j++)
            {
                out_arr(i,j) = std::atof(elems[j].c_str());
            }
        }

        return out_arr;

    }
    Eigen::MatrixXd read(std::string &filename, char delim)
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

        Eigen::MatrixXd out_arr = Eigen::MatrixXd::Zero(rows,cols);

        for(int i=0 ; i<rows;i++)
        {
            elems = split(arr[0], delim);
            for (signed int j=0;j<cols; j++)
            {
                out_arr(i,j) = std::atof(elems[j].c_str());
            }
        }

        return out_arr;

    }
    void check_mesh()
    {
        /* checks correct mesh data are imported */

        int maxelem = this->elements.maxCoeff();
        int maxpoint = this->points.rows();
        if (maxelem+1 != maxpoint)
        {
            throw std::invalid_argument("element connectivity and nodal coordinates do not match. This can "
                                        "caused be giving two files which do not correspond to each other");
        }

        // check for multiple line copies of elements, points, edges and faces
        double check_duplicated_rows;
        int flag_p = 0;
        for (int i=this->points.rows()-2; i<this->points.rows();i++)
        {
            check_duplicated_rows = (this->points.row(i) - this->points.row(i-1)).norm();
            if (std::abs(check_duplicated_rows)  < 1.0e-14)
            {
                flag_p = 1;
            }
        }
        if (flag_p == 1)
        {
            Eigen::MatrixXi a_rows = cnp::arange(0,this->points.rows()-1);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->points.cols());
            this->points = cnp::take(this->points,a_rows,a_cols);
        }

        // elements
        int flag_e = 0;
        for (int i=this->elements.rows()-2; i<this->elements.rows();i++)
        {
            check_duplicated_rows = (this->elements.row(i) - this->elements.row(i-1)).norm();
            if (std::abs(check_duplicated_rows)  < 1.0e-14)
            {
                flag_e = 1;
            }
        }
        if (flag_e == 1)
        {
            Eigen::MatrixXi a_rows = cnp::arange(0,this->elements.rows()-1);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->elements.cols());
            this->elements = cnp::take(this->elements,a_rows,a_cols);
        }

        // edges
        int flag_ed = 0;
        for (int i=this->edges.rows()-2; i<this->edges.rows();i++)
        {
            check_duplicated_rows = (this->edges.row(i) - this->edges.row(i-1)).norm();
            if (std::abs(check_duplicated_rows)  < 1.0e-14)
            {
                flag_ed = 1;
            }
        }
        if (flag_ed == 1)
        {
            Eigen::MatrixXi a_rows = cnp::arange(0,this->edges.rows()-1);
            Eigen::MatrixXi a_cols = cnp::arange(0,this->edges.cols());
            this->edges = cnp::take(this->edges,a_rows,a_cols);
        }

        // faces for 3D
        if (this->faces.cols()!=0)
        {
            int flag_f = 0;
            for (int i=this->faces.rows()-2; i<this->faces.rows();i++)
            {
                check_duplicated_rows = (this->faces.row(i) - this->faces.row(i-1)).norm();
                if (std::abs(check_duplicated_rows)  < 1.0e-14)
                {
                    flag_f = 1;
                }
            }
            if (flag_f == 1)
            {
                Eigen::MatrixXi a_rows = cnp::arange(0,this->faces.rows()-1);
                Eigen::MatrixXi a_cols = cnp::arange(0,this->edges.cols());
                this->faces = cnp::take(this->faces,a_rows,a_cols);
            }
        }

        std::cout << "all good with imported mesh. proceeding..." << std::endl;

    }
};

int main()
{
        clock_t begin = clock();

        string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle.dat";
        string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle.dat";
        string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle.dat";

        occ_backend occ_meshes = occ_backend();

        // read element connectivity, nodal coordinates and edges
        std::string element_type = "tri";
        occ_meshes.set_element_type(element_type);
        occ_meshes.read_mesh_elements(elem_file,',');
        occ_meshes.read_mesh_points(point_file,',');
        occ_meshes.read_mesh_edges(edge_file,',');
        occ_meshes.check_mesh();

        /*
//        print(cnp::arange(0,2));
//        cout << Eigen::VectorXi::LinSpaced(6,0,5).transpose() << endl;
//        cout << Eigen::VectorXi::LinSpaced(5,0,4).transpose() << endl;

//        int a=0; int b=10;
//        Eigen::MatrixXi aa = Eigen::MatrixXi::Random(6,4);
//        Eigen::MatrixXd aa = Eigen::MatrixXd::Random(6,4);
//        Eigen::MatrixXi a1 = cnp::arange(0,5);
//        Eigen::MatrixXi a2 = cnp::arange(0,4);

//        print(aa); cout<<endl;
//        Eigen::MatrixXi arr_reduced = take(aa,a1,a2);
//        Eigen::MatrixXd arr_reduced = cnp::take(aa,a1,a2);
//        print(arr_reduced);


//        Eigen::MatrixXd points = mesh.read(point_file,',');


//        print(mesh.elements);
//        print(mesh.points);
//        print(mesh.edges);


//        std::tuple<int,int> foo (0,5);
//        print("dcff", 3, 'a', 1.999, 42.5);
//        std::string dd = "what ever";
//        print("what ever"); // does not work
//        print(dd);
//        std::cout << dd << endl;
         */


        // read iges file now
        IGESControl_Reader reader;
        IFSelect_ReturnStatus stat  = reader.ReadFile("/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs");
//        IFSelect_ReturnStatus stat  = reader.ReadFile("/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs");
//        IFSelect_ReturnStatus stat  = reader.ReadFile("/home/roman/Desktop/Circle_1.igs");
        reader.PrintCheckLoad(Standard_True,IFSelect_GeneralInfo);
        reader.PrintCheckTransfer(Standard_True,IFSelect_ItemsByEntity);
//        if  (!Interface_Static::SetIVal ("read.iges.bspline.continuity",0))
//        {
//            cout << "what";
//        }
        // read the IGES file as is
        Interface_Static::SetIVal("read.iges.bspline.continuity",0);
        Standard_Integer ic =  Interface_Static::IVal("read.iges.bspline.continuity");
        if (ic !=0)
        {
            cout << "IGES file was not read as-is. The file was not read/transformed correctly";
        }
//        cout << ic;
//        if (ic==2)
//        {
//            throw invalid_argument("IGES file was not read/transformed correctly");
//        }

        Standard_Integer ok = reader.TransferRoots();
        Standard_Integer nbs = reader.NbShapes();
        TopoDS_Shape imported_shape  = reader.OneShape();

//        TopoDS_Edge new_edge;
        // explore the edges of the imported shape (loop essentially)
        std::vector<double> edge_knots_stl;
        edge_knots_stl.clear();
        std::vector<int> edge_mult_stl;
        edge_mult_stl.clear();
        for (TopExp_Explorer explorer(imported_shape,TopAbs_EDGE); explorer.More(); explorer.Next())
        {
            // get the edge
            TopoDS_Edge current_edge = TopoDS::Edge(explorer.Current());
            // convert to Geom_Curve
//            double first, last;
//            Handle(Geom_Curve) curve_2 = BRep_Tool::Curve(current_edge,first,last);

            BRepAdaptor_Curve curve = BRepAdaptor_Curve(current_edge);
            // after conversion get knot vector, control points, degree etc
            Standard_Real start = curve.FirstParameter();
            Standard_Real end = curve.LastParameter();
            Standard_Integer degree = curve.Degree();
            Standard_Integer nU = curve.NbKnots();
            Standard_Integer nPw = curve.NbPoles();
            Standard_Boolean isrational = curve.IsRational();
            Standard_Boolean isperiodic = curve.IsPeriodic();
            Standard_Boolean isclosed = curve.IsClosed();
            // get type, i.e.  Line, Circle, Ellipse, Hyperbola, Parabola, BezierCurve, BSplineCurve, OtherCurve
            GeomAbs_CurveType curve_type = curve.GetType(); // VERY IMPORTANT

            Handle(Geom_BSplineCurve) curve_bsp = curve.BSpline();
//            Handle_Geom2d_BSplineCurve spline_curve = Handle(Geom_BSplineCurve)::DownCast(curve);
//            Standard_Real First,Last;
//            Handle(Geom_Curve) myCurve = BRep_Tool::Curve(current_edge, First, Last);
//            Handle(Geom_TrimmedCurve) myTrimmed = new Geom_TrimmedCurve(myCurve, First, Last);
//            Handle(Geom_BSplineCurve) myBSplCurve = GeomConvert::CurveToBSplineCurve(myTrimmed);

//            int knot0 = curve_bsp->Knot(1);
            cout << "Number of poles (control points) is: " << curve_bsp->NbPoles() << endl;
            cout << "Number of knots is: " << curve_bsp->NbKnots() << endl;

//            GeomAPI_ProjectPointOnCurve(gp_Pnt(0,1,0),curve_bsp);


            // build a Geom2d_Bspline from Geom_Bspline
//            TColgp_Array1OfPnt Pa = new TColgp_Array1OfPnt(1,nPw); //curve_bsp->NbPoles())

            TColgp_Array1OfPnt2d Poles2d(1,nPw);
//            curve_bsp->Poles(Poles2d);
            TColStd_Array1OfReal Knots2d(1,nU);
            curve_bsp->Knots(Knots2d);
            TColStd_Array1OfInteger Mult2d(1,nU);
            curve_bsp->Multiplicities(Mult2d);


            Handle_Geom2d_BSplineCurve curve_bsp2D_h;
            curve_bsp2D_h->SetKnots(Knots2d);
//            curve_bsp2D_h->set


            for (int i=0;i<nPw;++i)
            {
                Poles2d.SetValue(i+1,gp_Pnt2d(curve_bsp->Pole(i).X(),curve_bsp->Pole(i).Y() ));
                curve_bsp2D_h->SetPole( i+1,gp_Pnt2d( curve_bsp->Pole(i).X(),curve_bsp->Pole(i).Y() ));

            }
//            TColStd_Array1OfReal Knots2d(1,nU);
//            TColStd_Array1OfInteger Mult2d(1,nU);
//            for (int i=0;i<nU;++i)
//            {
//                Knots2d.SetValue(i+1,curve_bsp->Knot(i) );
//                Mult2d.SetValue(i+1,curve_bsp->Multiplicity(i));
//            }

//            cout << Knots2d;

//            curve_bsp->Poles(P);
//            cout << P;
            Geom2d_BSplineCurve curve_bsp2D(Poles2d,Knots2d,Mult2d,degree,isperiodic);
//            curve_bsp2D.
//            Handle(Geom2d_Geometry) cc = curve_bsp2D.Copy();
//            Handle_Geom2d_BSplineCurve curve_bsp2D_h = Handle_Geom2d_BSplineCurve::DownCast(Handle(curve_bsp2D));

            gp_Pnt xx = curve_bsp->Pole(1);
//            cout << curve_bsp->Knot(0) << endl;
//            cout << curve_bsp->Multiplicity(1) << endl;

            // indices start from 1
//            #pragma omp parallel for
            for (int i=1; i<=nU; ++i)
            {
//                cout << curve_bsp->Knot(i) << " with multiplicity of " << curve_bsp->Multiplicity(i) << endl;
                // Put these into an Eigen matrix
                edge_knots_stl.push_back(curve_bsp->Knot(i));
                edge_mult_stl.push_back(curve_bsp->Multiplicity(i));

            }

//            cout << curve.NbKnots() << endl;
//            cout << knot0 <<  " " << xx.X() << " "<< xx.Y() << " "<< xx.Z()<< endl;

//            curve_bsp->Pole(0);
//            curve_bsp->Pole(1);

//            cout << curve_type;

//            if (imported_shape->IsKind(STANDARD_TYPE(Geom_BSplineCurve)))
//            {
//               Handle(Geom_BSplineCurve) spline_curve = Handle(Geom_BSplineCurve)::DownCast(curve);
//            }

//            spline_curvAccess()
//            BRepBuilderAPI_NurbsConvert nurbs = BRepBuilderAPI_NurbsConvert(imported_shape);
//            Geom_BSplineCurve curve_bsp1 = curve.BSpline();
//            curve_bsp1.
//            Standard_Transient curve_4 =  curve_bsp.
//            curve_4.
//            GeomAdaptor_Curve curve_3 = curve.Curve();

//            cout << degree << " " << nU <<  " " << nPw << "" << start << " " << end << endl;



            // projection
            gp_Pnt2d point_to_project = gp_Pnt2d(0,1);
//            Geom2dAPI_ProjectPointOnCurve projected_point;
            Geom2dAPI_ProjectPointOnCurve(point_to_project,curve_bsp2D_h);
//            Geom2dAPI_ProjectPointOnCurve(point_to_project,Handle(curve_bsp2D));
//            point2.Init(point,curve_bsp);
//            gp::gp_XOY();

        }


        Handle_Geom_Circle circle;// = circle.Copy();
        circle->SetRadius(1);
        circle->SetAxis(gp_Ax1());
        circle->SetLocation(gp_Pnt(0,0,0));

        GeomAPI_ProjectPointOnCurve projected_points;// = GeomAPI_ProjectPointOnCurve(gp_Pnt(0,1,0),circle);
        cout << projected_points.LowerDistance();
//        cout << project_points.NbPoints();


        Eigen::Map<Eigen::VectorXd> edge_knots(edge_knots_stl.data(),edge_knots_stl.size());
        Eigen::Map<Eigen::VectorXi> edge_mult(edge_mult_stl.data(),edge_mult_stl.size());
//        cout << edge_knots;
//        cout << edge_mult;

        std::vector<Eigen::MatrixXd> knots;
        for (int i; i<edge_knots.rows();i++)
        {
//            cout << "what";
            continue;
        }

        Eigen::MatrixXd knots_mult = Eigen::MatrixXd::Zero(edge_knots.rows(),2);
        for (int i=0; i< edge_knots.rows();++i)
        {
            knots_mult(i,0) = edge_knots(i);
            knots_mult(i,1) = (double) edge_mult(i);
        }

        cout << endl << knots_mult << endl;




//        for (int i=0;i<nbs;i++)
//        {
////            TopoDS_Shape imported_shape = reader.Shape(i);
//            TopoDS_Edge new_edge;
//            for (TopExp_Explorer explorer(imported_shape,TopAbs_EDGE); explorer.More(); explorer.Next())
//            {
//                TopoDS_Edge current_edge = TopoDS::Edge(explorer.Current());
//            }
//        }
//        cout << nbs << endl;






        clock_t end = clock();
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout << endl << "Total time elapsed was: " << elapsed_secs << " seconds" << endl;

    return 0;
}

