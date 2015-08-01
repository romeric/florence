//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <py_to_occ_frontend.hpp>

using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;



int main()
{

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();


//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
//    std::string unique_edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/elements_circle_p4.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/points_circle_p4.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/edges_circle_p4.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_half_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_half_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_half_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_twoarcs_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_twoarcs_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_twoarcs_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/elements_irae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/points_irae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/edges_irae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/elements_i2rae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/points_i2rae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/edges_i2rae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_mech2d_seg0_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_mech2d_seg0_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_mech2d_seg0_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_mech2d_seg2_p2.dat"; //#
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_mech2d_seg2_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_mech2d_seg2_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/MechanicalComponent2D/elements_mech2dn_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/MechanicalComponent2D/points_mech2dn_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/MechanicalComponent2D/edges_mech2dn_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_leftpartwithcircle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_leftpartwithcircle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_leftpartwithcircle_p2.dat";

    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/elements_leftcircle_p2.dat";
    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/points_leftcircle_p2.dat";
    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/edges_leftcircle_p2.dat";


//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs";
//    const char* iges_filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/Mech2D_Seg0.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/LeftPartWithCircle.igs";
    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/LeftCircle.iges";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/Mech2D_Seg2.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/Half_Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Misc/Two_Arcs.iges";




    // CALL PY_OCC_FRONTEND
    Eigen::MatrixI elements = OCC_FrontEnd::ReadI(elem_file,',');
    Eigen::MatrixR points = OCC_FrontEnd::ReadR(point_file,',');
    Eigen::MatrixI edges = OCC_FrontEnd::ReadI(edge_file,',');
    Eigen::MatrixI faces = Eigen::MatrixI::Zero(1,4);
//    Eigen::MatrixI unique_edges = Read(unique_edge_file);
//    Real scale = 1000.;
//    Real condition = 2000.;
//    Real condition = 1000.;
//    Real condition = 2.0e20;

//    Real scale = 1000.;
//    Real condition = 2.;
//    Real scale = 1.; // for iso rae
//    Real condition = 0.6; // for iso rae
//    Real scale = 1000.; // for half-circle and two-arcs
//    Real condition = 3000.; // for half-circle and two-arcs
    Real scale = 1.;
    Real condition = 1.0e10;

    Eigen::Matrix<Real,3,1> boundary_fekete;
//    Eigen::Matrix<Real,4,1> boundary_fekete;
//    Eigen::Matrix<Real,5,1> boundary_fekete;
    boundary_fekete << -1., 0., 1.;
//    boundary_fekete << -1.,-0.447213595499957983,0.447213595499957928,1.;
//    boundary_fekete <<-1.,-0.654653670707977198,0.,0.654653670707977198,1.;

//    points = (points.array()/1000.).eval().matrix();
//    print(elements);
//    print(points);
//    print(edges);
//    cout << points.block(0,0,100,2) << endl;


//    const char *projection_method = "Newton";
    const char *projection_method = "Bisection";



//    exit (EXIT_FAILURE);
    to_python_structs struct_to_python;
    struct_to_python = PyCppInterface(iges_filename,scale,points.data(),points.rows(), points.cols(),
                           elements.data(), elements.rows(), elements.cols(),
                           edges.data(), edges.rows(), edges.cols(),
                           faces.data(),  faces.rows(),  faces.cols(),condition,
                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols(),
                           projection_method);


//    cout << unique_edges << endl;
//    cout << struct_to_python.displacement_BC_stl << endl;


    end = std::chrono::system_clock::now();
    std::chrono::duration<Real> elapsed_secs = end-start;
    std::cout << std::endl << "Total time elapsed was " << elapsed_secs.count() << " seconds" << std::endl;

    //exit (EXIT_FAILURE);
    return 0;
}
