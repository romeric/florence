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

    clock_t begin = clock();


    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
//    std::string unique_edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";

//    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs";
//    const char* iges_filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
    const char* iges_filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";

    // CALL PY_OCC_FRONTEND
    Eigen::MatrixI elements = OCC_FrontEnd::ReadI(elem_file,',');
    Eigen::MatrixR points = OCC_FrontEnd::ReadR(point_file,',');
    Eigen::MatrixI edges = OCC_FrontEnd::ReadI(edge_file,',');
    Eigen::MatrixI faces = Eigen::MatrixI::Zero(1,4);
//    Eigen::MatrixI unique_edges = Read(unique_edge_file);
    Real scale = 1000.;
    Real condition = 2000.;

    Eigen::Matrix<Real,3,1> boundary_fekete;
    boundary_fekete << -1., 0., 1.;
//    boundary_fekete << -1.,-0.447213595499957983,0.447213595499957928,1.;
//    boundary_fekete <<-1.,-0.654653670707977198,0.,0.654653670707977198,1.;

//    cout << edges2 << endl;

//    Integer out[24];


    to_python_structs struct_to_python;
    struct_to_python = PyCppInterface(iges_filename,scale,points.data(),points.rows(), points.cols(),
                           elements.data(), elements.rows(), elements.cols(),
                           edges.data(), edges.rows(), edges.cols(),
                           faces.data(),  faces.rows(),  faces.cols(),condition,
                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols());


//    cout << unique_edges << endl;
//    cout << struct_to_python.displacement_BC_stl << endl;


    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "Total time elapsed was " << elapsed_secs << " seconds" << endl;

    //exit (EXIT_FAILURE);
    return 0;
}
