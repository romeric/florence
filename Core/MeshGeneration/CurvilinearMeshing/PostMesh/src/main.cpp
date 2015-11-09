//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <PyInterfaceEmulator.hpp>

using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;


Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &edges, Eigen::MatrixR &points, Real condition)
{
   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(edges.rows(),edges.cols());
   for (auto i=0; i<edges.rows();++i)
   {
       Real x1 = points(edges(i,0),0);
       Real y1 = points(edges(i,0),1);
       Real x2 = points(edges(i,1),0);
       Real y2 = points(edges(i,1),1);

       if ((std::sqrt(x1*x1+y1*y1)<condition) && ( std::sqrt(x2*x2+y2*y2)< condition))
       {
           criteria(i) = 1;
       }
   }
  return criteria;
}

Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &edges, Eigen::MatrixR &points)
{
   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(edges.rows(),edges.cols());
   for (auto i=0; i<edges.rows();++i)
   {
       Real x1 = points(edges(i,0),0);
       Real y1 = points(edges(i,0),1);
//       Real x2 = points(edges(i,1),0);
//       Real y2 = points(edges(i,1),1);

       if ( (x1 > -500) && (x1 < 500) && (y1 < -0.6) && (y1 > 0.6)  ) {
           criteria(i) = 1;
       }
   }
  return criteria;
}



int main()
{

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
//    std::string unique_edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/elements_circle_p4.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/points_circle_p4.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/edges_circle_p4.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_half_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_half_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_half_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_twoarcs_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_twoarcs_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_twoarcs_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/elements_rae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/points_rae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/edges_rae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/elements_check_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/points_check_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/edges_check_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/elements_irae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/points_irae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/edges_irae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/elements_i2rae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/points_i2rae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/edges_i2rae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_mech2d_seg0_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_mech2d_seg0_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_mech2d_seg0_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_mech2d_seg2_p2.dat"; //#
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_mech2d_seg2_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_mech2d_seg2_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/MechanicalComponent2D/elements_mech2dn_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/MechanicalComponent2D/points_mech2dn_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/MechanicalComponent2D/edges_mech2dn_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_leftpartwithcircle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_leftpartwithcircle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_leftpartwithcircle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/elements_leftcircle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/points_leftcircle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/edges_leftcircle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/elements_wing2d_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/points_wing2d_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/edges_wing2d_p2.dat";

    // 3D
//    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/elements_sphere_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/points_sphere_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/edges_sphere_p2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/faces_sphere_p2.dat";

    std::string elem_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/elements_sphere2_p2.dat";
    std::string point_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/points_sphere2_p2.dat";
    std::string edge_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/edges_sphere2_p2.dat";
    std::string face_file = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Tests/faces_sphere2_p2.dat";



//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/rae2822.igs"; //
//    const char* iges_filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/Mech2D_Seg0.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/LeftPartWithCircle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/LeftCircle.iges";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/Mech2D_Seg2.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Annular_Circle/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/Half_Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Misc/Two_Arcs.iges";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/sd7003.igs";

//    const char* iges_filename = "/home/roman/Dropbox/zdump/OCC_Geometry_Checks/Cylinder.igs";
    const char* iges_filename = "/home/roman/Dropbox/zdump/OCC_Geometry_Checks/Sphere.igs";




    // CALL PY_PostMeshCurve
    Eigen::MatrixUI elements = PostMeshCurve::ReadI(elem_file,',');
    Eigen::MatrixR points = PostMeshCurve::ReadR(point_file,',');
    Eigen::MatrixUI edges = PostMeshCurve::ReadI(edge_file,',');
    Eigen::MatrixUI faces = Eigen::MatrixUI::Zero(1,4); // 2D
//    Eigen::MatrixUI faces = PostMeshCurve::ReadI(face_file,','); // 3D
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
//    Real scale = 1.;
//    Real condition = 1.0e10;

    // anisotropic rae2822
    Real scale = 1.;
    Real condition = 5;

    //3D sphere
//    Real scale = 1000.;
//    Real condition = 1.0e20;

    Eigen::Matrix<Real,3,1> boundary_fekete;
//    Eigen::Matrix<Real,4,1> boundary_fekete;
//    Eigen::Matrix<Real,5,1> boundary_fekete;
    boundary_fekete << -1., 0., 1.;
//    boundary_fekete << -1.,-0.447213595499957983,0.447213595499957928,1.;
//    boundary_fekete <<-1.,-0.654653670707977198,0.,0.654653670707977198,1.;

    // anistropic rae2822
//    points = (points.array()/1000.).eval().matrix();
//    points.col(0) = (points.col(0).array()-0.5).eval().matrix();
//    print (points(89,0),points(89,1));

//    points = (points.array()/100000.).eval().matrix();
//    print(elements);
//    print(points);
//    print(edges);
//    print(faces);
//    print(faces.rows(),faces.cols());
//    print(points.row(92));
//    print (edges.rows(),elements.rows(),elements.cols());
//    cout << points.block(0,0,100,2) << endl;
//    println(points.minCoeff(),points.maxCoeff());

    auto criteria = ComputeCriteria(edges,points,condition);

//    Real precision = 1.0e-02;
    // 3D
    auto precision = 1.0e-04;


//    exit (EXIT_FAILURE);
    PassToPython struct_to_python;
//    struct_to_python = ComputeDirichleteData(iges_filename,scale,points.data(),points.rows(), points.cols(),
//                           elements.data(), elements.rows(), elements.cols(),
//                           edges.data(), edges.rows(), edges.cols(),
//                           faces.data(),  faces.rows(),  faces.cols(),condition,
//                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols(),
//                           criteria.data(), criteria.rows(), criteria.cols(), precision);

    struct_to_python = ComputeDirichleteData3D(iges_filename,scale,points.data(),points.rows(), points.cols(),
                           elements.data(), elements.rows(), elements.cols(),
                           edges.data(), edges.rows(), edges.cols(),
                           faces.data(),  faces.rows(),  faces.cols(),condition,
                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols(),
                           criteria.data(), criteria.rows(), criteria.cols(), precision);

//    print(struct_to_python.displacement_BC_stl);
//    print(elements);

    end = std::chrono::system_clock::now();
    std::chrono::duration<Real> elapsed_secs = end-start;
    std::cout << std::endl << "Total time elapsed was " << elapsed_secs.count() << " seconds" << std::endl;

    //exit (EXIT_FAILURE);
    return 0;
}