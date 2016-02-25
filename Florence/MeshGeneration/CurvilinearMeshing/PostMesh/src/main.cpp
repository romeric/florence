//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <PyInterfaceEmulator.hpp>

using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;


//Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &edges, Eigen::MatrixR &points, Real condition)
//{
//   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(edges.rows(),edges.cols());
//   for (auto i=0; i<edges.rows();++i)
//   {
//       Real x1 = points(edges(i,0),0);
//       Real y1 = points(edges(i,0),1);
//       Real x2 = points(edges(i,1),0);
//       Real y2 = points(edges(i,1),1);

//       if ((std::sqrt(x1*x1+y1*y1)<condition) && ( std::sqrt(x2*x2+y2*y2)< condition))
//       {
//           criteria(i) = 1;
//       }
//   }
//  return criteria;
//}

//Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &edges, Eigen::MatrixR &points)
//{
//   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(edges.rows(),edges.cols());
//   for (auto i=0; i<edges.rows();++i)
//   {
//       Real x1 = points(edges(i,0),0);
//       Real y1 = points(edges(i,0),1);
////       Real x2 = points(edges(i,1),0);
////       Real y2 = points(edges(i,1),1);

//       if ( (x1 > -500) && (x1 < 500) && (y1 < -0.6) && (y1 > 0.6)  ) {
//           criteria(i) = 1;
//       }
//   }
//  return criteria;
//}

//3D
Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &faces, Eigen::MatrixR &points, Real condition)
{
   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(faces.rows(),faces.cols());
   for (auto i=0; i<faces.rows();++i)
   {
       Real x1 = points(faces(i,0),0);
       Real y1 = points(faces(i,0),1);
       Real z1 = points(faces(i,0),2);

       Real x2 = points(faces(i,1),0);
       Real y2 = points(faces(i,1),1);
       Real z2 = points(faces(i,1),2);

       Real x3 = points(faces(i,2),0);
       Real y3 = points(faces(i,2),1);
       Real z3 = points(faces(i,2),2);

       if ((std::sqrt(x1*x1+y1*y1+z1*z1)<condition) && ( std::sqrt(x2*x2+y2*y2+z2*z2)< condition) && ( std::sqrt(x3*x3+y3*y3+z3*z3)< condition))
       {
           criteria(i) = 1;
       }
   }
  return criteria;
}


Eigen::MatrixUI ComputeCriteria(Eigen::MatrixUI &faces, Eigen::MatrixR &points)
{
   Eigen::MatrixUI criteria = Eigen::MatrixUI::Zero(faces.rows(),1);
   auto num = static_cast<Real>(faces.cols());
   for (auto i=0; i<faces.rows();++i)
   {
       auto current_row = faces.row(i);
       auto x = 0.;
       auto y = 0.;
       auto z = 0.;
       for (auto j=0; j<current_row.cols(); j++)
       {
           x += points(current_row(j),0);
           y += points(current_row(j),1);
           z += points(current_row(j),2);
       }
       x /= num;
       y /= num;
       z /= num;

//       if ( (x > -2.5) && (x < 2.5) && (y > -2.) && (y < 2.0) && (z > -2.) && (z < 2.) ) {
//           criteria(i) = 1;
//       }
       if ( x > -10. && x < 30. && y > -20. && y < 20. && z > -15. && z < 15.) {
           criteria(i) = 1;
       }
   }
  return criteria;
}



int main()
{

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
//    std::string unique_edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/elements_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/points_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/edges_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/elements_circle_p3.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/points_circle_p3.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/edges_circle_p3.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/elements_circle_p4.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/points_circle_p4.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/edges_circle_p4.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_half_circle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_half_circle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_half_circle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_twoarcs_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_twoarcs_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_twoarcs_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/elements_rae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/points_rae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/edges_rae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/elements_check_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/points_check_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/RAE2822/edges_check_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/elements_irae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/points_irae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/edges_irae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/elements_i2rae2822_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/points_i2rae2822_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Python/Examples/FiniteElements/RAE2822/edges_i2rae2822_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_mech2d_seg0_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_mech2d_seg0_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_mech2d_seg0_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_mech2d_seg2_p2.dat"; //#
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_mech2d_seg2_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_mech2d_seg2_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent2D/elements_mech2dn_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent2D/points_mech2dn_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent2D/edges_mech2dn_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_leftpartwithcircle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_leftpartwithcircle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_leftpartwithcircle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/elements_leftcircle_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/points_leftcircle_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/edges_leftcircle_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/elements_wing2d_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/points_wing2d_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/edges_wing2d_p2.dat";

    // 3D
//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/elements_sphere_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/points_sphere_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/edges_sphere_p2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/faces_sphere_p2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/elements_sphere2_p2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/points_sphere2_p2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/edges_sphere2_p2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/faces_sphere2_p2.dat";

    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Sphere/Sphere_elements_P2.dat";
    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Sphere/Sphere_points_P2.dat";
    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Sphere/Sphere_edges_P2.dat";
    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Sphere/Sphere_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H1_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H1_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H1_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H1_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2262_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2262_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2262_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2262_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2_1247_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2_1247_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2_1247_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2_1247_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/FullTPipe_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/FullTPipe_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/FullTPipe_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/FullTPipe_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/Hollow_Cylinder_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/Hollow_Cylinder_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/Hollow_Cylinder_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/Hollow_Cylinder_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1_faces_P2.dat";

//    std::string elem_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso_P2_elements_P2.dat";
//    std::string point_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso_P2_points_P2.dat";
//    std::string edge_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso_P2_edges_P2.dat";
//    std::string face_file = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso_P2_faces_P2.dat";


//    const char* iges_filename = "/home/roman/Dropbox/Florence/Problems/FiniteElements/RAE2822/rae2822.igs"; //
//    const char* iges_filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/Mech2D_Seg0.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/LeftPartWithCircle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/LeftCircle.iges";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/Mech2D_Seg2.igs";
//    const char* iges_filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle_Nurbs/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Annular_Circle/Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/Half_Circle.igs";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Misc/Two_Arcs.iges";
//    const char* iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/sd7003.igs";

    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Sphere/Sphere.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/PipeTShape.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/TPipe_2.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/FullTPipe.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/Hollow_Cylinder.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1.igs";
//    std::string iges_filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon.igs";


    // CALL PY_PostMeshCurve
    Eigen::MatrixUI elements = PostMeshBase::ReadI(elem_file,',');
    Eigen::MatrixR points = PostMeshBase::ReadR(point_file,',');

//    Eigen::MatrixUI edges = PostMeshCurve::ReadI(edge_file,','); // 2D
//    Eigen::MatrixUI faces = Eigen::MatrixUI::Zero(1,4); // 2D

    Eigen::MatrixUI edges = PostMeshBase::ReadI(edge_file,','); // 3D
    Eigen::MatrixUI faces = PostMeshBase::ReadI(face_file,','); // 3D


//    Real scale = 1000.;  // for annular circle nurbs
//    Real condition = 2000.; // for annular circle nurbs
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
//    Real scale = 1.;
//    Real condition = 5;

    //3D sphere
    Real scale = 1000.;
    Real condition = 1.0e10;

    // almond and falcon
//    Real scale = 25.4;
//    Real condition = 1.0e20;

    // TPipe
//    Real scale = 1000.;
//    Real condition = 1.0e20;

//    Real scale = 25.4;
//    Real condition = 1.0e20;

//    Eigen::Matrix<Real,3,1> boundary_fekete;
//    Eigen::Matrix<Real,4,1> boundary_fekete;
//    Eigen::Matrix<Real,5,1> boundary_fekete;
//    boundary_fekete << -1., 0., 1.;
//    boundary_fekete << -1.,-0.447213595499957983,0.447213595499957928,1.;
//    boundary_fekete <<-1.,-0.654653670707977198,0.,0.654653670707977198,1.;

    Eigen::Matrix<Real,6,2> boundary_fekete;
    boundary_fekete << -1., -1.,
                        1., -1.,
                       -1.,  1.,
                        0., -1.,
                       -1.,  0.,
                        0.,  0.;
//    print(boundary_fekete);

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
//    print(boundary_fekete);

//    auto criteria = ComputeCriteria(edges,points,condition);
//    auto criteria = ComputeCriteria(faces,points); // 3D
    auto criteria = ComputeCriteria(faces,points,condition); // 3D
//    print(criteria);

//    Real precision = 1.0e-02;
    // 3D
    auto precision = 1.0e-04;


//    exit (EXIT_FAILURE);
    DirichletData Dirichlet_data;
//    Dirichlet_data = ComputeDirichleteData(iges_filename.c_str(),scale,points.data(),points.rows(), points.cols(),
//                           elements.data(), elements.rows(), elements.cols(),
//                           edges.data(), edges.rows(), edges.cols(),
//                           faces.data(),  faces.rows(),  faces.cols(),condition,
//                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols(),
//                           criteria.data(), criteria.rows(), criteria.cols(), precision);

    Dirichlet_data = ComputeDirichleteData3D(iges_filename.c_str(),scale,points.data(),points.rows(), points.cols(),
                           elements.data(), elements.rows(), elements.cols(),
                           edges.data(), edges.rows(), edges.cols(),
                           faces.data(),  faces.rows(),  faces.cols(),condition,
                           boundary_fekete.data(), boundary_fekete.rows(), boundary_fekete.cols(),
                           criteria.data(), criteria.rows(), criteria.cols(), precision);

//    print(Dirichlet_data.displacement_BC_stl);
//    print(elements);

    end = std::chrono::system_clock::now();
    std::chrono::duration<Real> elapsed_secs = end-start;
    std::cout << std::endl << "Total time elapsed was " << elapsed_secs.count() << " seconds" << std::endl;

    //exit (EXIT_FAILURE);
    return 0;
}


//-------------------------------------------------------------
////    print(unique_surface_flags_ext);
//    Eigen::MatrixI map_int_ext = Eigen::MatrixI::Zero(unique_surface_flags_ext.size(),2);
//    for (UInteger i=0; i<this->geometry_points_on_surfaces.size(); ++i)
//    {
//        for (UInteger j=0; j<unique_surface_flags_ext.size(); ++j)
//        {
//            if (this->geometry_points_on_surfaces[i].rows()!=this->geometry_points_on_surfaces[j].rows())
//            {
//                continue;
//            }
//            else
//            {
//                if ((this->geometry_points_on_surfaces[i] - this->geometry_points_on_surfaces[j]).isMuchSmallerThan(1e-05))
//                {
//                    map_int_ext(i,0) = i;
//                    map_int_ext(i,1) = j;
//                }
////                print(this->geometry_points_on_surfaces[i] - points);
//            }
////            print(points);
//        }
//    }
//    print(map_int_ext);
////    print(counter);
//    exit(0);
//-------------------------------------------------------------
