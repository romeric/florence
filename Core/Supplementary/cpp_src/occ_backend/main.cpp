//#define NDEBUG
//#define EIGEN_NO_DEBUG

#include <assert.h>
#include <stdlib.h>

#include <occ_frontend.hpp>

using namespace std;
// A syntactic sugar equivalent to "import to numpy as np"
namespace cnp = cpp_numpy;



int main()
{

    clock_t begin = clock();


    string elem_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle_p2.dat";
    string point_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle_p2.dat";
    string edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/edges_circle_p2.dat";
    string unique_edge_file = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/unique_edges_circle_p2.dat";


    std::string element_type = "tri";
    Integer ndim = 2;
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

//    cout << MainDataCpp.geometry_edges.size() << " " << MainDataCpp.geometry_faces.size() << endl;

    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    //std::string method = "Newton";
    std::string method = "bisection";
    MainDataCpp.ProjectMeshOnCurve(method);
    // CONVERT CURVES TO BSPLINE CURVES
    MainDataCpp.CurvesToBsplineCurves();
    // FIX IAMGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    //MainDataCpp.RepairDualProjectedParameters_Old();
    MainDataCpp.RepairDualProjectedParameters();

    // PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    MainDataCpp.MeshPointInversionCurve();

//    Eigen::Matrix<int64_t,Eigen::Dynamic,Eigen::Dynamic> xx;



    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << endl << "Total time elapsed was " << elapsed_secs << " seconds" << endl;

    return 0;
}
