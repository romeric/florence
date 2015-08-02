
#include <OCCPluginInterface.hpp>
#include <OCCPlugin.hpp>


to_python_structs PyCppInterface(const char* iges_filename, Real scale, Real *points_array, Integer points_rows, Integer points_cols,
                      Integer *elements_array, Integer element_rows, Integer element_cols,
                      Integer *edges_array, Integer edges_rows, Integer edges_cols,
                      Integer *faces_array, Integer faces_rows, Integer faces_cols, Real condition,
                      Real *boundary_fekete, Integer fekete_rows, Integer fekete_cols, const char* projection_method)
{
    //Convert to Eigen matrix
//    Eigen::MatrixR points_eigen_matrix = Eigen::Map<Eigen::MatrixR>(points_array,points_rows,points_cols);
    //Eigen::MatrixR points_eigen_matrix(rows,cols); memcpy(eigen_matrix.data(), points_array, sizeof(double)*rows*cols);
//    Eigen::MatrixI elements_eigen_matrix = Eigen::Map<Eigen::MatrixI>(elements_array,element_rows,element_cols);

    Eigen::Map<Eigen::MatrixR> points_eigen_matrix(points_array,points_rows,points_cols);
    Eigen::Map<Eigen::MatrixI> elements_eigen_matrix(elements_array,element_rows,element_cols);

//    Eigen::MatrixR points_eigen_matrix;
//    Eigen::MatrixI elements_eigen_matrix;
//    new (&points_eigen_matrix) Eigen::Map<Eigen::MatrixR> (points_array,points_rows,points_cols);
//    new (&elements_eigen_matrix) Eigen::Map<Eigen::MatrixI> (elements_array,element_rows,element_cols);
//    print(points_array,points_eigen_matrix.data());


//    Integer ndim;
//    std::string element_type;
//    if (points_eigen_matrix.cols()==2)
//    {
//        ndim = 2;
//        element_type = "tri";
//    }
//    else if (points_eigen_matrix.rows()==3)
//    {
//        ndim = 3;
//        element_type = "tet";
//    }

    Integer dimension = points_eigen_matrix.cols();
    std::string element_type = points_eigen_matrix.cols()==2 ? "tri" : "tet";

    Eigen::Map<Eigen::MatrixI> edges_eigen_matrix(edges_array,edges_rows,edges_cols);
    Eigen::Map<Eigen::MatrixI> faces_eigen_matrix(faces_array,faces_rows,faces_cols);

    Eigen::Map<Eigen::MatrixR> eigen_boundary_fekete(boundary_fekete,fekete_rows,fekete_cols);

    OCCPlugin occ_interface = OCCPlugin(element_type,dimension);
    occ_interface.SetMeshElements(elements_eigen_matrix);
    occ_interface.SetMeshPoints(points_eigen_matrix);
    occ_interface.SetMeshEdges(edges_eigen_matrix);
    occ_interface.SetMeshFaces(faces_eigen_matrix);
    occ_interface.SetScale(scale);
    occ_interface.SetCondition(condition);
    occ_interface.ScaleMesh();

    occ_interface.InferInterpolationPolynomialDegree();
    occ_interface.SetFeketePoints(eigen_boundary_fekete);
    occ_interface.GetBoundaryPointsOrder();

    // READ THE GEOMETRY FROM THE IGES FILE
    occ_interface.ReadIGES(iges_filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    //occ_interface.GetGeomVertices();
    occ_interface.GetGeomEdges();
    occ_interface.GetGeomFaces();

    occ_interface.GetGeomPointsOnCorrespondingEdges();

    // FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
    occ_interface.IdentifyCurvesContainingEdges();
    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    occ_interface.ProjectMeshOnCurve(projection_method);
    // FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    occ_interface.RepairDualProjectedParameters();
    //PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    occ_interface.MeshPointInversionCurve();


    to_python_structs struct_to_python;
    struct_to_python.nodes_dir_size = occ_interface.nodes_dir.rows();
    // CONVERT FROM EIGEN TO STL VECTOR
    struct_to_python.nodes_dir_out_stl.assign(occ_interface.nodes_dir.data(),occ_interface.nodes_dir.data()+struct_to_python.nodes_dir_size);
    struct_to_python.displacement_BC_stl.assign(occ_interface.displacements_BC.data(),occ_interface.displacements_BC.data()+ \
                                                occ_interface.ndim*struct_to_python.nodes_dir_size);

//    std::free(points_eigen_matrix.data());
//    std::free(elements_eigen_matrix.data());
//    delete [] points_eigen_matrix.data();
//    delete [] elements_eigen_matrix.data();
//    points_eigen_matrix.resize(0,0);
//    elements_eigen_matrix.resize(0,0);

    return struct_to_python;

}
