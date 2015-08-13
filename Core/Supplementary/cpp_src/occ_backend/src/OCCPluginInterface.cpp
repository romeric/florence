
#include <OCCPluginInterface.hpp>
#include <OCCPlugin.hpp>


PassToPython  PyCppInterface(const char* iges_filename, Real scale, Real *points_array, const Integer points_rows, const Integer points_cols,
                      UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                      UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                      UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                      Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                      UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols, const char *projection_method)
{
    //Convert to Eigen matrix
//    Eigen::MatrixR points_eigen_matrix = Eigen::Map<Eigen::MatrixR>(points_array,points_rows,points_cols);
    //Eigen::MatrixR points_eigen_matrix(rows,cols); memcpy(eigen_matrix.data(), points_array, sizeof(double)*rows*cols);
//    Eigen::MatrixI elements_eigen_matrix = Eigen::Map<Eigen::MatrixI>(elements_array,element_rows,element_cols);

//    Eigen::Map<Eigen::MatrixR> points_eigen_matrix(points_array,points_rows,points_cols);
//    Eigen::Map<Eigen::MatrixI> elements_eigen_matrix(elements_array,element_rows,element_cols);

//    Eigen::MatrixR points_eigen_matrix;
//    Eigen::MatrixI elements_eigen_matrix;
//    new (&points_eigen_matrix) Eigen::Map<Eigen::MatrixR> (points_array,points_rows,points_cols);
//    new (&elements_eigen_matrix) Eigen::Map<Eigen::MatrixI> (elements_array,element_rows,element_cols);
//    print(points_array,points_eigen_matrix.data());

//    std::free(points_eigen_matrix.data());
//    std::free(elements_eigen_matrix.data());
//    delete [] points_eigen_matrix.data();
//    delete [] elements_eigen_matrix.data();
//    points_eigen_matrix.resize(0,0);
//    elements_eigen_matrix.resize(0,0);

    UInteger dimension = points_cols;
    std::string element_type = points_cols==2 ? "tri" : "tet";

    OCCPlugin occ_interface = OCCPlugin(element_type,dimension);
    occ_interface.SetMeshElements(elements_array,element_rows,element_cols);
    occ_interface.SetMeshPoints(points_array,points_rows,points_cols);
    occ_interface.SetMeshEdges(edges_array,edges_rows,edges_cols);
    occ_interface.SetMeshFaces(faces_array,faces_rows,faces_cols);
    occ_interface.SetScale(scale);
    occ_interface.SetCondition(condition);
    occ_interface.SetProjectionCriteria(criteria,criteria_rows,criteria_cols);
    occ_interface.ScaleMesh();

    occ_interface.InferInterpolationPolynomialDegree();
    occ_interface.SetFeketePoints(boundary_fekete,fekete_rows,fekete_cols);
    occ_interface.GetBoundaryPointsOrder();

    // READ THE GEOMETRY FROM THE IGES FILE
    occ_interface.ReadIGES(iges_filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    occ_interface.GetGeomVertices();
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
    // OBTAIN DIRICHLET DATA
    PassToPython struct_to_python = occ_interface.GetDirichletData();
//    PassToPython struct_to_python;


    return struct_to_python;
}
