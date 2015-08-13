
#include <PythonInterface.hpp>
#include <PostMeshCurve.hpp>


PassToPython  PyCppInterface(const char* iges_filename, Real scale, Real *points_array, const Integer points_rows, const Integer points_cols,
                      UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                      UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                      UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                      Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                      UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols, const char *projection_method)
{

//    UInteger dimension = points_cols;
//    std::string element_type = points_cols==2 ? "tri" : "tet";

//    PostMeshCurve occ_interface = PostMeshCurve(element_type,dimension);
//    PostMeshCurve *curvilinear_mesh = new PostMeshCurve();
    std::shared_ptr<PostMeshCurve> curvilinear_mesh = std::make_shared<PostMeshCurve>(PostMeshCurve());
    curvilinear_mesh->SetMeshElements(elements_array,element_rows,element_cols);
    curvilinear_mesh->SetMeshElements(elements_array,element_rows,element_cols);
    curvilinear_mesh->SetMeshPoints(points_array,points_rows,points_cols);
    curvilinear_mesh->SetMeshEdges(edges_array,edges_rows,edges_cols);
    curvilinear_mesh->SetMeshFaces(faces_array,faces_rows,faces_cols);
    curvilinear_mesh->SetScale(scale);
    curvilinear_mesh->SetCondition(condition);
    curvilinear_mesh->SetProjectionCriteria(criteria,criteria_rows,criteria_cols);
    curvilinear_mesh->ScaleMesh();

    curvilinear_mesh->InferInterpolationPolynomialDegree();
    curvilinear_mesh->SetFeketePoints(boundary_fekete,fekete_rows,fekete_cols);
    curvilinear_mesh->GetBoundaryPointsOrder();

    // READ THE GEOMETRY FROM THE IGES FILE
    curvilinear_mesh->ReadIGES(iges_filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    curvilinear_mesh->GetGeomVertices();
    curvilinear_mesh->GetGeomEdges();
    curvilinear_mesh->GetGeomFaces();

    curvilinear_mesh->GetGeomPointsOnCorrespondingEdges();

    // FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
    curvilinear_mesh->IdentifyCurvesContainingEdges();
    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    curvilinear_mesh->ProjectMeshOnCurve(projection_method);
    // FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    curvilinear_mesh->RepairDualProjectedParameters();
    //PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    curvilinear_mesh->MeshPointInversionCurve();
    // OBTAIN DIRICHLET DATA
    PassToPython struct_to_python = curvilinear_mesh->GetDirichletData();
//    PassToPython struct_to_python;


//    PostMeshCurve occ_interface = PostMeshCurve();
//    occ_interface.SetMeshElements(elements_array,element_rows,element_cols);
//    occ_interface.SetMeshPoints(points_array,points_rows,points_cols);
//    occ_interface.SetMeshEdges(edges_array,edges_rows,edges_cols);
//    occ_interface.SetMeshFaces(faces_array,faces_rows,faces_cols);
//    occ_interface.SetScale(scale);
//    occ_interface.SetCondition(condition);
//    occ_interface.SetProjectionCriteria(criteria,criteria_rows,criteria_cols);
//    occ_interface.ScaleMesh();

//    occ_interface.InferInterpolationPolynomialDegree();
//    occ_interface.SetFeketePoints(boundary_fekete,fekete_rows,fekete_cols);
//    occ_interface.GetBoundaryPointsOrder();

//    // READ THE GEOMETRY FROM THE IGES FILE
//    occ_interface.ReadIGES(iges_filename);

//    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
//    occ_interface.GetGeomVertices();
//    occ_interface.GetGeomEdges();
//    occ_interface.GetGeomFaces();

//    occ_interface.GetGeomPointsOnCorrespondingEdges();

//    // FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
//    occ_interface.IdentifyCurvesContainingEdges();
//    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
//    occ_interface.ProjectMeshOnCurve(projection_method);
//    // FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
//    occ_interface.RepairDualProjectedParameters();
//    //PERFORM POINT INVERTION FOR THE INTERIOR POINTS
//    occ_interface.MeshPointInversionCurve();
//    // OBTAIN DIRICHLET DATA
//    PassToPython struct_to_python = occ_interface.GetDirichletData();
////    PassToPython struct_to_python;


    return struct_to_python;
}
