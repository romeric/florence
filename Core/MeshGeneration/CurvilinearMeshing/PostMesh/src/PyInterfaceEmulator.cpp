
#include <PyInterfaceEmulator.hpp>
#include <PostMeshCurve.hpp>

#include <PostMeshSurface.hpp>


PassToPython  ComputeDirichleteData(const char* iges_filename, Real scale, Real *points_array, const Integer points_rows, const Integer points_cols,
                      UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                      UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                      UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                      Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                      UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols,
                      const char *projection_method, const Real &precision)
{

//    UInteger dimension = points_cols;
//    std::string element_type = points_cols==2 ? "tri" : "tet";

//    PostMeshCurve occ_interface = PostMeshCurve(element_type,dimension);
//    PostMeshCurve *curvilinear_mesh = new PostMeshCurve();
//    PostMeshCurve check;
    std::shared_ptr<PostMeshCurve> curvilinear_mesh = std::make_shared<PostMeshCurve>(PostMeshCurve());
    curvilinear_mesh->SetMeshElements(elements_array,element_rows,element_cols);
    curvilinear_mesh->SetMeshPoints(points_array,points_rows,points_cols);
    curvilinear_mesh->SetMeshEdges(edges_array,edges_rows,edges_cols);
    curvilinear_mesh->SetMeshFaces(faces_array,faces_rows,faces_cols);
    curvilinear_mesh->SetScale(scale);
    curvilinear_mesh->SetCondition(condition);
    curvilinear_mesh->SetProjectionPrecision(precision);
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


    return struct_to_python;
}
