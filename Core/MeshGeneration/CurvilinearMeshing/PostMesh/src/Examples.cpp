
#include <PyInterfaceEmulator.hpp>
#include <PostMeshCurve.hpp>

#include <PostMeshSurface.hpp>


DirichletData ComputeDirichleteData(const char* iges_filename, Real scale, Real *points_array, const Integer points_rows, const Integer points_cols,
                      UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                      UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                      UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                      Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                      UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols, const Real &precision)
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
    curvilinear_mesh->SetNodalSpacing(boundary_fekete,fekete_rows,fekete_cols);
    curvilinear_mesh->GetBoundaryPointsOrder();

    // READ THE GEOMETRY FROM THE IGES FILE
    curvilinear_mesh->ReadIGES(iges_filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    curvilinear_mesh->GetGeomVertices();
    curvilinear_mesh->GetGeomEdges();
//    curvilinear_mesh->GetGeomFaces();

    curvilinear_mesh->GetGeomPointsOnCorrespondingEdges();

    // FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
    curvilinear_mesh->IdentifyCurvesContainingEdges();
    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    curvilinear_mesh->ProjectMeshOnCurve();
    // FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    curvilinear_mesh->RepairDualProjectedParameters();
    //PERFORM POINT INVERSION FOR THE INTERIOR POINTS
    curvilinear_mesh->MeshPointInversionCurveArcLength();
    // OBTAIN DIRICHLET DATA
    DirichletData Dirichlet_data = curvilinear_mesh->GetDirichletData();
//    DirichletData Dirichlet_data;


    return Dirichlet_data;
}



DirichletData ComputeDirichleteData3D(const char* iges_filename, Real scale, Real *points_array, const Integer points_rows, const Integer points_cols,
                      UInteger *elements_array, const Integer element_rows, const Integer element_cols,
                      UInteger *edges_array, const Integer &edges_rows, const Integer &edges_cols,
                      UInteger *faces_array, const Integer &faces_rows, const Integer &faces_cols, Real condition,
                      Real *boundary_fekete, const Integer fekete_rows, const Integer fekete_cols,
                      UInteger *criteria, const Integer criteria_rows, const Integer criteria_cols, const Real &precision)
{

//    UInteger dimension = points_cols;
//    std::string element_type = points_cols==2 ? "tri" : "tet";

//    PostMeshCurve occ_interface = PostMeshCurve(element_type,dimension);
//    PostMeshCurve *curvilinear_mesh = new PostMeshCurve();

    std::shared_ptr<PostMeshSurface> curvilinear_mesh = std::make_shared<PostMeshSurface>(PostMeshSurface());
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
    curvilinear_mesh->SetNodalSpacing(boundary_fekete,fekete_rows,fekete_cols);
//    curvilinear_mesh->GetBoundaryPointsOrder();

    // READ THE GEOMETRY FROM THE IGES FILE
    curvilinear_mesh->ReadIGES(iges_filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    curvilinear_mesh->GetGeomVertices();
    curvilinear_mesh->GetGeomEdges();
    curvilinear_mesh->GetGeomFaces();

//    curvilinear_mesh->GetSurfacesParameters();
    curvilinear_mesh->GetGeomPointsOnCorrespondingFaces();

    // FIRST IDENTIFY WHICH SURFACES CONTAIN WHICH FACES
//    curvilinear_mesh->IdentifySurfacesContainingFaces();
//    curvilinear_mesh->IdentifySurfacesContainingFacesByProjection();
    curvilinear_mesh->IdentifySurfacesContainingFacesByPureProjection();

//    std::string filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/form1_face_to_surface_P2.dat";
//    auto face_to_surface = PostMeshSurface::Read(filename);
//    print(face_to_surface);
//    curvilinear_mesh->SupplySurfacesContainingFaces(face_to_surface.data(),face_to_surface.rows());
//    print(curvilinear_mesh->)
    // IDENTIFY WHICH EDGES ARE SHARED BETWEEN SURFACES
    curvilinear_mesh->IdentifySurfacesIntersections();
    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE SURFACE
    curvilinear_mesh->ProjectMeshOnSurface();
//    curvilinear_mesh->RepairDualProjectedParameters();
    // FIX ACTUAL MESH
    curvilinear_mesh->ReturnModifiedMeshPoints(points_array);
//    //PERFORM POINT INVERSION FOR THE INTERIOR POINTS
//    curvilinear_mesh->MeshPointInversionSurface(0);

    // Read FE bases
    std::string bases_file = "/home/roman/Dropbox/neval.dat";
    Eigen::MatrixR FEbases = PostMeshBase::ReadR(bases_file,','); // 3D
    auto OrthTol=0.5;
    curvilinear_mesh->MeshPointInversionSurfaceArcLength(0,OrthTol,FEbases.data(),FEbases.rows(),FEbases.cols());
//    // OBTAIN DIRICHLET DATA
    DirichletData Dirichlet_data = curvilinear_mesh->GetDirichletData();
//    DirichletData Dirichlet_data;
//    print(curvilinear_mesh->mesh_elements);



    return Dirichlet_data;
}
