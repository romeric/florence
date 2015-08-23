
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



    PostMeshCurve xx;
    xx.SetMeshElements(elements_array,element_rows,element_cols);
    xx.SetMeshElements(elements_array,element_rows,element_cols);
    xx.SetMeshPoints(points_array,points_rows,points_cols);
    xx.SetMeshEdges(edges_array,edges_rows,edges_cols);
    xx.SetMeshFaces(faces_array,faces_rows,faces_cols);
    xx.SetScale(scale);
    xx.SetCondition(condition);
    xx.SetProjectionPrecision(precision);
    xx.SetProjectionCriteria(criteria,criteria_rows,criteria_cols);
    xx.ScaleMesh();
    xx.InferInterpolationPolynomialDegree();
    xx.SetFeketePoints(boundary_fekete,fekete_rows,fekete_cols);
    xx.GetBoundaryPointsOrder();
    xx.ReadIGES(iges_filename);
    xx.GetGeomVertices();
    xx.GetGeomEdges();
    xx.GetGeomFaces();
//    print(xx.ndim,xx.scale,xx.condition);
//    print(xx.mesh_elements);
//    PostMeshCurve yy(std::move(xx));
    PostMeshCurve yy(xx);
    print(yy.mesh_elements);
    print(xx.mesh_elements);
    PostMeshCurve zz(std::move(xx));
    print("is anything printed in front of this",xx.mesh_elements);
    print("there should be something in front of this",zz.mesh_elements);
//    print(yy.geometry_points[1].X());
//    print(xx.geometry_points[1].X());

    Eigen::MatrixR m; m.setRandom(2,2);
    Eigen::MatrixR m2 = std::move(m);
//    Eigen::MatrixR m2 = m;
    print(m);
    print(m2);

    std::vector<int> ss  = {1,2,3};
    std::vector<int> tt = ss;
    print(ss); print(tt);
    std::vector<int> ww = std::move(ss);
    print(ss); print(ww);

    cout << endl;
    PostMeshSurface b1 = PostMeshSurface();
    PostMeshSurface b2 = b1;
    PostMeshSurface b3;
    b3 = b1;
    b2 = std::move(b3);


    return struct_to_python;
}
