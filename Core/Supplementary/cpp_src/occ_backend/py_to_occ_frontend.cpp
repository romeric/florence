

#include <py_to_occ_frontend.hpp>
#include <occ_frontend.hpp>

to_python_structs py_cpp_interface(Real *points_array, Integer points_rows, Integer points_cols,
                      Integer *elements_array, Integer element_rows, Integer element_cols,
                      Integer *edges_array, Integer edges_rows, Integer edges_cols,
                      Integer *faces_array, Integer faces_rows, Integer faces_cols,
                      Real *c_displacement_BC, Integer *nodes_dir_out, Integer &nodes_dir_out_size)
{
    //Convert to Eigen matrix
    Eigen::MatrixR points_eigen_matrix = Eigen::Map<Eigen::MatrixR>(points_array,points_rows,points_cols);

//    Eigen::MatrixR points_eigen_matrix(rows,cols);
//       memcpy(eigen_matrix.data(), points_array, sizeof(double)*rows*cols);

    Eigen::MatrixI elements_eigen_matrix = Eigen::Map<Eigen::MatrixI>(elements_array,element_rows,element_cols);

    Integer ndim;
    std::string element_type;
    if (points_eigen_matrix.cols()==2)
    {
        ndim = 2;
        element_type = "tri";
    }
    else if (points_eigen_matrix.rows()==3)
    {
        ndim = 3;
        element_type = "tet";
    }

    Eigen::MatrixI edges_eigen_matrix = Eigen::Map<Eigen::MatrixI>(edges_array,edges_rows,edges_cols);

    Eigen::MatrixI faces_eigen_matrix;

    if (ndim==3)
    {
        faces_eigen_matrix = Eigen::Map<Eigen::MatrixI>(faces_array,faces_rows,faces_cols);

    }

    OCC_FrontEnd occ_interface = OCC_FrontEnd(element_type,ndim);
    occ_interface.SetElements(elements_eigen_matrix);
    occ_interface.SetPoints(points_eigen_matrix);
    occ_interface.SetEdges(edges_eigen_matrix);
    occ_interface.SetFaces(faces_eigen_matrix);

//    std::cout << occ_interface.mesh_elements << std::endl<<endl;
//    std::cout << occ_interface.mesh_edges << std::endl<<endl;


    Standard_Real condition = 2000.;
//    Standard_Real condition = 2.;
    occ_interface.SetCondition(condition);
    occ_interface.mesh_points *= 1000.0;

    // READ THE GEOMETRY FROM THE IGES FILE
    //std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/RAE2822/rae2822.igs";
//    std::string filename = "/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.igs";
    std::string filename = "/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/Circle.igs";

//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Line.igs";
//    std::string filename = "/home/roman/Dropbox/OCC_Geometry_Checks/Sphere.igs";
    occ_interface.ReadIGES(filename);

    // EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
    occ_interface.GetGeomEdges();
    occ_interface.GetGeomFaces();


    // PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
    //std::string method = "Newton";
    std::string method = "bisection";
    occ_interface.ProjectMeshOnCurve(method);
    // FIX IAMGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
    occ_interface.RepairDualProjectedParameters();

    // PERFORM POINT INVERTION FOR THE INTERIOR POINTS
    occ_interface.MeshPointInversionCurve();

//    std::cout << occ_interface.displacements_BC << std::endl;

    c_displacement_BC = occ_interface.displacements_BC.data();
//    nodes_dir_out = occ_interface.nodes_dir.data();
    nodes_dir_out_size = occ_interface.nodes_dir.rows();


    nodes_dir_out = occ_interface.nodes_dir.data();


    to_python_structs struct_to_python;
    struct_to_python.nodes_dir_size = occ_interface.nodes_dir.rows();
    // COPYLESS CONVERSION FROM EIGEN TO C ARRAY
    Integer *c_array_nodes_dir;
    c_array_nodes_dir = occ_interface.nodes_dir.data();
    // COPYLESS CONVERSION FROM C ARRAY TO STL VECTOR
    struct_to_python.nodes_dir_out_stl.assign(c_array_nodes_dir,c_array_nodes_dir+nodes_dir_out_size);
    Real *c_array_displacement;
    c_array_displacement = occ_interface.displacements_BC.data();
    struct_to_python.displacement_BC_stl.assign(c_array_displacement,c_array_displacement+occ_interface.ndim*nodes_dir_out_size);

//    for (int i=0; i<48; ++i)
//    {
//        cout << struct_to_python.displacement_BC_stl[i] << endl;
//    }


    return struct_to_python;

}
