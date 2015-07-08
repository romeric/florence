

#include <py_to_occ_frontend.hpp>
#include <occ_frontend.hpp>

void py_cpp_interface(double * points_array, int points_rows, int points_cols, int *elements_array, int element_rows, int element_cols)
{
    //Convert to Eigen matrix
    Eigen::MatrixXdr points_eigen_matrix = Eigen::Map<Eigen::MatrixXdr>(points_array,points_rows,points_cols);

//    Eigen::MatrixXdr eigen_matrix(rows,cols);
//       memcpy(eigen_matrix.data(), c_array, sizeof(double)*rows*cols);

    Eigen::MatrixXir elements_eigen_matrix = Eigen::Map<Eigen::MatrixXir>(elements_array,element_rows,element_cols);

//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> eigen_points = Eigen::Map<Eigen::Matrix<double,
//            Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(point_array,rows,cols);

//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> eigen_points = Eigen::Map<Eigen::Matrix<double,
//            Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(point_array,rows,cols);

    int ndim;
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

    OCC_FrontEnd occ_interface = OCC_FrontEnd(element_type,ndim);
    occ_interface.SetElements(elements_eigen_matrix);
    occ_interface.SetPoints(points_eigen_matrix);

    std::cout << occ_interface.mesh_elements << std::endl;


    return ;

}
