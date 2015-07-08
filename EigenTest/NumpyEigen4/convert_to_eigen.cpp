/*
    Takes numpy array data as C pointer and converts it to Eigen matrix
*/

#include <convert_to_eigen.h>



void convert_to_eigen (double* c_array, int rows, int cols) {

    // call eigen functions
    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> eigen_matrix = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> >(c_array,rows,cols);
//    Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> eigen_matrix(rows,cols);
//    memcpy(eigen_matrix.data(), c_array, sizeof(double)*rows*cols);

    eigen_matrix = (eigen_matrix.array() + 2.).matrix();

    //std::cout << eigen_matrix << std::endl;
    return ;
}
