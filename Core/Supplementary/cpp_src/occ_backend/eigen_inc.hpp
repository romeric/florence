#ifndef EIGEN_INC_HPP
#define EIGEN_INC_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>

namespace Eigen {
/* RowMajor matrix */
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXdr;
typedef Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXir;
}


#endif // EIGEN_INC_HPP

