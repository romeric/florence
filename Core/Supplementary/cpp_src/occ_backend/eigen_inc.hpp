#ifndef EIGEN_INC_HPP
#define EIGEN_INC_HPP


#ifndef STD_INC_HPP
#include <std_inc.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>

namespace Eigen {
/* RowMajor matrix */
typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixR;
typedef Eigen::Matrix<Integer,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixI;
}


#endif // EIGEN_INC_HPP

