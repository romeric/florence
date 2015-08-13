#ifndef EIGEN_INC_HPP
#define EIGEN_INC_HPP


#ifndef STL_INC_HPP
#define STL_INC_HPP
#include <STL_INC.hpp>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>

#ifdef EIGEN_VECTORIZE
    #define EIGEN_VECTORIZE
#endif

namespace Eigen {
/* RowMajor matrix */
typedef Eigen::Matrix<Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixR;
typedef Eigen::Matrix<Integer,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixI;
typedef Eigen::Matrix<UInteger,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixUI;
}


#endif // EIGEN_INC_HPP

