#ifndef EIGEN_INC_HPP
#define EIGEN_INC_HPP


#ifndef STL_INC_HPP
#define STL_INC_HPP
#include <STL_INC.hpp>
#endif

#define EIGEN_VECTORIZE
#define EIGEN_DEFAULT_TO_ROW_MAJOR

#include <Eigen/Core>
#include <Eigen/Dense>
#include<Eigen/StdVector>

// DEFINE STORAGE ORDERS
#define C_Contiguous Eigen::RowMajor
#define F_Contiguous Eigen::ColMajor
// GENERIC ALIGNED
#define POSTMESH_ALIGNED C_Contiguous


// DYNAMIC MATRICES
#define DYNAMIC Eigen::Dynamic

namespace Eigen {
// DEFINE Real, Integer AND UInteger BASED MATRICES
typedef Eigen::Matrix<Real,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> MatrixR;
typedef Eigen::Matrix<Integer,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> MatrixI;
typedef Eigen::Matrix<UInteger,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> MatrixUI;
}


#endif // EIGEN_INC_HPP

