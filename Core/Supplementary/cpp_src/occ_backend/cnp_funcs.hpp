#ifndef CNP_FUNCS_H
#define CNP_FUNCS_H

#ifndef EIGEN_INC_HPP
#include <eigen_inc.hpp>
#endif

#ifndef OCC_INC_HPP
#include <occ_inc.hpp>
#endif

namespace Eigen {
/* RowMajor matrix */
typedef Eigen::Matrix<Standard_Real,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> MatrixXdr;
}

template<typename T> struct unique_container
{
    std::vector<T> uniques;
    std::vector<int> unique_positions;
};

// a list of numpy inspired functions, kept inside a namespace just for convenience
namespace cpp_numpy {

inline Eigen::MatrixXi arange(int a, int b)
{
    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixXi arange(int b=1)
{
    /* default arange starting from zero and ending at 1.
     * b is optional and a is always zero
     */
    int a = 0;
    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixXi arange(int &a, int &b)
{
    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T> Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> take(Eigen::Matrix<T,
                                                                                         Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> &arr,
                                                                                         Eigen::MatrixXi &arr_row, Eigen::MatrixXi &arr_col)
{
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> arr_reduced(arr_row.rows(),arr_col.rows());

    for (int i=0; i<arr_row.rows();i++)
    {
        for (int j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

template<typename T> inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> take(Eigen::Matrix<T,
                                                                                                Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> &arr,
                                                                                                Eigen::MatrixXi &arr_row, Eigen::MatrixXi &arr_col)
{
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> arr_reduced(arr_row.rows(),arr_col.rows());

    for (int i=0; i<arr_row.rows();i++)
    {
        for (int j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

inline Standard_Real length(Handle_Geom_Curve &curve, Standard_Real scale=0.001)
{
    // GET LENGTH OF THE CURVE
    GeomAdaptor_Curve current_curve(curve);
    Standard_Real curve_length = GCPnts_AbscissaPoint::Length(current_curve);
    // CHANGE THE SCALE TO 1. IF NEEDED
    return scale*curve_length;
}

inline void sort_rows(Eigen::MatrixXdr & arr)
{
    /* Sorts a 2D array row by row*/

    for (int i=0; i<arr.rows(); ++i)
    {
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T> inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ravel(Eigen::Matrix<T,
                                                                                                 Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> &arr)
{
    /* irrespective of the array format (C/fortran), ravels the array row by row. However note that this involves a copy */
    assert (arr.cols()!=0);

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ravel_arr(arr.rows()*arr.cols(),1);
    int counter =0;
    for (int i=0; i<arr.rows(); ++i)
    {
        for (int j=0; j<arr.cols(); ++j)
        {
            ravel_arr(counter) = arr(i,j);
            counter += 1;
        }
    }

    return ravel_arr;
}

//template<typename Derived, typename T> inline unique_container<T> unique(Eigen::MatrixBase<Derived> & arr, T tol=1e-12)
//{
//    /*Find unique values of a 1D array - TODO */

//    // FIRST SORT THE ARRAY
//    std::vector<int> _unique_positions;
//    std::vector<T> _uniques;
//    _unique_positions.clear(); _uniques.clear();

//    for (int counter=0; counter<arr.rows(); ++counter)
//    {
//        T current_item = arr(counter);
//        for (int i=0;i<arr.rows(); ++i)
//        {
//            if ((current_item-arr(i)) < tol )
//            {
//                _unique_positions.push_back(i);
//            }
//        }
//    }
//    unique_container<T> _unique;
//    _unique.unique_positions = _unique_positions;
//    _unique.uniques = _uniques;


//    return _unique;

//}

}
// end of namespace

#endif // CNP_FUNCS_H

