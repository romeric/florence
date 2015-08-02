#ifndef CNP_FUNCS_H
#define CNP_FUNCS_H

#ifndef EIGEN_INC_HPP
#include <EIGEN_INC.hpp>
#endif

#ifndef OCC_INC_HPP
#include <OCC_INC.hpp>
#endif

template<typename T> struct unique_container
{
    std::vector<T> uniques;
    std::vector<int> unique_positions;
};

// a list of numpy inspired functions, kept inside a namespace just for convenience
namespace cpp_numpy {

inline Eigen::MatrixI arange(int a, int b)
{
//    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,Eigen::Dynamic,1>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixI arange(int b=1)
{
    /* default arange starting from zero and ending at 1.
     * b is optional and a is always zero
     */
    int a = 0;
//    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,Eigen::Dynamic,1>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixI arange(int &a, int &b)
{
    //return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,Eigen::Dynamic,1>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T> Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> take(Eigen::Matrix<T,
                                                                                         Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> &arr,
                                                                                         Eigen::MatrixI &arr_row, Eigen::MatrixI &arr_col)
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
                                                                                                Eigen::MatrixI &arr_row, Eigen::MatrixI &arr_col)
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
                                                                                                Eigen::MatrixI &arr_idx)
{
    assert (arr_idx.rows()<=arr.rows());
    assert (arr_idx.cols()<=arr.cols());

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> arr_reduced(arr_idx.rows(),arr_idx.cols());

    for (int i=0; i<arr_idx.rows();i++)
    {
        for (int j=0; j<arr_idx.cols();j++)
        {
            arr_reduced(i,j) = arr(arr_idx(i),arr_idx(j));
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

template <typename T> std::vector<Integer> argsort(const std::vector<T> &v) {

  // initialize original index locations
  std::vector<Integer> idx(v.size());
  for (Integer i = 0; i != idx.size(); ++i) idx[i] = i;
  // sort indices based on comparing values in v
  std::sort(idx.begin(), idx.end(),[&v](Integer i1, Integer i2) {return v[i1] < v[i2];});

  return idx;
}

inline void sort_rows(Eigen::MatrixR & arr)
{
    /* Sorts a 2D array row by row*/

    for (Integer i=0; i<arr.rows(); ++i)
    {
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T> inline void sort_rows(Eigen::Matrix<T,-1,-1,1> & arr,Eigen::MatrixI &indices)
{
    /* Sorts a 2D array row by row*/

    for (Integer i=0; i<arr.rows(); ++i)
    {
//        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
        std::vector<Integer> row_indices;
        std::vector<T> row_arr;
        row_arr.assign(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
        row_indices = argsort(row_arr);
        indices.block(i,0,1,indices.cols()) = Eigen::Map<Eigen::MatrixI>(row_indices.data(),1,row_indices.size());

        // SORT THE ACTUAL ARRAY NOW
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T>
inline void sort_back_rows(Eigen::Matrix<T,-1,-1,1>&arr,Eigen::MatrixI &idx)
{
    /* Sorts back the array row-wise to its original shape given the sort indices idx. No copy involved */
    assert (idx.rows()==arr.rows() && idx.cols()==arr.cols());

    for (Integer i=0; i<arr.rows(); ++i)
    {
        Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> \
                current_row = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>::Zero(1,arr.cols());
        for (Integer j=0; j<arr.cols(); ++j)
        {
            current_row(j) = arr(i,idx(i,j));
        }
        arr.block(i,0,1,arr.cols()) = current_row;
    }
}


template<typename T> inline Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ravel(Eigen::Matrix<T,
                                                                                                 Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> &arr)
{
    /* irrespective of the array format (C/fortran), ravels the array row by row. However note that this involves a copy */
    assert (arr.cols()!=0);

    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> ravel_arr(arr.rows()*arr.cols(),1);
    int counter =0;
    for (Integer i=0; i<arr.rows(); ++i)
    {
        for (Integer j=0; j<arr.cols(); ++j)
        {
            ravel_arr(counter) = arr(i,j);
            counter += 1;
        }
    }

    return ravel_arr;
}

template<typename Derived>
inline std::tuple<Eigen::MatrixI,Eigen::MatrixI > where_eq(Eigen::Matrix<Derived,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> &arr,
                                                                                      Derived num, Real tolerance=1e-14)

//template<typename Derived>
//inline std::tuple<Eigen::MatrixI,Eigen::MatrixI > where_eq(Eigen::MatrixBase<Derived> &arr,Derived &num, Real tolerance=1e-14)
{
//    Eigen::MatrixBase<Derived> idx_rows;
//    Eigen::MatrixBase<Derived> idx_cols;
    std::vector<Integer> idx_rows;
    std::vector<Integer> idx_cols;
    idx_rows.clear(); idx_cols.clear();
    for (Integer i=0; i<arr.rows();++i)
    {
        for (Integer j=0; j<arr.cols();++j)
        {
            if (abs(arr(i,j)-num)<tolerance)
            {
                idx_rows.push_back(i);
                idx_cols.push_back(j);
            }
        }
    }

    return std::make_tuple( Eigen::Map<Eigen::MatrixI>(idx_rows.data(),idx_rows.size(),1),
                            Eigen::Map<Eigen::MatrixI>(idx_cols.data(),idx_cols.size(),1));
}


template<typename T>
Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> \
append(Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> &arr, T num)
{
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> arr_pushed(arr.rows()+1,1);
    arr_pushed.block(0,0,arr.rows(),1) = arr;
    arr_pushed(arr.rows()) = num;

    return arr_pushed;
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

