#ifndef CNP_FUNCS_HPP
#define CNP_FUNCS_HPP

#include <EIGEN_INC.hpp>
#include <OCC_INC.hpp>

template<typename T> struct unique_container
{
    std::vector<T> uniques;
    std::vector<Integer> unique_positions;
};

// A LIST OF NUMPY-LIKE FUNCTIONS
namespace cpp_numpy {

inline Eigen::MatrixI arange(Integer a, Integer b)
{
//    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,DYNAMIC,1,POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixI arange(Integer b=1)
{
    /* default arange starting from zero and ending at 1.
     * b is optional and a is always zero
     */
    Integer a = 0;
//    return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,DYNAMIC,1,POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}
inline Eigen::MatrixI arange(Integer &a, Integer &b)
{
    //return Eigen::VectorXi::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
    return Eigen::Matrix<Integer,DYNAMIC,1,POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T> Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> take(Eigen::Matrix<T,
                                                                                         DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr,
                                                                                         Eigen::MatrixI &arr_row, Eigen::MatrixI &arr_col)
{
    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> arr_reduced(arr_row.rows(),arr_col.rows());

    for (auto i=0; i<arr_row.rows();i++)
    {
        for (auto j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

template<typename T> Eigen::Matrix<T,DYNAMIC,DYNAMIC,F_Contiguous> take(Eigen::Matrix<T,
                                                                                         DYNAMIC,DYNAMIC,F_Contiguous> &arr,
                                                                                         Eigen::MatrixI &arr_row, Eigen::MatrixI &arr_col)
{
    Eigen::Matrix<T,DYNAMIC,DYNAMIC,F_Contiguous> arr_reduced(arr_row.rows(),arr_col.rows());

    for (auto i=0; i<arr_row.rows();i++)
    {
        for (auto j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

template<typename T> inline Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> take(Eigen::Matrix<T,
                                                                                                DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr,
                                                                                                Eigen::MatrixI &arr_idx)
{
    assert (arr_idx.rows()<=arr.rows());
    assert (arr_idx.cols()<=arr.cols());

    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> arr_reduced(arr_idx.rows(),arr_idx.cols());

    for (auto i=0; i<arr_idx.rows();i++)
    {
        for (auto j=0; j<arr_idx.cols();j++)
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

  // INITIALIZE ORIGINAL INDEX LOCATIONS
  std::vector<Integer> idx(v.size());
  std::iota(idx.begin(),idx.end(),0);
  // SORT INDICES BY COMPARING VALUES IN V USING LAMBDA FUNCTION
  std::sort(idx.begin(), idx.end(),[&v](Integer i1, Integer i2) {return v[i1] < v[i2];});

  return idx;
}

template<typename T>
inline void sort_rows(Eigen::MatrixBase<T> &arr)
{
    //! SORTS A 2D ARRAY ROW BY ROW
    for (auto i=0; i<arr.rows(); ++i)
    {
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T>
inline void sort_rows(Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> & arr,Eigen::MatrixI &indices)
{
    // SORTS A 2D ARRAY ROW BY ROW
    for (auto i=0; i<arr.rows(); ++i)
    {
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
inline void sort_back_rows(Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED>&arr,Eigen::MatrixI &idx)
{
    //! SORTS BACK THE ARRAY ROW-WISE TO ITS ORIGINAL SHAPE GIVEN THE SORT INDICES IDX. nO COPY INVOLVED
    assert (idx.rows()==arr.rows() && idx.cols()==arr.cols());

    for (auto i=0; i<arr.rows(); ++i)
    {
        Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> \
                current_row = Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED>::Zero(1,arr.cols());
        for (auto j=0; j<arr.cols(); ++j)
        {
            current_row(j) = arr(i,idx(i,j));
        }
        arr.block(i,0,1,arr.cols()) = current_row;
    }
}


template<typename T> inline Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> ravel(Eigen::Matrix<T,
                                                                                                 DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr)
{
    //! IRRESPECTIVE OF THE ARRAY CONTIGUOUSNESS (C/FORTRAN), RAVELS THE ARRAY ROW BY ROW. NOTE THAT THIS INVOLVES A COPY
    assert (arr.cols()!=0);

    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> ravel_arr(arr.rows()*arr.cols(),1);
    Integer counter =0;
    for (auto i=0; i<arr.rows(); ++i)
    {
        for (auto j=0; j<arr.cols(); ++j)
        {
            ravel_arr(counter) = arr(i,j);
            counter += 1;
        }
    }

    return ravel_arr;
}

template<typename T>
inline std::tuple<Eigen::MatrixUI,Eigen::MatrixUI > where_eq(Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr,
                                                                                      T num, Real tolerance=1e-14)
{
    std::vector<UInteger> idx_rows;
    std::vector<UInteger> idx_cols;
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

    return std::make_tuple( Eigen::Map<Eigen::MatrixUI>(idx_rows.data(),idx_rows.size(),1),
                            Eigen::Map<Eigen::MatrixUI>(idx_cols.data(),idx_cols.size(),1));
}


template<typename T>
inline Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> \
append(Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr, T num)
{
    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> arr_pushed(arr.rows()+1,1);
    arr_pushed.block(0,0,arr.rows(),1) = arr;
    arr_pushed(arr.rows()) = num;

    return arr_pushed;
}


}
// end of namespace

// A shorthanded version equivalent to "import numpy as np"
namespace cnp = cpp_numpy;

#endif // CNP_FUNCS_H

