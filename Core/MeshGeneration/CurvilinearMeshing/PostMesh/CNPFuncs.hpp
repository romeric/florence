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

STATIC ALWAYS_INLINE Eigen::MatrixI arange(Integer a, Integer b)
{
    return Eigen::Matrix<Integer,DYNAMIC,1,
            POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

STATIC ALWAYS_INLINE Eigen::MatrixI arange(Integer b=1)
{
    /* DEFAULT ARANGE STARTING FROM ZERO AND ENDING AT 1.
     * b IS OPTIONAL AND A IS ALWAYS ZERO
     */
    Integer a = 0;
    return Eigen::Matrix<Integer,DYNAMIC,1,
            POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

STATIC ALWAYS_INLINE Eigen::MatrixI arange(Integer &a, Integer &b)
{
    return Eigen::Matrix<Integer,DYNAMIC,1,
            POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T>
Eigen::PlainObjectBase<T>
STATIC take(Eigen::PlainObjectBase<T> &arr, Eigen::MatrixI &arr_row, Eigen::MatrixI &arr_col)
{
    Eigen::PlainObjectBase<T> arr_reduced;
    arr_reduced.setZero(arr_row.rows(),arr_col.rows());

    for (auto i=0; i<arr_row.rows();i++)
    {
        for (auto j=0; j<arr_col.rows();j++)
        {
            arr_reduced(i,j) = arr(arr_row(i),arr_col(j));
        }
    }

    return arr_reduced;
}

template<typename T>
Eigen::PlainObjectBase<T>
STATIC take(Eigen::PlainObjectBase<T> &arr, Eigen::MatrixI &arr_idx)
{
    assert (arr_idx.rows()<=arr.rows());
    assert (arr_idx.cols()<=arr.cols());

    Eigen::PlainObjectBase<T> arr_reduced;
    arr_reduced.setZero(arr_idx.rows(),arr_idx.cols());

    for (auto i=0; i<arr_idx.rows();i++)
    {
        for (auto j=0; j<arr_idx.cols();j++)
        {
            arr_reduced(i,j) = arr(arr_idx(i),arr_idx(j));
        }
    }

    return arr_reduced;
}

STATIC ALWAYS_INLINE Real length(Handle_Geom_Curve &curve, Standard_Real scale=0.001)
{
    // GET LENGTH OF THE CURVE
    GeomAdaptor_Curve current_curve(curve);
    Real curve_length = GCPnts_AbscissaPoint::Length(current_curve);
    // CHANGE THE SCALE TO 1. IF NEEDED
    return scale*curve_length;
}

template <typename T>
STATIC std::vector<Integer> argsort(const std::vector<T> &v) {

  // INITIALIZE ORIGINAL INDEX LOCATIONS
  std::vector<Integer> idx(v.size());
  std::iota(idx.begin(),idx.end(),0);
  // SORT INDICES BY COMPARING VALUES IN V USING LAMBDA FUNCTION
  std::sort(idx.begin(), idx.end(),[&v](Integer i1, Integer i2) {return v[i1] < v[i2];});

  return idx;
}

template<typename T>
STATIC ALWAYS_INLINE void sort_rows(Eigen::MatrixBase<T> &arr)
{
    //! SORTS A 2D ARRAY ROW BY ROW
    for (auto i=0; i<arr.rows(); ++i)
    {
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T>
STATIC void sort_rows(Eigen::PlainObjectBase<T> & arr,Eigen::MatrixI &indices)
{
    //! SORTS A 2D ARRAY ROW BY ROW
    for (auto i=0; i<arr.rows(); ++i)
    {
        std::vector<Integer> row_indices;
        std::vector<typename Eigen::PlainObjectBase<T>::Scalar> row_arr;
        row_arr.assign(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
        row_indices = argsort(row_arr);
        indices.block(i,0,1,indices.cols()) = \
                Eigen::Map<Eigen::MatrixI>(row_indices.data(),1,row_indices.size());
        // SORT THE ACTUAL ARRAY NOW
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T>
STATIC void sort_back_rows(Eigen::PlainObjectBase<T>&arr,Eigen::MatrixI &idx)
{
    //! SORTS BACK THE ARRAY ROW-WISE TO ITS ORIGINAL SHAPE GIVEN THE SORT INDICES IDX.
    //! NO COPY INVOLVED
    assert (idx.rows()==arr.rows() && idx.cols()==arr.cols());

    for (auto i=0; i<arr.rows(); ++i)
    {
        Eigen::PlainObjectBase<T> current_row;
        current_row.setZero(1,arr.cols());
        for (auto j=0; j<arr.cols(); ++j)
        {
            current_row(j) = arr(i,idx(i,j));
        }
        arr.block(i,0,1,arr.cols()) = current_row;
    }
}

template<typename T>
STATIC ALWAYS_INLINE Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED>
ravel(Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr)
{
    //! RAVEL/FLATTEN THE ARRAY RESPECTING DATA CONTIGUOUSNESS. MAKES A COPY
    return Eigen::Map<Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> >
            (arr.data(),arr.rows()*arr.cols(),1);
}

template<typename T, typename U = T>
std::tuple<Eigen::MatrixUI,Eigen::MatrixUI >
STATIC where_eq(Eigen::PlainObjectBase<T> &arr,
         U num, Real tolerance=1e-14)
{
    //! FIND THE OCCURENCES OF VALUE IN A MATRIX
    std::vector<UInteger> idx_rows;
    std::vector<UInteger> idx_cols;
    idx_rows.clear(); idx_cols.clear();
    for (Integer i=0; i<arr.rows();++i)
    {
        for (Integer j=0; j<arr.cols();++j)
        {
            if (static_cast<Real>(abs(arr(i,j)-num))<tolerance)
            {
                idx_rows.push_back(i);
                idx_cols.push_back(j);
            }
        }
    }

    return std::make_tuple(
                Eigen::Map<Eigen::MatrixUI>
                (idx_rows.data(),idx_rows.size(),1),
                Eigen::Map<Eigen::MatrixUI>
                (idx_cols.data(),idx_cols.size(),1));
}

template<typename T, typename U = T>
STATIC ALWAYS_INLINE Eigen::PlainObjectBase<T>
append(Eigen::PlainObjectBase<T> &arr, U num)
{
    //! APPEND TO AN EIGEN VECTOR, SIMILAR TO PUSH_BACK. MAKES A COPY
    assert(arr.cols()==1 && "YOU CANNOT APPEND TO MULTI-DIMENSIONAL MATRICES. "
                            "APPEND IS STRICTLY FOR MATRICES WITH COLS()==1");

    Eigen::PlainObjectBase<T> new_arr;
    new_arr.setZero(arr.rows()+1,1);
    new_arr.block(0,0,arr.rows(),1) = arr;
    new_arr(arr.rows()) = static_cast<typename Eigen::PlainObjectBase<T>::Scalar>(num);

    return new_arr;
}

template<typename T = Integer>
std::tuple<std::vector<T>,std::vector<size_t> > unique(std::vector<T> &arr) {

    std::vector<T> uniques;
    std::vector<UInteger> idx;

    for (auto i=0; i<arr.size(); ++i) {
        bool isunique = true;
        for (auto j=0; j<=i; ++j) {
            if (arr[i]==arr[j] && i!=j) {
                isunique = false;
                break;
            }
        }

        if (isunique==true) {
            uniques.push_back(arr[i]);
            idx.push_back(i);
        }
    }

    // SORT UNIQUE VALUES
    auto sorter = argsort(uniques);
    std::sort(uniques.begin(),uniques.end());
    std::vector<UInteger> idx_sorted(idx.size());
    for (auto i=0; i<uniques.size();++i) {
        idx_sorted[i] = idx[sorter[i]];
    }

    std::tuple<std::vector<T>,std::vector<size_t> >
            uniques_idx = std::make_tuple(uniques,idx_sorted);

    return uniques_idx;
}


}
// end of namespace

// A shorthanded version equivalent to "import numpy as np"
namespace cnp = cpp_numpy;

#endif // CNP_FUNCS_H

