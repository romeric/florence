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

template<typename T>
STATIC ALWAYS_INLINE
Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> arange(T a, T b)
{
    //! EQUIVALENT TO NUMPY UNIQUE. GET A LINEARLY SPACED VECTOR
    return Eigen::Matrix<T,DYNAMIC,1,
            POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T>
STATIC ALWAYS_INLINE
Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED>
arange(T b=1)
{
    //! EQUIVALENT TO NUMPY UNIQUE. GET A LINEARLY SPACED VECTOR
    //! DEFAULT ARANGE STARTING FROM ZERO AND ENDING AT 1.
    //! b IS OPTIONAL AND A IS ALWAYS ZERO

    Integer a = 0;
    return Eigen::Matrix<T,DYNAMIC,1,
            POSTMESH_ALIGNED>::LinSpaced(Eigen::Sequential,(b-a),a,b-1);
}

template<typename T, typename U>
Eigen::PlainObjectBase<T>
STATIC take(const Eigen::PlainObjectBase<T> &arr, const Eigen::PlainObjectBase<U> &arr_row, const Eigen::PlainObjectBase<U> &arr_col)
{
    //! TAKE OUT PART OF A 2D ARRAY. MAKES A COPY
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
STATIC take(const Eigen::PlainObjectBase<T> &arr, const Eigen::MatrixI &arr_idx)
{
    //! TAKE OUT PART OF A 2D ARRAY. MAKES A COPY
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

template<typename T, typename U>
void STATIC put(Eigen::PlainObjectBase<T> &arr_to_put, const Eigen::PlainObjectBase<T> &arr_to_take,
                const Eigen::PlainObjectBase<U> &arr_row, const Eigen::PlainObjectBase<U> &arr_col)
{
    //! PUT A SUBARRAY INTO AN ARRAY. THIS IS IN-PLACE OPERATION
    assert(arr_to_put.rows()==arr_to_take.rows()
           && arr_to_put.cols()==arr_to_take.cols()
           && "ARRAY_TO_PUT_AND_ARRAY_TO_TAKE_VALUES_FROM_SHOULD_HAVE_THE_SAME_VALUES");

    for (auto i=0; i<arr_row.rows();i++)
    {
        for (auto j=0; j<arr_col.rows();j++)
        {
            arr_to_put(arr_row(i),arr_col(j)) = arr_to_take(arr_row(i),arr_col(j));
        }
    }
}

template<typename T, typename U>
void STATIC put(Eigen::PlainObjectBase<T> &arr_to_put, typename Eigen::PlainObjectBase<T>::Scalar value,
                const Eigen::PlainObjectBase<U> &arr_row, const Eigen::PlainObjectBase<U> &arr_col)
{
    //! PUT A VALUE INTO PART OF AN ARRAY. THIS IS IN-PLACE OPERATION
    for (auto i=0; i<arr_row.rows();i++)
    {
        for (auto j=0; j<arr_col.rows();j++)
        {
            arr_to_put(arr_row(i),arr_col(j)) = value;
        }
    }
}

STATIC ALWAYS_INLINE Real length(const Handle_Geom_Curve &curve, Standard_Real scale=0.001)
{
    //! GET LENGTH OF A GEOMETRICAL CURVE
    GeomAdaptor_Curve current_curve(curve);
    Real curve_length = GCPnts_AbscissaPoint::Length(current_curve);
    // CHANGE THE SCALE TO 1. IF NEEDED
    return scale*curve_length;
}

template <typename T>
STATIC std::vector<Integer> argsort(const std::vector<T> &v)
{
    //! GET INDICES OF A SORTED VECTOR
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
STATIC void sort_rows(Eigen::PlainObjectBase<T> &arr, Eigen::MatrixI &idx)
{
    //! SORTS A 2D ARRAY ROW BY ROW
    for (auto i=0; i<arr.rows(); ++i)
    {
        std::vector<Integer> row_indices;
        std::vector<typename Eigen::PlainObjectBase<T>::Scalar> row_arr;
        row_arr.assign(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
        row_indices = argsort(row_arr);
        idx.block(i,0,1,idx.cols()) = \
                Eigen::Map<Eigen::MatrixI>(row_indices.data(),1,row_indices.size());
        // SORT THE ACTUAL ARRAY NOW
        std::sort(arr.row(i).data(),arr.row(i).data()+arr.row(i).size());
    }
}

template<typename T>
STATIC void sort_back_rows(Eigen::PlainObjectBase<T> &arr, const Eigen::MatrixI &idx)
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
STATIC where_eq(const Eigen::PlainObjectBase<T> &arr,
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
append(const Eigen::PlainObjectBase<T> &arr, U num)
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

template<typename T>
STATIC std::vector<std::vector<T> >
toSTL(const Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> &arr)
{
    //! CONVERT EIGEN MATRIX TO STL VECTOR OF VECTORS.
    //! IS STRICTLY VALID FOR MATRICES

    std::vector<std::vector<T> > arr_stl(arr.rows());
    for (auto i=0; i < arr.rows(); ++i)
    {
        std::vector<Integer> current_row(arr.cols());
        for (auto j=0; j < arr.cols(); ++j)
        {
            current_row[j] = arr(i,j);
        }
        arr_stl[i] = current_row;
    }
    return arr_stl;
}

template<typename T>
STATIC Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED>
toEigen(const std::vector<std::vector<T> > &arr_stl)
{
    //! CONVERT STL VECTOR OF VECTORS TO EIGEN MATRIX.
    //! ALL VECTORS SHOULD HAVE THE SAME LENGTH (STRUCTURED)

    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> arr(arr_stl.size(),arr_stl[0].size());
    for (UInteger i=0; i < arr_stl.size(); ++i)
    {
        for (UInteger j=0; j < arr_stl[0].size(); ++j)
        {
            arr(i,j) = arr_stl[i][j];
        }
    }
    return arr;
}

template <typename T>
STATIC std::vector<T> intersect(const std::vector<T>& vec1, const std::vector<T>& vec2)
{
    std::vector<T> commons;
    for (auto &iter1: vec1)
    {
        for (auto &iter2: vec2)
        {
            if (iter1==iter2)
            {
                commons.push_back(iter1);
                break;
            }
        }
    }

    return commons;
}

template <typename T, typename ... Rest>
STATIC std::vector<T> intersect(const std::vector<T>& vec1, const std::vector<T>& vec2, const std::vector<Rest>& ... rest)
{
    auto commons = intersect(vec1,vec2);
    commons = intersect(commons,rest...);
    return commons;
}

template<typename T>
STATIC std::tuple<std::vector<typename Eigen::PlainObjectBase<T>::Scalar>,std::vector<Integer> >
unique(const Eigen::PlainObjectBase<T> &arr)
{
    //! RETURNS UNIQUE VALUES AND UNIQUE INDICES OF AN EIGEN MATRIX
    assert(arr.cols()==1 && "UNIQUE_METHOD_IS_ONLY_AVAILABLE_FOR_1D_ARRAYS/MATRICES");
    std::vector<typename Eigen::PlainObjectBase<T>::Scalar> uniques;
    std::vector<Integer> idx;

    for (auto i=0; i<arr.rows(); ++i) {
        bool isunique = true;
        for (auto j=0; j<=i; ++j) {
            if (arr(i)==arr(j) && i!=j) {
                isunique = false;
                break;
            }
        }

        if (isunique==true) {
            uniques.push_back(arr(i));
            idx.push_back(i);
        }
    }

    // SORT UNIQUE VALUES
    auto sorter = argsort(uniques);
    std::sort(uniques.begin(),uniques.end());
    std::vector<Integer> idx_sorted(idx.size());
    for (UInteger i=0; i<uniques.size();++i) {
        idx_sorted[i] = idx[sorter[i]];
    }

    std::tuple<std::vector<typename Eigen::PlainObjectBase<T>::Scalar>,std::vector<Integer> >
            uniques_idx = std::make_tuple(uniques,idx_sorted);

    return uniques_idx;
}

template<typename T>
STATIC std::tuple<std::vector<T>,std::vector<Integer> > 
unique(const std::vector<T> &v, bool return_index=false) 
{
    //! RETURNS UNIQUE VALUES AND UNIQUE INDICES OF A STD::VECTOR
    if (return_index == false) {
        std::vector<T> uniques(v.begin(),v.end());
        std::sort(uniques.begin(),uniques.end());
        uniques.erase(std::unique(uniques.begin(),uniques.end()),uniques.end());

        return std::make_tuple(uniques,std::vector<Integer>(0));
    }

    auto sorter = argsort(v);
    auto last = std::unique(sorter.begin(),sorter.end(),[&v](T a, T b){return v[a]==v[b];});
    sorter.erase(last,sorter.end());

    std::vector<T> uniques(sorter.size());
    auto counter = 0;
    for (auto &k: sorter) {
        uniques[counter] = v[k];
        counter++;
    }

    return std::make_tuple(uniques,sorter);
}

template<typename T>
Eigen::PlainObjectBase<T> itemfreq(const Eigen::PlainObjectBase<T> &arr)
{
    //! FINDS THE NUMBER OF OCCURENCE OF EACH VALUE IN AN EIGEN MATRIX
    std::vector<typename Eigen::PlainObjectBase<T>::Scalar> uniques;
    std::tie(uniques,std::ignore) = unique(arr);
    Eigen::PlainObjectBase<T> freqs;
    freqs.setZero(uniques.size(),2);

    auto counter = 0;
    for (auto &i: uniques)
    {
        Integer counts = std::count(arr.data(),arr.data()+arr.rows(),i);
        freqs(counter,0) = i;
        freqs(counter,1) = counts;
        counter++;
    }

    return freqs;
}

template<typename T>
Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> itemfreq(const std::vector<T> &arr)
{
    //! FINDS THE NUMBER OF OCCURENCE OF EACH VALUE IN A VECTOR
    std::vector<T> uniques;
    std::tie(uniques,std::ignore) = unique(arr,false);
    Eigen::Matrix<T,DYNAMIC,DYNAMIC,POSTMESH_ALIGNED> freqs(uniques.size(),2);

    auto counter = 0;
    for (auto &i: uniques)
    {
        Integer counts = std::count(arr.begin(),arr.end(),i);
        freqs(counter,0) = i;
        freqs(counter,1) = counts;
        counter++;
    }

    return freqs;
}


}
// end of namespace

// SHORTEN THE NAMESPACE
namespace cnp = cpp_numpy;

#endif // CNP_FUNCS_H

