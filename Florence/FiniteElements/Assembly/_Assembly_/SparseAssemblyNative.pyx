from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.stdint cimport int64_t, uint64_t

ctypedef int64_t Integer
ctypedef uint64_t UInteger
ctypedef double Real

ctypedef np.int32_t Int # for I, J storage indices
ctypedef np.float64_t Float


cdef extern from "SparseAssemblyNative.h":
    void SparseAssemblyNativeCSR_RecomputeDataIndex_(
        const double *coeff,
        int *indices,
        int *indptr,
        double *data,
        int elem,
        int nvar,
        int nodeperelem,
        const UInteger *elements,
        const Integer *sorter)

    void SparseAssemblyNativeCSR_(
            const double *coeff,
            const int *data_local_indices,
            const int *data_global_indices,
            int elem,
            int local_capacity,
            double *data
        )



@boundscheck(False)
@wraparound(False)
def SparseAssemblyNative(np.ndarray[long] i, np.ndarray[long] j,
    np.ndarray[double] coeff, np.ndarray[Int] I, np.ndarray[Int] J,
    np.ndarray[Float] V, Int elem, Int nvar, Int nodeperelem,
    np.ndarray[UInteger,ndim=2, mode='c'] elements):

    cdef long i_shape = i.shape[0]
    cdef long j_shape = j.shape[0]
    assert i_shape==j_shape

    SparseAssemblyNative_(&i[0],&j[0],&coeff[0],&I[0],&J[0],&V[0],
        elem,nvar,nodeperelem,&elements[0,0],i_shape,j_shape)




cdef void SparseAssemblyNative_(
    const long *i,
    const long *j,
    const double *coeff,
    Int *I,
    Int *J,
    Float *V,
    Int elem,
    Int nvar,
    Int nodeperelem,
    const UInteger *elements,
    long i_shape,
    long j_shape) nogil:

    cdef long iterator, counter, ncounter
    cdef Int ndof = nvar*nodeperelem
    cdef Int local_capacity = ndof*ndof

    cdef long *current_row_column = <long*>malloc(sizeof(long)*nvar*nodeperelem)
    cdef long *full_current_row = <long*>malloc(sizeof(long)*i_shape)
    cdef long *full_current_column = <long*>malloc(sizeof(long)*j_shape)

    cdef long const_elem_retriever
    for counter in range(nodeperelem):
        const_elem_retriever = nvar*elements[elem*nodeperelem+counter]
        for ncounter in range(nvar):
            current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter

    memcpy(full_current_row,i,i_shape*sizeof(long))
    memcpy(full_current_column,j,j_shape*sizeof(long))

    cdef long const_I_retriever
    for counter in range(ndof):
        const_I_retriever = current_row_column[counter]
        for iterator in range(ndof):
            full_current_row[counter*ndof+iterator]    = const_I_retriever
            full_current_column[counter*ndof+iterator] = current_row_column[iterator]


    # STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
    cdef long low, high
    low = local_capacity*elem
    high = local_capacity*(elem+1)

    cdef long incrementer = 0
    for counter in range(low,high):
        I[counter] = full_current_row[incrementer]
        J[counter] = full_current_column[incrementer]
        V[counter] = coeff[incrementer]

        incrementer += 1


    free(full_current_row)
    free(full_current_column)
    free(current_row_column)








@boundscheck(False)
def SparseAssemblyNativeCSR(
    np.ndarray[int] data_local_indices,
    np.ndarray[int] data_global_indices,
    np.ndarray[double] coeff,
    np.ndarray[double] data,
    int elem,
    int local_capacity):

    SparseAssemblyNativeCSR_(
            &coeff[0],
            &data_local_indices[0],
            &data_global_indices[0],
            elem,
            local_capacity,
            &data[0]
        )



@boundscheck(False)
def SparseAssemblyNativeCSR_RecomputeDataIndex(
    mesh,
    np.ndarray[double] coeff,
    np.ndarray[int] indices,
    np.ndarray[int] indptr,
    np.ndarray[double] data,
    int elem,
    int nvar):

    cdef int nodeperelem = mesh.elements.shape[1]
    # cdef np.ndarray[long,ndim=2, mode='c'] sorter = np.argsort(elements,axis=1)
    # cdef np.ndarray[unsigned long,ndim=2, mode='c'] to_c_elements = np.copy(elements)
    # to_c_elements.sort(axis=1)
    cdef np.ndarray[Integer,ndim=2, mode='c'] sorter = mesh.element_sorter
    cdef np.ndarray[UInteger,ndim=2, mode='c'] to_c_elements = mesh.sorted_elements

    SparseAssemblyNativeCSR_RecomputeDataIndex_(&coeff[0],&indices[0], &indptr[0],&data[0],
        elem,nvar,nodeperelem,&to_c_elements[0,0],&sorter[0,0])

