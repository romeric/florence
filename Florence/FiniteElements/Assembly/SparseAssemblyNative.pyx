from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np 
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


ctypedef np.int32_t Int # for I, J storage indices
ctypedef np.float64_t Float

# The following fused typedefs are provided for
# cases when assembly (LHS) matrix is lowered to 32 bit, but their
# performance is terrible. Recompile instead for 32bit LHSes 
# ctypedef fused Int:
#     np.int32_t
#     np.int64_t

# ctypedef fused Float:
#     np.float32_t
#     np.float64_t

@boundscheck(False)
@wraparound(False)
def SparseAssemblyNative(np.ndarray[long] i, np.ndarray[long] j,
    np.ndarray[double] coeff, np.ndarray[Int] I, np.ndarray[Int] J,
    np.ndarray[Float] V, Int elem, Int nvar, Int nodeperelem,
    np.ndarray[unsigned long,ndim=2, mode='c'] elements):

    cdef long i_shape = i.shape[0]
    cdef long j_shape = j.shape[0]
    assert i_shape==j_shape

    SparseAssemblyNative_(&i[0],&j[0],&coeff[0],&I[0],&J[0],&V[0],
        elem,nvar,nodeperelem,&elements[0,0],i_shape,j_shape)




cdef void SparseAssemblyNative_(const long *i, const long *j, const double *coeff, Int *I, Int *J,
    Float *V, Int elem, Int nvar, Int nodeperelem, const unsigned long *elements,long i_shape, long j_shape) nogil:

    cdef long *current_row_column = <long*>malloc(sizeof(long)*nvar*nodeperelem)
    cdef long *full_current_row = <long*>malloc(sizeof(long)*i_shape)
    cdef long *full_current_column = <long*>malloc(sizeof(long)*j_shape)

    cdef long iterator, counter, ncounter
    # sqrt of local capacity to be precise 
    cdef Int local_capacity = nvar*nodeperelem 

    cdef long const_elem_retriever 
    for counter in range(nodeperelem):
        const_elem_retriever = nvar*elements[elem*nodeperelem+counter]
        for ncounter in range(nvar):
            current_row_column[nvar*counter+ncounter] = const_elem_retriever+ncounter

    memcpy(full_current_row,i,i_shape*sizeof(long))
    memcpy(full_current_column,j,j_shape*sizeof(long))

    cdef long const_I_retriever 
    for counter in range(local_capacity):
        const_I_retriever = current_row_column[counter]
        for iterator in range(local_capacity):
            full_current_row[counter*local_capacity+iterator]    = const_I_retriever
            full_current_column[counter*local_capacity+iterator] = current_row_column[iterator]


    # STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
    cdef long low, high
    low = local_capacity*local_capacity*elem
    high = local_capacity*local_capacity*(elem+1)

    cdef long incrementer = 0
    for counter in range(low,high):
        I[counter] = full_current_row[incrementer]
        J[counter] = full_current_column[incrementer]
        V[counter] = coeff[incrementer]

        incrementer += 1


    free(full_current_row)
    free(full_current_column)
    free(current_row_column)
