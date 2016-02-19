from cython import boundscheck, wraparound
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "findin.cpp":
    vector[long] FindEqual(const long int *arr, long int size, long int num)
    vector[vector[long]] get_indices_cpp(long int *inv, long int *unique_inv, long int size_inv, long int size_unique_inv)

@boundscheck(False)
@wraparound(False)
cdef makezero(np.ndarray[np.float64_t, ndim=2, mode="c"] A, np.float64_t tol=1.0e-14):
    cdef int i,j
    cdef int a1 = A.shape[0]
    cdef int a2 = A.shape[1]
    for i in range(a1):
        for j in range(a2):
            if np.abs(A[i,j]) < tol:
                A[i,j] = 0

    return A 

cdef get_indices(long int *inv, long int *unique_inv, long int size_inv, long int size_unique_inv):

    cdef:
        long int i
        int j
        vector[long] vec, vec_1, vec_2


    for i in range(size_unique_inv):
        vec = FindEqual(inv,size_inv,unique_inv[i])
        if vec.size() > 1:
            for j in range(vec.size()-1):
                vec_1.push_back(vec[0])
            for j in range(1,vec.size()):
                vec_2.push_back(vec[j])

    return vec_1, vec_2


cdef get_indices_no_pushback(long int *inv, long int *unique_inv, long int size_inv, long int size_unique_inv):

    cdef:
        long int i, counter_1, counter_2
        int j
        vector[long] vec, vec_1, vec_2

    vec_1.resize(size_inv)
    vec_2.resize(size_inv)

    counter_1 = 0
    counter_2 = 0
    for i in range(size_unique_inv):
        vec = FindEqual(inv,size_inv,unique_inv[i])
        if vec.size() > 1:
            for j in range(vec.size()-1):
                vec_1[counter_1] = vec[0]
                counter_1 +=1
            for j in range(1,vec.size()):
                vec_2[counter_2] = vec[j]
                counter_2 +=1

    vec_1.resize(counter_1)
    vec_2.resize(counter_2)

    return vec_1, vec_2





def get_duplicates(np.ndarray[np.float64_t, ndim=2, mode="c"] A, int Decimals, Parallel=True):
    """Computes numpy array of duplicate rows"""
    A = makezero(A.copy())
    a = np.round(A,decimals=Decimals)
    # a = makezero(A.copy())
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    cdef np.ndarray[long int, ndim=1, mode="c"] inv = np.unique(b,return_inverse=True)[1]
    cdef np.ndarray[long int, ndim=1, mode="c"] unique_inv = np.unique(inv)

    if Parallel:
        return get_indices_cpp(&inv[0],&unique_inv[0], inv.shape[0],unique_inv.shape[0])
    else:
        vec_1, vec_2 = get_indices(&inv[0],&unique_inv[0], inv.shape[0],unique_inv.shape[0])
        # vec_1, vec_2 = get_indices_no_pushback(&inv[0],&unique_inv[0], inv.shape[0],unique_inv.shape[0])
        return np.concatenate((vec_1,vec_2)).reshape(2,len(vec_1)).T.copy()


