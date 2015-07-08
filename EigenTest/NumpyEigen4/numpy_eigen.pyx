"""
multiply.pyx

simple cython test of accessing a numpy array's data

the C function: c_multiply multiplies all the values in a 2-d array by a scalar, in place.

"""

import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern from "convert_to_eigen.h": 
    void convert_to_eigen (double* array, int m, int n)

@cython.boundscheck(False)
@cython.wraparound(False)
def pass_to_eigen(np.ndarray[double, ndim=2, mode="c"] arr not None):
    """
    convert_to_eigen (arr)

    Takes a numpy arry as input, and multiplies each elemetn by value, in place

    param: array -- a 2-d numpy array of np.float64
    param: value -- a number that will be multiplied by each element in the array

    """
    cdef int m, n

    m, n = arr.shape[0], arr.shape[1]

    convert_to_eigen (&arr[0,0], m, n)

    return None

