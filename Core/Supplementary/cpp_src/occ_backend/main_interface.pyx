"""
main interface between python and cpp's occ_frontend 
"""

import cython

import numpy as np
cimport numpy as np

cdef extern from "py_to_occ_frontend.hpp": 
    # void convert_to_eigen (double* array, int m, int n)
    void py_cpp_interface (double* points_array, int points_rows, int points_cols, int* elements_array, int element_rows, int element_cols)

@cython.boundscheck(False)
@cython.wraparound(False)
def main_interface(np.ndarray[int, ndim=2, mode="c"] elements not None,np.ndarray[double, ndim=2, mode="c"] points not None):
    """
def main_interface(np.ndarray[np.int64_t, ndim=2, mode="c"] elements not None,np.ndarray[np.float64_t, ndim=2, mode="c"] points not None):
    """
    cdef int element_rows, points_rows, element_cols, points_cols

    element_rows, element_cols  = elements.shape[0], elements.shape[1]
    points_rows, points_cols  = points.shape[0], points.shape[1]

    py_cpp_interface (&points[0,0] ,points_rows, points_cols, &elements[0,0], element_rows, element_cols)

    return None

