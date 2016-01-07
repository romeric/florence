# distutils: language = c++
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def remove_duplicates(np.ndarray[unsigned long,ndim=2,mode='c'] arr, int axis=0):
    """Remove duplicated rows or columns form a 2D array"""

    if axis == 1:
        arr = arr.T.copy()

    cdef np.ndarray[unsigned long,ndim=2] sorted_arr = np.sort(arr,axis=1)
    cdef vector[int] vec = remove_duplicates_c(&sorted_arr[0,0],arr.shape[0],arr.shape[1])
    cdef np.ndarray[long] to_remove = np.array(vec,copy=False)
    return np.delete(arr,np.unique(to_remove),0)


cdef remove_duplicates_c(unsigned long *arr, int rows, int cols):

    cdef:
        int i, j, k, l, summer, counter
        vector[int] vec

    for i in range(rows):
        for j in range(rows):
            if i != j:
                summer = 0
                for k in range(cols):
                    if arr[i*cols+k] == arr[j*cols+k]:
                        summer += 1

                if summer == cols:
                    counter = 0
                    for l in range(vec.size()):
                        if vec[l]==i:
                            counter = 1
                            break
                    if counter == 0:
                        vec.push_back(j)

    return vec
