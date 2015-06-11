import numpy as np
cimport numpy as np
import cython 

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEI = np.int64
ctypedef np.int64_t DTYPEI_t

@cython.boundscheck(False)
def TwoLoopNode_Cython(tuple duplicates_list,np.ndarray[DTYPEI_t, ndim=2] duplicates,DTYPEI_t dups0, DTYPEI_t counter):
	cdef DTYPEI_t j, k, dshape0, dshape1
	dshape0 = len(duplicates_list)
	for j in range(dshape0):
		dshape1 = duplicates_list[j].shape[0]
		for k in range(1,dshape1):
			duplicates[counter,:] = dups0+np.array([duplicates_list[j][0],duplicates_list[j][k]])
			counter +=1
	return duplicates, counter