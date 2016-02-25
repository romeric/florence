
import numpy as np
cimport numpy as np
import cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEI = np.int64
ctypedef np.int64_t DTYPEI_t

@cython.boundscheck(False)
@cython.wraparound(False)
def NPFROMFILE_Loop_Cython(np.ndarray[DTYPE_t, ndim=1] FileContent,DTYPEI_t nnode):
	# DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)
	cdef DTYPEI_t cols1 = np.int64((FileContent[4*nnode+2+1] - 100) + 2)
	cdef DTYPEI_t cols2 = 0
	cdef DTYPEI_t counter_edge = 0
	cdef DTYPEI_t i 
	for i in range(4*nnode+2+1,FileContent.shape[0]):
		if FileContent[4*nnode+2+1+cols1*counter_edge] == FileContent[4*nnode+2+1]:
			counter_edge +=1
		else:
			cols2 = np.int64((FileContent[4*nnode+2+1+cols1*counter_edge] - 200)+2)
			break
	cdef DTYPEI_t counter_face = 0
	for i in range(4*nnode+2+1+cols1*counter_edge,FileContent.shape[0]):
		if 4*nnode+2+1+cols1*counter_edge+cols2*counter_face >= FileContent.shape[0]:
			break
		else:
			if FileContent[4*nnode+2+1+cols1*counter_edge+cols2*counter_face] == FileContent[4*nnode+2+1+cols1*counter_edge]:
				counter_face +=1
			else:
				break
	
	cdef np.ndarray nelse_type = np.zeros((2,2),dtype=np.int64); 
	nelse_type[0,0] = np.int64(FileContent[4*nnode+2+1]); nelse_type[1,0] = np.int64(FileContent[4*nnode+2+1+cols1*counter_edge])
	nelse_type[0,1] = counter_edge; nelse_type[1,1]=counter_face

	return nelse_type 