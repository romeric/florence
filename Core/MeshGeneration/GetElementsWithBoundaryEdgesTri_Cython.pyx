import cython
import numpy as np 
cimport numpy as np 
from Core.Supplementary.Where import whereEQ

@cython.boundscheck(False)
@cython.wraparound(False)
def GetElementsWithBoundaryEdgesTri_Cython(np.ndarray[np.int64_t, ndim=2] elements, np.ndarray[np.int64_t, ndim=2] edges):


	cdef np.ndarray edge_elements = np.zeros(edges.shape[0],dtype=np.int64)
	cdef np.int64_t i,j 
	cdef np.ndarray[np.int64_t, ndim=1] x, y 
	for i in range(edges.shape[0]):
		x = np.array([],dtype=np.int64)
		for j in range(edges.shape[1]):
			x = np.asarray(np.append(x,whereEQ(elements,edges[i,j])[0]))
		for k in range(x.shape[0]):
			y = np.where(x==x[k])[0]
			if y.shape[0]==edges.shape[1]:
				edge_elements[i] = x[k]
				break

	return edge_elements