from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np 
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

@boundscheck(False)
@wraparound(False)
def SparseAssemblyNative(np.ndarray[long] i, np.ndarray[long] j,
	np.ndarray[double] coeff, np.ndarray[long] I, np.ndarray[long] J,
	np.ndarray[double] V, long elem, int nvar, int nodeperelem,
	np.ndarray[unsigned long,ndim=2, mode='c'] elements):

	cdef int i_shape = i.shape[0]
	cdef int j_shape = j.shape[0]

	SparseAssemblyNative_(&i[0],&j[0],&coeff[0],&I[0],&J[0],&V[0],
		elem,nvar,nodeperelem,&elements[0,0],i_shape,j_shape)




cdef void SparseAssemblyNative_(const long *i, const long *j, const double *coeff, long *I, long *J,
	double *V, long elem, int nvar, int nodeperelem, const unsigned long *elements,int i_shape, int j_shape):

	cdef int *current_row_column = <int*>malloc(sizeof(int)*nvar*nodeperelem)
	

	cdef long *full_current_row = <long*>malloc(sizeof(long)*i_shape)
	cdef long *full_current_column = <long*>malloc(sizeof(long)*j_shape)

	cdef int iterator, counter, ncounter

	for counter in range(nodeperelem):
		for ncounter in range(nvar):
			current_row_column[nvar*counter+ncounter] = nvar*elements[elem*nodeperelem+counter]+ncounter


	memcpy(full_current_row,i,i_shape*sizeof(long))
	memcpy(full_current_column,j,j_shape*sizeof(long))


	for counter in range(0,nvar*nodeperelem):
		for iterator in range(i_shape):
			if i[iterator]==counter:
				full_current_row[iterator] = current_row_column[counter]
		for iterator in range(j_shape):
			if j[iterator]==counter:
				full_current_column[iterator] = current_row_column[counter]


	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
	cdef int low, high
	low = (nvar*nodeperelem)*(nvar*nodeperelem)*elem
	high = (nvar*nodeperelem)*(nvar*nodeperelem)*(elem+1)

	cdef int incrementer = 0
	for counter in range(low,high):
		I[counter] = full_current_row[incrementer]
		J[counter] = full_current_column[incrementer]
		V[counter] = coeff[incrementer]

		incrementer += 1


	free(full_current_row)
	free(full_current_column)
	free(current_row_column)