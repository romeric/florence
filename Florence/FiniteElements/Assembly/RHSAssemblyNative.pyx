from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np 

# ctypedef Int long long int
ctypedef np.int64_t Int
ctypedef np.float64_t Float


# The following fused typedefs are provided for
# cases when RHS is lowered to 32 bit, but their
# performance is terrible. Recompile instead for 32bit RHSes 
# ctypedef fused Float:
#     np.float32_t
#     np.float64_t

@boundscheck(False)
@wraparound(False)
def RHSAssemblyNative(np.ndarray[Float,ndim=2] T, np.ndarray[Float,ndim=2] t, 
    Int elem, Int nvar, Int nodeperelem,
    np.ndarray[unsigned long,ndim=2, mode='c'] elements):

    cdef long nelem = elements.shape[0]

    RHSAssemblyNative_(&T[0,0],&t[0,0],elem,nvar,nodeperelem,nelem,&elements[0,0])




cdef inline void RHSAssemblyNative_(Float *T, const Float *t, 
    Int elem, Int nvar, Int nodeperelem, long nelem, const unsigned long *elements):

    cdef Int i, T_idx, iterator

    # INTERNAL TRACTION FORCE ASSEMBLY
    for i in range(nodeperelem):
        T_idx = elements[elem*nodeperelem+i]*nvar
        for iterator in range(nvar):
            T[T_idx+iterator] += t[i*nvar+iterator]

    # For the record the following is the numpy equivalent version of above
    # for iterator in range(0,nvar):
        # T[mesh.elements[elem,:]*nvar+iterator,0]+=t[iterator::nvar,0]
