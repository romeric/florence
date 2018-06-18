import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

cdef inline void FillConstitutiveB_(double *B, const double* SpatialGradient,
                     int ndim, int nvar, int rows, int cols) nogil:
    cdef int i

    if ndim == 2:

        for i in range(rows):
            B[i*cols*nvar] = SpatialGradient[i]
            B[i*cols*nvar+cols+1] = SpatialGradient[i+rows]
            
            # SEEMS BUGGY
            # B[i*cols*nvar+(cols-1)] = SpatialGradient[i+rows]
            # B[i*cols*nvar+(2*cols-1)] = SpatialGradient[i]

            B[i*cols*nvar+2] = SpatialGradient[i+rows]
            B[i*cols*nvar+cols+2] = SpatialGradient[i]

    elif ndim == 3:

        for i in range(rows):
            B[i*cols*nvar] = SpatialGradient[i]
            B[i*cols*nvar+cols+1] = SpatialGradient[i+rows]
            B[i*cols*nvar+2*(cols+1)] = SpatialGradient[i+2*rows]

            B[i*cols*nvar+cols+5] = SpatialGradient[i+2*rows]
            B[i*cols*nvar+2*cols+5] = SpatialGradient[i+rows]

            B[i*cols*nvar+4] = SpatialGradient[i+2*rows]
            B[i*cols*nvar+2*cols+4] = SpatialGradient[i]

            B[i*cols*nvar+3] = SpatialGradient[i+rows]
            B[i*cols*nvar+cols+3] = SpatialGradient[i]

            




@boundscheck(False)
@wraparound(False)
def FillConstitutiveB(np.ndarray[double,ndim=2,mode='c'] B, 
                     np.ndarray[double,ndim=2,mode='c'] SpatialGradient,
                     int ndim, int nvar):
    
    cdef int rows = SpatialGradient.shape[1]
    cdef int cols = B.shape[1]
    

    FillConstitutiveB_(&B[0,0],&SpatialGradient[0,0],
        ndim,nvar,rows,cols)











cdef inline void FillGeometricB_(double *B,double *SpatialGradient, 
                        double *S, double *CauchyStressTensor,
                        int ndim, int nvar, int rows, int cols) nogil:
    cdef int i

    if ndim == 2:

        for i in range(rows):
            B[i*cols*nvar] = SpatialGradient[i]
            B[i*cols*nvar+1] = SpatialGradient[i+rows]
            
            B[i*cols*nvar+(cols+2)] = SpatialGradient[i]
            B[i*cols*nvar+(cols+2)+1] = SpatialGradient[i+rows]

        S[0] = CauchyStressTensor[0]
        S[1] = CauchyStressTensor[1]
        S[4] = CauchyStressTensor[2]
        S[5] = CauchyStressTensor[3]

        S[10] = CauchyStressTensor[0]
        S[11] = CauchyStressTensor[1]
        S[14] = CauchyStressTensor[2]
        S[15] = CauchyStressTensor[3]

    elif ndim == 3:

        for i in range(rows):
            B[i*cols*nvar] = SpatialGradient[i]
            B[i*cols*nvar+1] = SpatialGradient[i+rows]
            B[i*cols*nvar+2] = SpatialGradient[i+2*rows]

            B[i*cols*nvar+(cols+3)] = SpatialGradient[i]
            B[i*cols*nvar+(cols+3)+1] = SpatialGradient[i+rows]
            B[i*cols*nvar+(cols+3)+2] = SpatialGradient[i+2*rows]

            B[i*cols*nvar+2*(cols+3)] = SpatialGradient[i]
            B[i*cols*nvar+2*(cols+3)+1] = SpatialGradient[i+rows]
            B[i*cols*nvar+2*(cols+3)+2] = SpatialGradient[i+2*rows]

        S[0] = CauchyStressTensor[0]
        S[1] = CauchyStressTensor[1]
        S[2] = CauchyStressTensor[2]
        S[9] = CauchyStressTensor[3]
        S[10] = CauchyStressTensor[4]
        S[11] = CauchyStressTensor[5]
        S[18] = CauchyStressTensor[6]
        S[19] = CauchyStressTensor[7]
        S[20] = CauchyStressTensor[8]

        S[30] = CauchyStressTensor[0]
        S[31] = CauchyStressTensor[1]
        S[32] = CauchyStressTensor[2]
        S[39] = CauchyStressTensor[3]
        S[40] = CauchyStressTensor[4]
        S[41] = CauchyStressTensor[5]
        S[48] = CauchyStressTensor[6]
        S[49] = CauchyStressTensor[7]
        S[50] = CauchyStressTensor[8]

        S[60] = CauchyStressTensor[0]
        S[61] = CauchyStressTensor[1]
        S[62] = CauchyStressTensor[2]
        S[69] = CauchyStressTensor[3]
        S[70] = CauchyStressTensor[4]
        S[71] = CauchyStressTensor[5]
        S[78] = CauchyStressTensor[6]
        S[79] = CauchyStressTensor[7]
        S[80] = CauchyStressTensor[8]


@boundscheck(False)
@wraparound(False)
def FillGeometricB(np.ndarray[double,ndim=2,mode='c'] B, 
                     np.ndarray[double,ndim=2, mode='c'] SpatialGradient,
                     np.ndarray[double,ndim=2,mode='c'] S,
                     np.ndarray[double,ndim=2,mode='c'] CauchyStressTensor, int ndim, int nvar):
    
    cdef int rows = SpatialGradient.shape[1]
    cdef int cols = B.shape[1]
    
    FillGeometricB_(&B[0,0],&SpatialGradient[0,0],&S[0,0],
        &CauchyStressTensor[0,0],ndim,nvar,rows,cols)








cdef inline void GetTotalTraction_(double *TotalTraction, const double *CauchyStressTensor, int ndim):
    if ndim == 2:
        TotalTraction[0] = CauchyStressTensor[0]
        TotalTraction[1] = CauchyStressTensor[3]
        TotalTraction[2] = CauchyStressTensor[1]
    elif ndim==3:
        TotalTraction[0] = CauchyStressTensor[0]
        TotalTraction[1] = CauchyStressTensor[4]
        TotalTraction[2] = CauchyStressTensor[8]
        TotalTraction[3] = CauchyStressTensor[1]
        TotalTraction[4] = CauchyStressTensor[2]
        TotalTraction[5] = CauchyStressTensor[5]




@boundscheck(False)
@wraparound(False)
def GetTotalTraction(np.ndarray[double,ndim=2,mode='c'] CauchyStressTensor):
    cdef np.ndarray[double,ndim=2,mode='c'] TotalTraction
    cdef int ndim = CauchyStressTensor.shape[0]
    if ndim == 2:
        TotalTraction = np.zeros((3,1),dtype=np.float64)
    elif ndim == 3:
        TotalTraction = np.zeros((6,1),dtype=np.float64)
    GetTotalTraction_(&TotalTraction[0,0], &CauchyStressTensor[0,0], ndim)
    return TotalTraction