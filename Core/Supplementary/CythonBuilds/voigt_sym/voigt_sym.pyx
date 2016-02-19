
import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

cdef Voigtc_3(const double *C, double *VoigtA):
    VoigtA[0] = C[0]
    VoigtA[1] = C[4]
    VoigtA[2] = C[8]
    VoigtA[3] = 0.5*(C[1]+C[3])
    VoigtA[4] = 0.5*(C[2]+C[6])
    VoigtA[5] = 0.5*(C[5]+C[7])
    VoigtA[6] = VoigtA[1]
    VoigtA[7] = C[40]
    VoigtA[8] = C[44]
    VoigtA[9] = 0.5*(C[37]+C[39])
    VoigtA[10] = 0.5*(C[38]+C[42])
    VoigtA[11] = 0.5*(C[41]+C[43])
    VoigtA[12] = VoigtA[2]
    VoigtA[13] = VoigtA[8]
    VoigtA[14] = C[80]
    VoigtA[15] = 0.5*(C[73]+C[75])
    VoigtA[16] = 0.5*(C[74]+C[78])
    VoigtA[17] = 0.5*(C[77]+C[79])
    VoigtA[18] = VoigtA[3]
    VoigtA[19] = VoigtA[9]
    VoigtA[20] = VoigtA[15]
    VoigtA[21] = 0.5*(C[10]+C[12])
    VoigtA[22] = 0.5*(C[11]+C[15])
    VoigtA[23] = 0.5*(C[14]+C[16])
    VoigtA[24] = VoigtA[4]
    VoigtA[25] = VoigtA[10]
    VoigtA[26] = VoigtA[16]
    VoigtA[27] = VoigtA[22]
    VoigtA[28] = 0.5*(C[20]+C[24])
    VoigtA[29] = 0.5*(C[23]+C[25])
    VoigtA[30] = VoigtA[5]
    VoigtA[31] = VoigtA[11]
    VoigtA[32] = VoigtA[17]
    VoigtA[33] = VoigtA[23]
    VoigtA[34] = VoigtA[29]
    VoigtA[35] = 0.5*(C[50]+C[52])
    
    
cdef Voigtc_2(const double *C, double *VoigtA):
    VoigtA[0] = C[0]
    VoigtA[1] = C[3]
    VoigtA[2] = 0.5*(C[1]+C[2])
    VoigtA[3] = VoigtA[1]
    VoigtA[4] = C[15]
    VoigtA[5] = 0.5*(C[13]+C[14])
    VoigtA[6] = VoigtA[2]
    VoigtA[7] = VoigtA[5]
    VoigtA[8] = 0.5*(C[5]+C[6])
    

@boundscheck(False)
@wraparound(False)
def voigt_sym(np.ndarray[double,ndim=4,mode='c'] C):
    cdef np.ndarray[double, ndim=2,mode='c'] VoigtA 
    cdef int n1dim = C.shape[0]
    if n1dim == 3:
        VoigtA = np.zeros((6,6),dtype=np.float64)
        Voigtc_3(&C[0,0,0,0],&VoigtA[0,0])
    elif n1dim == 2:
        VoigtA = np.zeros((3,3),dtype=np.float64)
        Voigtc_2(&C[0,0,0,0],&VoigtA[0,0])

    return VoigtA
