import numpy as np
cimport numpy as np

def _hessian(np.ndarray[double, ndim=2, mode='c'] F, np.ndarray[double, ndim=2, mode='c'] H, 
    np.ndarray[double, ndim=2, mode='c'] b, double J, np.ndarray[double, ndim=2] D0, np.ndarray[double, ndim=2] N,
    double mu1, double mu2, double mu3, double lamb, double eps_1, double eps_2, double eps_3):

    cdef int ndim = F.shape[0]
    cdef np.ndarray[double, ndim=2, mode='c'] C

    if ndim==3:
        C = np.zeros((9,9),dtype=np.float64)
        _hessian2D(&F[0,0], &H[0,0], &b[0,0], J, &D0[0,0], &N[0,0], 
            mu1, mu2, mu3, lamb, eps_1, eps_2, eps_3,
            &C[0,0])
    elif ndim==2:
        C = np.zeros((5,5),dtype=np.float64)
        _hessian2D(&F[0,0], &H[0,0], &b[0,0], J, &D0[0,0], &N[0,0], 
            mu1, mu2, mu3, lamb, eps_1, eps_2, eps_3,
            &C[0,0])

    return C

cdef inline _hessian3D(const double *F, const double *H, const double *b, double J, double *D0, double *N,
    double mu1, double mu2, double mu3, double lamb, double eps_1, double eps_2, double eps_3,
    double *C):

    cdef double F11 = F[0]
    cdef double F12 = F[1]
    cdef double F13 = F[2]
    cdef double F21 = F[3]
    cdef double F22 = F[4]
    cdef double F23 = F[5]
    cdef double F31 = F[6]
    cdef double F32 = F[7]
    cdef double F33 = F[8]

    cdef double H11 = H[0]
    cdef double H12 = H[1]
    cdef double H13 = H[2]
    cdef double H21 = H[3]
    cdef double H22 = H[4]
    cdef double H23 = H[5]
    cdef double H31 = H[6]
    cdef double H32 = H[7]
    cdef double H33 = H[8]

    cdef double b11 = b[0]
    cdef double b12 = b[1]
    cdef double b13 = b[2]
    cdef double b21 = b[3]
    cdef double b22 = b[4]
    cdef double b23 = b[5]
    cdef double b31 = b[6]
    cdef double b32 = b[7]
    cdef double b33 = b[8]

    cdef double D1 = D0[0]
    cdef double D2 = D0[1]
    cdef double D3 = D0[2]

    cdef double N1 = N[0]
    cdef double N2 = N[1]
    cdef double N3 = N[2]


    C[0] = (- D1**2/2 + (3*D2**2)/2 + (3*D3**2)/2)/eps_2 - 2*lamb*(J - 1) + lamb*(2*J - 1)
    C[1] = lamb*(2*J - 1) - (D1**2/2 + D2**2/2 - D3**2/2)/eps_2 - (4*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + (H21*N1 + H22*N2 + H23*N3)**2))/J + \
        (2*mu2*(- 2*b12**2 + 2*b11*b22))/J + (4*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J
    C[2] = lamb*(2*J - 1) - (D1**2/2 - D2**2/2 + D3**2/2)/eps_2 - (4*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J + \
        (2*mu2*(- 2*b13**2 + 2*b11*b33))/J + (4*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J
    C[3] = - (mu2*(2*b11*b12 - 2*b11*b21))/J - (D1*D2)/eps_2
    C[4] = - (mu2*(2*b11*b13 - 2*b11*b31))/J - (D1*D3)/eps_2
    C[5] = - (mu2*(2*b12*b13 - 2*b11*b23))/J - (mu2*(2*b12*b13 - 2*b11*b32))/J - \
        (D2*D3)/eps_2 - (4*mu3*(H21*N1 + H22*N2 + H23*N3)*(H31*N1 + H32*N2 + H33*N3))/J
    C[6] = C[1]
    C[7] = ((3*D1**2)/2 - D2**2/2 + (3*D3**2)/2)/eps_2 - 2*lamb*(J - 1) + lamb*(2*J - 1)
    C[8] = lamb*(2*J - 1) - (- D1**2/2 + D2**2/2 + D3**2/2)/eps_2 - (4*mu3*((H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J + \
        (2*mu2*(- 2*b23**2 + 2*b22*b33))/J + (4*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J
    C[9] = (mu2*(2*b12*b22 - 2*b21*b22))/J - (D1*D2)/eps_2
    C[10] = (mu2*(2*b13*b22 - 2*b21*b23))/J - (mu2*(2*b21*b23 - 2*b22*b31))/J - (D1*D3)/eps_2 - \
        (4*mu3*(H11*N1 + H12*N2 + H13*N3)*(H31*N1 + H32*N2 + H33*N3))/J
    C[11] = - (mu2*(2*b22*b23 - 2*b22*b32))/J - (D2*D3)/eps_2
    C[12] = C[2]
    C[13] = C[8]
    C[14] = ((3*D1**2)/2 + (3*D2**2)/2 - D3**2/2)/eps_2 - 2*lamb*(J - 1) + lamb*(2*J - 1)
    C[15] = (mu2*(2*b12*b33 - 2*b31*b32))/J + (mu2*(2*b21*b33 - 2*b31*b32))/J - (D1*D2)/eps_2 - \
        (4*mu3*(H11*N1 + H12*N2 + H13*N3)*(H21*N1 + H22*N2 + H23*N3))/J
    C[16] = (mu2*(2*b13*b33 - 2*b31*b33))/J - (D1*D3)/eps_2
    C[17] = (mu2*(2*b23*b33 - 2*b32*b33))/J - (D2*D3)/eps_2
    C[18] = C[3]
    C[19] = C[9]
    C[20] = C[15]
    C[21] = (D1**2/2 + D2**2/2 + D3**2/2)/eps_2 - lamb*(J - 1) + (2*mu3*((H11*N1 + H12*N2 + \
        H13*N3)**2 + (H21*N1 + H22*N2 + H23*N3)**2))/J - (mu2*(b11*b22 - b12*b21))/J - \
        (mu2*(- 2*b12**2 + b21*b12 + b11*b22))/J - (2*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + \
            (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J
    C[22] = (2*mu3*(H21*N1 + H22*N2 + H23*N3)*(H31*N1 + H32*N2 + H33*N3))/J - \
        (mu2*(b11*b23 + b13*b21 - 2*b12*b31))/J - (mu2*(b11*b23 - 2*b12*b13 + b13*b21))/J
    C[23] = (mu2*(b12*b23 - b13*b22))/J - (mu2*(b12*b23 + b13*b22 - 2*b12*b32))/J + (2*mu3*(H11*N1 + H12*N2 + H13*N3)*(H31*N1 + H32*N2 + H33*N3))/J
    C[24] = C[4]
    C[25] = C[10]
    C[26] = C[16]
    C[27] = C[22]
    C[28] = (D1**2/2 + D2**2/2 + D3**2/2)/eps_2 - lamb*(J - 1) + (2*mu3*((H11*N1 + H12*N2 + H13*N3)**2 +\
        (H31*N1 + H32*N2 + H33*N3)**2))/J - (mu2*(b11*b33 - b13*b31))/J - \
        (mu2*(- 2*b13**2 + b31*b13 + b11*b33))/J - (2*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + \
            (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J
    C[29] = (2*mu3*(H11*N1 + H12*N2 + H13*N3)*(H21*N1 + H22*N2 + H23*N3))/J - (mu2*(b12*b33 - 2*b13*b23 + b13*b32))/J - (mu2*(b12*b33 - b13*b32))/J
    C[30] = C[5]
    C[31] = C[11]
    C[32] = C[17]
    C[33] = C[23]
    C[34] = C[29]
    C[35] = (D1**2/2 + D2**2/2 + D3**2/2)/eps_2 - lamb*(J - 1) + (2*mu3*((H21*N1 + H22*N2 + H23*N3)**2 + \
        (H31*N1 + H32*N2 + H33*N3)**2))/J - (mu2*(b22*b33 - b23*b32))/J - \
        (mu2*(- 2*b23**2 + b32*b23 + b22*b33))/J - (2*mu3*((H11*N1 + H12*N2 + H13*N3)**2 + \
            (H21*N1 + H22*N2 + H23*N3)**2 + (H31*N1 + H32*N2 + H33*N3)**2))/J



cdef inline _hessian2D(const double *F, const double *H, const double *b, double J, double *D0, double *N,
    double mu1, double mu2, double mu3, double lamb, double eps_1, double eps_2, double eps_3,
    double *C):

    cdef double F11 = F[0]
    cdef double F12 = F[1]
    cdef double F21 = F[2]
    cdef double F22 = F[3]

    cdef double H11 = H[0]
    cdef double H12 = H[1]
    cdef double H21 = H[2]
    cdef double H22 = H[3]

    cdef double b11 = b[0]
    cdef double b12 = b[1]
    cdef double b21 = b[2]
    cdef double b22 = b[3]

    cdef double D1 = D0[0]
    cdef double D2 = D0[1]

    cdef double N1 = N[0]
    cdef double N2 = N[1]


    # C[0] = lamb*(2*J - 1) - (D1**2/2 - (3*D2**2)/2)/eps_2 - 2*lamb*(J - 1)
    # C[1] = lamb*(2*J - 1) - (D1**2/2 + D2**2/2)/eps_2 + (2*mu2*(- 2*b12**2 + 2*b11*b22))/J
    # C[2] = - (mu2*(2*b11*b12 - 2*b11*b21))/J - (D1*D2)/eps_2
    # C[3] = C[1]
    # C[4] = ((3*D1**2)/2 - D2**2/2)/eps_2 - 2*lamb*(J - 1) + lamb*(2*J - 1)
    # C[5] = (mu2*(2*b12*b22 - 2*b21*b22))/J - (D1*D2)/eps_2
    # C[6] = C[2]
    # C[7] = C[5]
    # C[8] = (D1**2/2 + D2**2/2)/eps_2 - lamb*(J - 1) - (mu2*(b11*b22 - b12*b21))/J - (mu2*(- 2*b12**2 + b21*b12 + b11*b22))/J



    C[0] =2.*lamb*(2*J - 1) - 4*lamb*(J - 1) - (2*(D1**2/2 - (3*D2**2)/2))/eps_2 - ((eps_1*eps_2*eps_3*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (D2*J*b21*eps_1*eps_2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2)) - (D2*((D2*eps_1*eps_3*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b12*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))))/eps_2
    C[1] =2.*lamb*(2*J - 1) - (2*(D1**2/2 + D2**2/2))/eps_2 + ((D1*eps_1*eps_3*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b21*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2)) + (4*mu2*(2*b11*b22 - 2*b12**2))/J + (D2*((eps_1*eps_2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (D1*J*b12*eps_1*eps_2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))))/eps_2
    C[2] =(D2*((eps_1*eps_2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b12*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))))/eps_2 - (2*mu2*(2*b11*b12 - 2*b11*b21))/J - ((eps_1*eps_2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b21*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2)) - (2*D1*D2)/eps_2
    C[3] =(D2*J*b21*eps_1*eps_2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (eps_1*eps_2*eps_3*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[4] =(D2*eps_1*eps_3*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b12*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(2*F11*N1 + 2*F12*N2)*(mu3/eps_3)**(1/2) + 4*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[5] = C[1]
    C[6] =(2.*((3*D1**2)/2 - D2**2/2))/eps_2 - 4*lamb*(J - 1) - ((eps_1*eps_2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (D1*J*b12*eps_1*eps_2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2)) + 2*lamb*(2*J - 1) - (D1*((D1*eps_1*eps_3*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b21*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))))/eps_2
    C[7] =(2.*mu2*(2*b12*b22 - 2*b21*b22))/J - ((eps_1*eps_2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b12*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2)) + (D1*((eps_1*eps_2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b21*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))))/eps_2 - (2*D1*D2)/eps_2
    C[8] =(D1*eps_1*eps_3*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b21*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[9] =(D1*J*b12*eps_1*eps_2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (eps_1*eps_2*eps_3*(D2/eps_2 + 2*(2*F21*N1 + 2*F22*N2)*(mu3/eps_3)**(1/2) + 4*D2*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[10] = C[2]
    C[11] = C[7]
    C[12] =(2*(D1**2/2 + D2**2/2))/eps_2 - ((eps_1*eps_2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b21*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2)) - 2*lamb*(J - 1) - ((eps_1*eps_2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) + (J*b12*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)))*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2)) - (2*mu2*(b11*b22 - b12*b21))/J - (2*mu2*(b11*b22 + b12*b21 - 2*b12**2))/J
    C[13] =- (eps_1*eps_2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2))*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b21*eps_1*eps_2**2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[14] =- (eps_1*eps_2*eps_3*(D1/eps_2 + 2*(F11*N1 + F12*N2)*(mu3/eps_3)**(1/2) + 2*D1*J*(mu3/eps_3)**(1/2))*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2)) - (J*b12*eps_1*eps_2**2*eps_3*(D2/eps_2 + 2*(F21*N1 + F22*N2)*(mu3/eps_3)**(1/2) + 2*D2*J*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[15] = C[3]
    C[16] = C[8]
    C[17] = C[13]
    C[18] =-(eps_1*eps_2*eps_3*(J*b11*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[19] =-(J*b12*eps_1*eps_2**2*eps_3)/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
    C[20] = C[4]
    C[21] = C[9]
    C[22] = C[14]
    C[23] = C[19]
    C[24] =-(eps_1*eps_2*eps_3*(J*b22*eps_2 + b11*b22*eps_1 - b12*b21*eps_1 + 2*J*b11*b22*eps_1*eps_2*(mu3/eps_3)**(1/2) - 2*J*b12*b21*eps_1*eps_2*(mu3/eps_3)**(1/2)))/(J**2*eps_2**2*eps_3 + b11*b22*eps_1**2*eps_3 - b12*b21*eps_1**2*eps_3 + J*b11*eps_1*eps_2*eps_3 + J*b22*eps_1*eps_2*eps_3 + 4*J**2*b11*b22*eps_1**2*eps_2**2*mu3 - 4*J**2*b12*b21*eps_1**2*eps_2**2*mu3 + 2*J**2*b11*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 2*J**2*b22*eps_1*eps_2**2*eps_3*(mu3/eps_3)**(1/2) + 4*J*b11*b22*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2) - 4*J*b12*b21*eps_1**2*eps_2*eps_3*(mu3/eps_3)**(1/2))
     

