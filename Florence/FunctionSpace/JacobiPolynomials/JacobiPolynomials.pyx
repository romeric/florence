from libc.stdlib cimport malloc, free 
from cython cimport double, sizeof, boundscheck, wraparound

# cdef extern from "jacobi.c" nogil:
#     void jacobi(const unsigned short n, const double xi,
#             const double a, const double b, double *P)
#     void diffjacobi(const unsigned short n, const double xi,
#             const double a, const double b, const unsigned short opt, double *dP)


cdef inline void jacobi(const unsigned short n, const double xi, const double a, const double b, double *P) nogil:
    cdef double a1n,a2n,a3n,a4n;
    cdef unsigned short p;

    P[0]=1.0
    if n>0:
        P[1] = 0.5*((a-b)+(a+b+2)*xi)
    if n>1:
        for p in range(1,n):
            a1n = 2*(p+1)*(p+a+b+1)*(2*p+a+b)
            a2n = (2*p+a+b+1)*(a*a-b*b)
            a3n = (2*p+a+b)*(2*p+a+b+1)*(2*p+a+b+2)
            a4n = 2*(p+a)*(p+b)*(2*p+a+b+2)

            P[p+1] = ((a2n+a3n*xi)*P[p]-a4n*P[p-1])/a1n


cdef inline void diffjacobi(const unsigned short n, const double xi, const double a, const double b, const unsigned short opt, double *dP) nogil:
    cdef unsigned short p
    cdef double *P = <double*>malloc( (n+1)*sizeof(double))
    if opt==1:
        jacobi(n,xi,a+1,b+1,P)
    else:
        jacobi(n,xi,a,b,P)

    for p in range(1,n+1):
        dP[p] = 0.5*(a+b+p+1)*P[p-1]

    free(P)


@boundscheck(False)
@wraparound(False)
def JacobiPolynomials(const unsigned short n, double xi, double a=0., double b=0.):
    cdef:
        int i
        double *P = <double*>malloc( (n+1)*sizeof(double))
    jacobi(n,xi,a,b,P);
    P_py=[0]*(n+1)
    for i in range(n+1):
        P_py[i] = P[i]

    free(P)
    return P_py

@boundscheck(False)
@wraparound(False)
def DiffJacobiPolynomials(const int n,double xi,double a=0.,double b=0.,int opt=0):
    cdef:
        int i
        double *dP = <double*>malloc( (n+1)*sizeof(double))
    
    diffjacobi(n,xi,a,b,opt,dP);
    dP_py=[0]*(n+1)
    for i in range(n+1):
        dP_py[i] = dP[i]
    
    free(dP);
    return dP_py





#======================================================================#
#======================================================================#
#======================================================================#

from libc cimport math 
import numpy as np
cimport numpy as np



# 1D - LINE
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

@boundscheck(False)
def NormalisedJacobi1D(int C, double x):

    cdef:
        double[::1] p = np.zeros(C+2)
        int i

    for i in range(0,C+2):
        p[i] = JacobiPolynomials(i,x,0,0)[-1]*math.sqrt((2.*i+1.)/2.)

    return p  


# 2D - TRI
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

@boundscheck(False)
def NormalisedJacobi2D(int C, x):
    """Computes the orthogonal base of 2D polynomials of degree less than
        or equal to C+1 at the point x=(r,s) in [-1,1]**2 (i.e. on the reference quad)"""


    cdef:
        int N = (C+2)*(C+3)/2 
        double[::1] p = np.zeros(N)
        double r = x[0]
        double s = x[1]
        int nDeg, i, j
        double p_i, p_j, q_i, q_j

    # ORDERING: FIRST INCREASING THE DEGREE AND THEN LEXOGRAPHIC ORDER
    cdef int ncount = 0 
    # LOOP ON DEGREE
    for nDeg in range(0,C+2):
      # LOOP INCREASING I
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1.
                q_i = 1.  
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1]
                q_i = q_i*(1.-s)/2.
            # VALUE FOR J
            j = nDeg-i
            if j==0:
               p_j = 1.
            else:
               p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]
            
            # factor = np.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            factor = math.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            p[ncount] = ( p_i*q_i*p_j )*factor

            ncount += 1
    return p 



def NormalisedJacobiTri(int C, x):   
    """Computes the orthogonal base of 2D polynomials of degree less than
        or equal to n at the point x=(xi,eta) in the reference triangle"""

    cdef:
        double xi = x[0]
        double eta = x[1]
        double r, s 

    if eta==1.: 
        r = -1.
        s = 1.
    else:
        r = 2.*(1+xi)/(1.-eta)-1.
        s = eta

    # return NormalisedJacobi2D(C,np.array([r,s]))
    return NormalisedJacobi2D(C,[r,s]) 


@boundscheck(False)
def GradNormalisedJacobiTri(int C,x, EvalOpt=0):
    """Computes the orthogonal base of 2D polynomials of degree less than
        or equal to n at the point x=(r,s) in [-1,1]**2"""
    
    cdef:
        int N = (C+2)*(C+3)/2 
        double[::1] p = np.zeros(N);
        double[::1] dp_dxi  = np.zeros(N)
        double[::1] dp_deta = np.zeros(N)
        double r = x[0]
        double s = x[1]
        double xi, eta, dr_dxi, dr_deta
        double dp_dr, dp_ds
        int nDeg, i, j
        double factor 

    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if EvalOpt==1:
        if s==1:
            s=0.99999999999999

    xi = (1.+r)*(1.-s)/2.-1
    eta = s

    dr_dxi  = 2./(1.-eta)
    dr_deta = 2.*(1.+xi)/(1.-eta)**2
    # DERIVATIVE OF s IS NOT NEEDED SINCE s=eta

    # ORDERING: FIRST INCREASING THE DEGREE AND THEN LEXOGRAPHIC ORDER
    cdef int ncount = 0
    # Loop on degree
    for nDeg in range(0,C+2):
      # LOOP INCREASING I
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1.  
                q_i = 1.  
                dp_i = 0.
                dq_i = 0.
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1] 
                dp_i = JacobiPolynomials(i-1,r,1.,1.)[-1]*(i+1.)/2.    
                
                q_i = q_i*(1.-s)/2.
                dq_i = 1.*q_i*(-i)/(1-s)
            
            # VALUE FOR J
            j = nDeg-i
            if j==0:
                p_j = 1.
                dp_j = 0.
            else:
                p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]
                dp_j = JacobiPolynomials(j-1,s,2.*i+2.,1.)[-1]*(j+2.*i+2.)/2.  
            
            # factor = np.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            factor = math.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            # NORMALISED POLYNOMIAL
            p[ncount] = ( p_i*q_i*p_j )*factor
            # DERIVATIVE WITH RESPECT TO (r,s)
            dp_dr = ( (dp_i)*q_i*p_j )*factor
            dp_ds = ( p_i*(dq_i*p_j+q_i*dp_j) )*factor
            # DERIVATIVE WITH RESPECT TO (xi,eta)
            dp_dxi[ncount]  = dp_dr*dr_dxi
            dp_deta[ncount] = dp_dr*dr_deta + dp_ds

            ncount += 1

    return p,dp_dxi,dp_deta



# 3D - TET
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

@boundscheck(False)
def NormalisedJacobi3D(int C, x):

    """Computes the orthogonal base of 3D polynomials of degree less than
        or equal to n at the point x=(r,s,t) in [-1,1]**3
    """

    cdef int N = (C+2)*(C+3)*(C+4)/6
    cdef double[::1] p = np.zeros(N)

    cdef:
        double r = x[0]
        double s = x[1]
        double t = x[2]
        int nDeg, i, j, k
        double p_i, p_j, p_k, q_i, q_j, q_k
        double factor 

    # ORDERING: FIRST INCREASING THE DEGREE AND THEN LEXOGRAPHIC ORDER
    cdef int ncount = 0
    # LOOP ON DEGREE
    for nDeg in range(0,C+2):
        # LOOP INCREASING I
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1;  q_i = 1 
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1]
                q_i = q_i*(1.-s)/2.
            # LOOP INCREASING J
            for j in range(0,nDeg-i+1):
                if j==0:
                    p_j = 1
                    q_j = ((1.-t)/2.)**i
                else:
                    p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]
                    q_j = q_j*(1.-t)/2.
                # VALUE FOR K
                k = nDeg-(i+j)
                if k==0:
                   p_k = 1.
                else:
                    p_k = JacobiPolynomials(k,t,2.*(i+j)+2.,0.)[-1]
                factor = math.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                p[ncount] = ( p_i*q_i*p_j*q_j*p_k )*factor
                ncount += 1

    return p


def NormalisedJacobiTet(int C,x):
    """Computes the orthogonal base of 3D polynomials of degree less than
        or equal to n at the point x=(xi,eta,zeta) in the reference tetrahedra"""

    cdef:
        double xi = x[0] 
        double eta = x[1] 
        double zeta = x[2]
        double r,s,t

    if (eta+zeta)==0.:
        r = -1.
        s=1.
    elif zeta==1.:
        r = -1. 
        s = 1.  # or s=-1 (check that nothing changes)
    else:
        r = -2.*(1+xi)/(eta+zeta)-1.;
        s = 2.*(1+eta)/(1-zeta)-1.;
    t = zeta

    return NormalisedJacobi3D(C,[r,s,t])
    # return NormalisedJacobi3D_Native(C,[r,s,t])


@boundscheck(False)
def GradNormalisedJacobiTet(int C,x,EvalOpt=0):
    """Computes the orthogonal base of 3D polynomials of degree less than
        or equal to n at the point x=(r,s,t) in [-1,1]**3"""

    cdef:
        int N = (C+2)*(C+3)*(C+4)/6
        double[::1] p = np.zeros(N)
        double[::1] dp_dxi   = np.zeros(N)
        double[::1] dp_deta  = np.zeros(N)
        double [::1] dp_dzeta = np.zeros(N)
        double r = x[0] 
        double s = x[1]
        double t = x[2]
        double xi, eta, zeta, dr_dxi, dr_deta, dr_dzeta, ds_deta, ds_dzeta
        double p_i, p_j, p_k, dp_i, dp_j, dq_i, dq_j
        double dp_dr, dp_ds, dp_dt
        int nDeg, i, j, k
        double factor
        double eta_zeta

    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if EvalOpt==1:
        if t==1.:
            t=0.99999999999
        if np.isclose(s,1.):
            s=0.99999999999

    if np.isclose(s,1.):
            s=0.999999999999

    eta = (1./2.)*(s-s*t-1.-t)
    xi = -(1./2.)*(r+1)*(eta+t)-1.
    zeta = 1.0*t
    
    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if eta == 0. and zeta == 0.:
        eta = 1.0e-14
        zeta = 1e-14

    eta_zeta = eta+zeta
    if np.isclose(eta_zeta,0.):
        eta_zeta = 0.000000001
    # dr_dxi   = -2./(eta+zeta)
    # dr_deta  = 2.*(1.+xi)/(eta+zeta)**2
    dr_dxi   = -2./eta_zeta
    dr_deta  = 2.*(1.+xi)/eta_zeta**2
    dr_dzeta = dr_deta

    ds_deta  = 2./(1.-zeta)
    ds_dzeta = 2.*(1.+eta)/(1.-zeta)**2

    # DERIVATIVE OF t IS NOT REQUIRED AS t=zeta


    # ORDERING: FIRST INCREASING THE DEGREE AND THEN LEXOGRAPHIC ORDER
    cdef int ncount = 0
    # LOOP ON DEGREE
    for nDeg in range(0,C+2):
        # LOOP INCREASING i
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1.
                q_i = 1.
                dp_i = 0.
                dq_i = 0.
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1]
                dp_i = JacobiPolynomials(i-1,r,1.,1.)[-1]*(i+1.)/2.

                q_i = q_i*(1.-s)/2.
                dq_i = q_i*(-i)/(1.-s)
            # LOOP INCREASING j
            for j in range(0,nDeg-i+1):
                if j==0:
                    p_j = 1.
                    q_j = ((1.-t)/2.)**i
                    dp_j = 0.
                    dq_j = q_j*(-(i+j))/(1.-t)  
                else:
                    p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]
                    dp_j = JacobiPolynomials(j-1,s,2.*i+2.,1.)[-1]*(j+2.*i+2.)/2.  
                    
                    q_j = q_j*(1.-t)/2.
                    dq_j = q_j*(-(i+j))/(1.-t)

                # Value for k
                k = nDeg-(i+j);
                if k==0:
                    p_k = 1.;  dp_k = 0.; 
                else:
                    p_k = JacobiPolynomials(k,t,2.*(i+j)+2.,0.)[-1]
                    dp_k = JacobiPolynomials(k-1,t,2.*(i+j)+3.,1.)[-1]*(k+2.*i+2.*j+3.)/2.
        
                factor = math.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                # NORMALISED POLYNOMIAL
                p[ncount] = ( p_i*q_i*p_j*q_j*p_k )*factor
                # Derivatives with respect to (r,s,t)
                dp_dr = ( (dp_i)*q_i*p_j*q_j*p_k )*factor 
                dp_ds = ( p_i*(dq_i*p_j+q_i*dp_j)*q_j*p_k )*factor
                dp_dt = ( p_i*q_i*p_j*(dq_j*p_k+q_j*dp_k) )*factor
                # Derivatives with respect to (xi,eta,zeta)
                dp_dxi[ncount]   = dp_dr*dr_dxi
                dp_deta[ncount]  = dp_dr*dr_deta + dp_ds*ds_deta
                dp_dzeta[ncount] = dp_dr*dr_dzeta + dp_ds*ds_dzeta + dp_dt

                ncount += 1

    return p,dp_dxi,dp_deta,dp_dzeta

