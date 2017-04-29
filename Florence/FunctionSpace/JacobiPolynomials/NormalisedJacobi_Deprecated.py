import numpy as np 
from JacobiPolynomials import *
import math

# 1D - LINE
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
def NormalisedJacobi1D(C,x):
    p = np.zeros(C+2)

    for i in range(0,C+2):
        p[i] = JacobiPolynomials(i,x,0,0)[-1]*np.sqrt((2.*i+1.)/2.)

    return p  


# 2D - TRI
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

def NormalisedJacobi2D(C,x):

    # Computes the ortogonal base of 2D polynomials of degree less 
    # or equal to C+1 at the point x=(r,s) in [-1,1]^2 (i.e. on the reference quad)


    N = int( (C+2.)*(C+3.)/2. ) 
    p = np.zeros(N)

    r = x[0]; s = x[1]

    # Ordering: 1st increasing the degree and 2nd lexicogafic order
    ncount = 0 # counter for the polynomials order
    # Loop on degree
    for nDeg in range(0,C+2):
      # Loop by increasing i
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1.;  q_i = 1.  
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1];     q_i = q_i*(1.-s)/2.
            # Value for j
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


def NormalisedJacobiTri(C,x):   
    # Computes the ortogonal base of 2D polynomials of degree less 
    # or equal to n at the point x=(xi,eta) in the reference triangle

    xi = x[0]; eta = x[1] 
    if eta==1: 
        r = -1.; s=1.;
    else:
        r = 2.*(1+xi)/(1.-eta)-1.
        s = eta

    return NormalisedJacobi2D(C,np.array([r,s])) 



def GradNormalisedJacobiTri(C,x,EvalOpt=0):

    # Computes the ortogonal base of 2D polynomials of degree less 
    # or equal to n at the point x=(r,s) in [-1,1]^2
    
    N = int((C+2.)*(C+3.)/2.) 
    p = np.zeros(N);
    dp_dxi  = np.zeros(N)
    dp_deta = np.zeros(N)

    r = x[0]; s = x[1] 

    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if EvalOpt==1:
        if s==1:
            s=0.99999999999999

    xi = (1.+r)*(1.-s)/2.-1
    eta = s

    dr_dxi  = 2./(1.-eta)
    dr_deta = 2.*(1.+xi)/(1.-eta)**2
    # Derivative of s is not needed because s=eta

    # Ordering: 1st increasing the degree and 2nd lexicogafic order
    ncount = 0
    # Loop on degree
    for nDeg in range(0,C+2):
      # Loop increasing i
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1;  q_i = 1;  dp_i = 0; dq_i = 0
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1]; dp_i = JacobiPolynomials(i-1,r,1.,1.)[-1]*(i+1.)/2.    
                q_i = q_i*(1.-s)/2.; dq_i = 1.*q_i*(-i)/(1-s)
            
            # Value for j
            j = nDeg-i
            if j==0:
                p_j = 1;  dp_j = 0
            else:
                p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]; dp_j = JacobiPolynomials(j-1,s,2.*i+2.,1.)[-1]*(j+2.*i+2.)/2.  
            
            # factor = np.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            factor = math.sqrt( (2.*i+1.)*(i+j+1.)/2. )
            # Normalized polynomial
            p[ncount] = ( p_i*q_i*p_j )*factor
            # Derivatives with respect to (r,s)
            dp_dr = ( (dp_i)*q_i*p_j )*factor
            dp_ds = ( p_i*(dq_i*p_j+q_i*dp_j) )*factor
            # Derivatives with respect to (xi,eta)
            dp_dxi[ncount]  = dp_dr*dr_dxi
            dp_deta[ncount] = dp_dr*dr_deta + dp_ds

            ncount += 1

    return p,dp_dxi,dp_deta



# 3D - TET
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#
#------------------------------------------------------------------------------------------------------------------#

def NormalisedJacobi3D(C,x):

    # Computes the ortogonal base of 3D polynomials of degree less 
    # or equal to n at the point x=(r,s,t) in [-1,1]^3

    N = int((C+2)*(C+3)*(C+4)/6.)
    p = np.zeros(N)

    r = x[0]; s = x[1]; t = x[2]

    # Ordering: 1st incresing the degree and 2nd lexicogafic order
    ncount = 0
    # Loop on degree
    for nDeg in range(0,C+2):
        # Loop increasing i
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1;  q_i = 1 
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1];     q_i = q_i*(1.-s)/2.
            # Loop increasing j
            for j in range(0,nDeg-i+1):
                if j==0:
                    p_j = 1;  q_j = ((1.-t)/2.)**i
                else:
                    p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1];    q_j = q_j*(1.-t)/2.
                # Value for k
                k = nDeg-(i+j)
                if k==0:
                   p_k = 1.
                else:
                    p_k = JacobiPolynomials(k,t,2.*(i+j)+2.,0.)[-1]
                # factor = np.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                factor = math.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                p[ncount] = ( p_i*q_i*p_j*q_j*p_k )*factor
                ncount += 1

    return p


def NormalisedJacobiTet(C,x):

    # Computes the ortogonal base of 3D polynomials of degree less 
    # or equal to n at the point x=(xi,eta,zeta) in the reference tetrahedra


    xi = x[0]; eta = x[1]; zeta = x[2]

    if (eta+zeta)==0:
        r = -1; s=1
    elif zeta==1:
        r = -1; s=1  # or s=-1 (check that nothing changes)
    else:
        r = -2.*(1+xi)/(eta+zeta)-1.;
        s = 2.*(1+eta)/(1-zeta)-1.;
    t = zeta

    return NormalisedJacobi3D(C,[r,s,t])
    # return NormalisedJacobi3D_Native(C,[r,s,t])


def GradNormalisedJacobiTet(C,x,EvalOpt=0):

    # Computes the ortogonal base of 3D polynomials of degree less 
    # or equal to n at the point x=(r,s,t) in [-1,1]^3

    N = int((C+2)*(C+3)*(C+4)/6.)
    p = np.zeros(N)

    dp_dxi   = np.zeros(N)
    dp_deta  = np.zeros(N)
    dp_dzeta = np.zeros(N)

    r = x[0]; s = x[1]; t = x[2]

    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if EvalOpt==1:
        if t==1.:
            t=0.999999999999
        if np.isclose(s,1.):
            s=0.999999999999

    if np.isclose(s,1.):
            s=0.99999999999999

    eta = (1./2.)*(s-s*t-1.-t)
    xi = -(1./2.)*(r+1)*(eta+t)-1.
    zeta = 1.0*t
    
    # THIS MAY RUIN THE CONVERGENCE, BUT FOR POST PROCESSING ITS FINE
    if eta == 0. and zeta == 0.:
        eta = 1.0e-14
        zeta = 1e-14

    dr_dxi   = -2./(eta+zeta)
    dr_deta  = 2.*(1.+xi)/(eta+zeta)**2
    dr_dzeta = dr_deta

    ds_deta  = 2./(1.-zeta)
    ds_dzeta = 2.*(1.+eta)/(1.-zeta)**2

    # Derivative of t is not needed because t=zeta

    #--------------------------------------------------------
    # if np.allclose(eta+zeta,0):
    #     dr_dxi   = -2./(0.001)**2
    #     dr_deta  = 2.*(1.+xi)/(0.001)**2
    # else:
    #     dr_dxi   = -2./(eta+zeta)
    #     dr_deta  = 2.*(1.+xi)/(eta+zeta)**2
    
    # dr_dzeta = dr_deta

    # if np.allclose(eta+zeta,0):
    #     ds_deta = 2./(0.001)
    #     ds_dzeta =  2.*(1.+eta)/(0.001)**2
    # else: 
    #     ds_deta  = 2./(1.-zeta)
    #     ds_dzeta = 2.*(1.+eta)/(1.-zeta)**2
    #--------------------------------------------------------

    # Ordering: 1st increasing the degree and 2nd lexicogafic order
    ncount = 0
    # Loop on degree
    for nDeg in range(0,C+2):
        # Loop increasing i
        for i in range(0,nDeg+1):
            if i==0:
                p_i = 1.;  q_i = 1.;  dp_i = 0.; dq_i = 0.
            else:
                p_i = JacobiPolynomials(i,r,0.,0.)[-1]; dp_i = JacobiPolynomials(i-1,r,1.,1.)[-1]*(i+1.)/2.    
                q_i = q_i*(1.-s)/2.; dq_i = q_i*(-i)/(1.-s)
            # Loop increasing j
            for j in range(0,nDeg-i+1):
                if j==0:
                    p_j = 1;  q_j = ((1.-t)/2.)**i;  dp_j = 0; dq_j = q_j*(-(i+j))/(1.-t);  
                else:
                    p_j = JacobiPolynomials(j,s,2.*i+1.,0.)[-1]; dp_j = JacobiPolynomials(j-1,s,2.*i+2.,1.)[-1]*(j+2.*i+2.)/2.  
                    q_j = q_j*(1.-t)/2.;  dq_j = q_j*(-(i+j))/(1.-t)

                # Value for k
                k = nDeg-(i+j);
                if k==0:
                    p_k = 1.;  dp_k = 0.; 
                else:
                    p_k = JacobiPolynomials(k,t,2.*(i+j)+2.,0.)[-1];  dp_k = JacobiPolynomials(k-1,t,2.*(i+j)+3.,1.)[-1]*(k+2.*i+2.*j+3.)/2.
        
                # factor = np.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                factor = math.sqrt( (2.*i+1.)*(i+j+1.)*(2.*(i+j+k)+3.)/4. )
                # Normalized polynomial
                p[ncount] = ( p_i*q_i*p_j*q_j*p_k )*factor
                # Derivatives with respect to (r,s,t)
                dp_dr = ( (dp_i)*q_i*p_j*q_j*p_k )*factor 
                dp_ds = ( p_i*(dq_i*p_j+q_i*dp_j)*q_j*p_k )*factor
                dp_dt = ( p_i*q_i*p_j*(dq_j*p_k+q_j*dp_k) )*factor
                # if np.isnan(dp_ds):
                    # print p_i, dq_i, p_j, q_i, dp_j, q_j, p_k, factor
                # Derivatives with respect to (xi,eta,zeta)
                dp_dxi[ncount]   = dp_dr*dr_dxi
                dp_deta[ncount]  = dp_dr*dr_deta + dp_ds*ds_deta
                dp_dzeta[ncount] = dp_dr*dr_dzeta + dp_ds*ds_dzeta + dp_dt

                ncount += 1

    return p,dp_dxi,dp_deta,dp_dzeta