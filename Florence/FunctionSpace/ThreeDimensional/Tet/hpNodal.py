from __future__ import print_function
import numpy as np 
from Florence.QuadratureRules.FeketePointsTet import *
from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTet
from Florence.FunctionSpace.JacobiPolynomials import *
from Florence.FunctionSpace.DegenerateMappings import MapXiEtaZeta2RST

def hpBases(C, xi, eta, zeta, Transform=0, EvalOpt=0, equally_spaced=False):
    """
        Transform:                  transform to from degenrate quad
        EvalOpt:                    evaluate 1 as an approximation 0.9999999
    """

    eps = FeketePointsTet(C)
    if equally_spaced:
        eps = EquallySpacedPointsTet(C)

    N = eps.shape[0]
    # Make the Vandermonde matrix
    V = np.zeros((N,N))

    for i in range(0,N):
        x = eps[i,:]
        p = NormalisedJacobiTet(C,x)
        V[i,:] = p

    

    nsize = int((C+2.)*(C+3.)*(C+4.)/6.)

    Bases = np.zeros((nsize,1))
    gBases = np.zeros((nsize,3))

    # GradNormalisedJacobiTet IS ALWAYS CALLED WITH (HEX) R,S,T COORDINATE
    if Transform:
        # IF A TRANSFORMATION TO HEX COORDINATE IS NEEDED
        r, s, t = MapXiEtaZeta2RST(xi,eta,zeta)
        p, dp_dxi, dp_deta, dp_dzeta = GradNormalisedJacobiTet(C,np.array([r,s,t]),EvalOpt)
    else:
        # IF XI,ETA,ZETA ARE DIRECTLY GIVEN IN HEX FORMAT
        p, dp_dxi, dp_deta, dp_dzeta = GradNormalisedJacobiTet(C,np.array([xi,eta,zeta]),EvalOpt)

    # ACTUAL WAY
    # Bases = np.linalg.solve(V.T,p)
    # gBases[:,0] = np.linalg.solve(V.T,dp_dxi)
    # gBases[:,1] = np.linalg.solve(V.T,dp_deta)
    # gBases[:,2] = np.linalg.solve(V.T,dp_dzeta)

    # SOLVE FOR MULTIPLE RIGHT HAND SIDES
    # ps = np.concatenate((p[:,None],dp_dxi[:,None],dp_deta[:,None],dp_dzeta[:,None]),axis=1)
    ps = np.concatenate((p,dp_dxi,dp_deta,dp_dzeta)).reshape(4,p.shape[0]).T
    tup = np.linalg.solve(V.T,ps)
    Bases = tup[:,0]
    gBases[:,0] = tup[:,1]
    gBases[:,1] = tup[:,2]
    gBases[:,2] = tup[:,3]


    return Bases, gBases



def GradhpBases(C,r,s):
    print('For nodal tetrahedral bases gradient of bases is computed within the hpBases itself')