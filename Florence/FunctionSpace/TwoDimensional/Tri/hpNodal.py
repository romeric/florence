from __future__ import print_function
import numpy as np 
from Florence.QuadratureRules.FeketePointsTri import *
from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
from Florence.FunctionSpace.JacobiPolynomials.JacobiPolynomials import *
from Florence.FunctionSpace.DegenerateMappings import MapXiEta2RS

def hpBases(C, xi, eta, Transform=0, EvalOpt=0, equally_spaced=False):
    """
        Transform:                  transform to from degenrate quad
        EvalOpt:                    evaluate 1 as an approximation 0.9999999
    """

    eps = FeketePointsTri(C)
    if equally_spaced:
        eps = EquallySpacedPointsTri(C)
        
    N = eps.shape[0]
    # Make the Vandermonde matrix
    V = np.zeros((N,N),dtype=np.float64)

    for i in range(0,N):
        x = eps[i,:]
        p1 = NormalisedJacobiTri(C,x)
        V[i,:] = p1



    nsize = int((C+2)*(C+3)/2.)

    Bases = np.zeros((nsize,1))
    gBases = np.zeros((nsize,2))

    # GradNormalisedJacobiTri IS ALWAYS CALLED WITH (QUAD) R,S COORDINATE
    if Transform:
        # IF TRANSFORMATION TO QUAD COORDINATE IS NEEDED
        r, s = MapXiEta2RS(xi,eta)
        p,dp_dxi,dp_deta = GradNormalisedJacobiTri(C,np.array([r,s]),EvalOpt)
    else:
        # IF XI,ETA ARE DIRECTLY GIVEN IN QUAD FORMAT
        p,dp_dxi,dp_deta = GradNormalisedJacobiTri(C,np.array([xi,eta]),EvalOpt)

    Bases = np.linalg.solve(V.T,p)
    gBases[:,0] = np.linalg.solve(V.T,dp_dxi)
    gBases[:,1] = np.linalg.solve(V.T,dp_deta)

    return Bases, gBases



def GradhpBases(C,r,s):
    print('For nodal triangular bases gradient of bases is computed within the hpBases itself')

