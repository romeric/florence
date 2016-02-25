import numpy as np 
from Florence.QuadratureRules.FeketePointsTet import *
from Florence.InterpolationFunctions.JacobiPolynomials import *
from Florence.InterpolationFunctions.DegenerateMappings import MapXiEtaZeta2RST

def hpBases(C,xi,eta,zeta,Transform=0,EvalOpt=0):

    eps = FeketePointsTet(C)
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


    Bases = np.linalg.solve(V.T,p)
    gBases[:,0] = np.linalg.solve(V.T,dp_dxi)
    gBases[:,1] = np.linalg.solve(V.T,dp_deta)
    gBases[:,2] = np.linalg.solve(V.T,dp_dzeta)

    return Bases, gBases



def GradhpBases(C,r,s):
    print 'For nodal tetrahedral bases gradient of bases is computed within the hpBases itself'