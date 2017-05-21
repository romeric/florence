import numpy as np
from numpy.linalg import norm


def PointInversionIsoparametricFEM(element_type, C, LagrangeElemCoords, point, equally_spaced=False, tolerance=1.0e-9, maxiter=5):
    """ This is the inverse isoparametric map, in that given a physical point,
        find the isoparametric coordinates, provided that the coordinates of 
        physical element 'LagrangeElemCoords' is known

    input:
        point:                  [2S(3D) vector] containing X,Y,(Z) value of the physical point
    """


    from Florence.FunctionSpace import Tri, Tet, Quad, Hex

    # Initial guess
    p_isoparametric = np.zeros(LagrangeElemCoords.shape[1])

    for niter in range(maxiter):

        # Using the current iterative solution of isoparametric coordinate evaluate bases and gradient of bases
        if element_type == "tet":
            Neval, gradient = Tet.hpNodal.hpBases(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2],True)
        elif element_type == "hex":
            Neval = Hex.LagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2]).flatten()
            gradient = Hex.GradLagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2])
        elif element_type == "tri":
            Neval, gradient = Tri.hpNodal.hpBases(C,p_isoparametric[0],p_isoparametric[1],True,1)
        elif element_type == "quad":
            Neval = Quad.LagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1]).flatten()
            gradient = Quad.GradLagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1])

        # interp_point = np.dot(Neval,LagrangeElemCoords)
        # residual = point - interp_point

        # ParentGradientX = np.dot(gradient.T,LagrangeElemCoords)

        # dp_isoparametric = np.dot(np.linalg.inv(ParentGradientX), residual)
        # p_isoparametric += dp_isoparametric

        # Find the residual X - X(N) [i.e. point - FEM interpolated point]
        residual = point - np.dot(Neval,LagrangeElemCoords)
        # print(point,np.dot(Neval,LagrangeElemCoords))
        # Find the isoparametric (Jacobian) matrix dX/d\Xi
        ParentGradientX = np.dot(gradient.T,LagrangeElemCoords)
        # ParentGradientX = np.fliplr(ParentGradientX)
        # print(Neval)
        # print(np.dot(Neval,LagrangeElemCoords))
        # exit()
        # Solve and update incremental solution
        old_p_isoparametric = np.copy(p_isoparametric)
        p_isoparametric += np.dot(np.linalg.inv(ParentGradientX), residual)
        # Check tolerance 
        # if np.linalg.norm(residual) < tolerance:
        # print(np.linalg.norm(p_isoparametric - old_p_isoparametric) )
        if np.linalg.norm(p_isoparametric - old_p_isoparametric) < tolerance:
            break



    return p_isoparametric