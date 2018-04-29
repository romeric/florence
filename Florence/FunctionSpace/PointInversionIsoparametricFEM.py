import numpy as np
from numpy.linalg import norm

__all__ = ["PointInversionIsoparametricFEM"]


def hpBasesTetSimple(eta,zeta,xi):
    # THESE BASES ARE A BIT NUMERICALLY STABLE BECAUSE THEY ARE NOT BEING ROUNDED OFF
    # LIKE THE ONES COMING FROM Jacobi - ALTHOUGH LOWERING THE TOLERANCE FOR Jacobi GIVES THE SAME
    # MOREOVER THESE ARE ONLY FOR C=0

    # from Florence.FunctionSpace.DegenerateMappings import MapXiEtaZeta2RST
    # eta, zeta, xi = MapXiEtaZeta2RST(eta,zeta,xi)
    # eta, zeta, xi = eta[0], zeta[0], xi[0]
    N = np.array([  1.-eta-zeta-xi,
                    eta,
                    zeta,
                    xi
        ])

    dN = np.array([
        [-1.,-1.,-1.],
        [1.,0.,0.],
        [0.,1.,0.],
        [0.,0.,1.],
        ])

    return N, dN


def PointInversionIsoparametricFEM(element_type, C, LagrangeElemCoords, point,
    equally_spaced=False, tolerance=1.0e-7, maxiter=20, verbose=False, use_simple_bases=False, initial_guess=None):
    """ This is the inverse isoparametric map, in that given a physical point,
        find the isoparametric coordinates, provided that the coordinates of
        physical element 'LagrangeElemCoords' is known

        input:
            point:                  [2S(3D) vector] containing X,Y,(Z) value of the physical point
        return:
            iso_parametric_point    [2S(3D) vector] containing X,Y,(Z) value of the parametric point
            convergence_info        [boolean] if the iteration converged or not
    """


    from Florence.FunctionSpace import Tri, Tet, Quad, QuadES, Hex, HexES
    from Florence.Tensor import makezero

    # INITIAL GUESS - VERY IMPORTANT
    if initial_guess is None:
        # p_isoparametric = np.zeros(LagrangeElemCoords.shape[1])
        p_isoparametric = -np.ones(LagrangeElemCoords.shape[1])
    else:
        p_isoparametric = initial_guess
    residual = 0.
    blow_up_value = 1e10
    convergence_info = False

    for niter in range(maxiter):

        # Using the current iterative solution of isoparametric coordinate evaluate bases and gradient of bases
        if element_type == "tet":
            if not use_simple_bases:
                Neval, gradient = Tet.hpNodal.hpBases(C,p_isoparametric[0],p_isoparametric[1],
                    p_isoparametric[2], False, 1, equally_spaced=equally_spaced)
            else:
                Neval, gradient = hpBasesTetSimple(p_isoparametric[0],p_isoparametric[1],p_isoparametric[2])
        elif element_type == "hex":
            if not equally_spaced:
                Neval = Hex.LagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2]).flatten()
                gradient = Hex.GradLagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2])
            else:
                Neval = HexES.Lagrange(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2]).flatten()
                gradient = HexES.GradLagrange(C,p_isoparametric[0],p_isoparametric[1],p_isoparametric[2])
        elif element_type == "tri":
            Neval, gradient = Tri.hpNodal.hpBases(C,p_isoparametric[0],p_isoparametric[1], False, 1, equally_spaced=equally_spaced)
        elif element_type == "quad":
            if not equally_spaced:
                Neval = Quad.LagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1]).flatten()
                gradient = Quad.GradLagrangeGaussLobatto(C,p_isoparametric[0],p_isoparametric[1])
            else:
                Neval = QuadES.Lagrange(C,p_isoparametric[0],p_isoparametric[1]).flatten()
                gradient = QuadES.GradLagrange(C,p_isoparametric[0],p_isoparametric[1])
        makezero(np.atleast_2d(Neval))
        makezero(gradient)

        # Find the residual X - X(N) [i.e. point - FEM interpolated point]
        residual = point - np.dot(Neval,LagrangeElemCoords)
        # print(np.dot(Neval,LagrangeElemCoords))
        # Find the isoparametric (Jacobian) matrix dX/d\Xi
        ParentGradientX = np.dot(gradient.T,LagrangeElemCoords)
        # ParentGradientX = np.fliplr(ParentGradientX)
        # Solve and update incremental solution
        old_p_isoparametric = np.copy(p_isoparametric)
        # print(p_isoparametric)
        try:
            # p_isoparametric += np.dot(np.linalg.inv(ParentGradientX), residual)
            p_isoparametric += np.linalg.solve(ParentGradientX, residual)
        except:
            convergence_info = False
            break

        # CHECK IF WITHIN TOLERANCE
        if norm(p_isoparametric - old_p_isoparametric) < tolerance:
            convergence_info = True
            break
        # print(np.linalg.norm(residual))
        if norm(residual) < tolerance*1e2:
            convergence_info = True
            break
        # OTHERWISE DEAL WITH BLOW UP SITUATION
        if np.isnan(norm(residual)) > blow_up_value or norm(residual) > blow_up_value:
            break
        if np.isnan(norm(p_isoparametric)) or norm(p_isoparametric) > blow_up_value:
            break

    if verbose:
        return p_isoparametric, convergence_info
    return p_isoparametric