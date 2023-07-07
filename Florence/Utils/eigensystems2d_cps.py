from __future__ import print_function
from sympy import *
from write_eigensystems import write_eigensystem

def get_analytic_eigensystem2d_cps(Psi, fmt="python"):

    I1, I2, I3, J = symbols("I1 I2 I3 J")
    s1, s2 = symbols("s1 s2")

    Psi = Psi.subs(I1, s1 + s2)
    Psi = Psi.subs(I2, s1**2 + s2**2)
    Psi = Psi.subs(I3, s1 * s2)
    Psi = Psi.subs(J, s1 * s2)


    # ------------------------------------------------------------------#
    # Second Piola-Kirchhoff stress tensor
    # [S = V * sigmaS * V^T] where U and V are obtained from SVD of F
    sigmaS = zeros(2,1)
    sigmaS[0] = diff(Psi, s1) / s1
    sigmaS[1] = diff(Psi, s2) / s2

    sigmaS = sigmaS.subs(s1*s2, J)
    sigmaS = sigmaS.subs(s1+s2, I1)
    sigmaS = sigmaS.subs(s1**2+s2**2, I2)

    write_eigensystem(sigmaS, "g", "C", fmt=fmt)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    lambdas = zeros(1,1)
    # flip
    lambdas[0] = (sigmaS[0] - sigmaS[1]) / (s1**2 - s2**2)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    Hw = zeros(2,2)

    # Diagonal entries
    Hw[0,0] = 1 / s1**2 * diff(Psi, s1, 2) - 1 / s1**3 * diff(Psi, s1)
    Hw[1,1] = 1 / s2**2 * diff(Psi, s2, 2) - 1 / s2**3 * diff(Psi, s2)

    # Off-diagonal entries
    Hw[0,1] = diff(diff(Psi, s1), s2) / s1 / s2

    # Symmetrise
    Hw[1,0] = Hw[0,1]
    # ------------------------------------------------------------------#

    Hw = Hw.subs(s1*s2, J)
    Hw = Hw.subs(s1+s2, I1)
    Hw = Hw.subs(s1**2+s2**2, I2)
    # Hw = simplify(Hw)

    write_eigensystem(Hw, "Hw", "C", fmt=fmt)

    # ------------------------------------------------------------------#

    lambdas = simplify(lambdas)
    write_eigensystem(lambdas, "flip", "C", fmt=fmt)




