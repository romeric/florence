from __future__ import print_function
from .write_eigensystems import write_eigensystem
from sympy import *

def get_analytic_eigensystem3d_fps(Psi, fmt="python"):

    I1, I2, I3, J = symbols("I1 I2 I3 J")
    s1, s2, s3 = symbols("s1 s2 s3")

    Psi = Psi.subs(I1, s1 + s2 + s3)
    Psi = Psi.subs(I2, s1**2 + s2**2 + s3**2)
    Psi = Psi.subs(I3, s1 * s2 * s3)
    Psi = Psi.subs(J, s1 * s2 * s3)

    # ------------------------------------------------------------------#
    # First Piola-Kirchhoff stress tensor
    # [P = U * sigmaP * V^T] where U and V are obtained from SVD of F
    sigmaP = zeros(3,1)
    sigmaP[0] = diff(Psi, s1)
    sigmaP[1] = diff(Psi, s2)
    sigmaP[2] = diff(Psi, s3)

    sigmaP = sigmaP.subs(s1+s2+s3, I1)
    sigmaP = sigmaP.subs(s1**2+s2**2+s3**2, I2)
    sigmaP = sigmaP.subs(s1*s2*s3, J)

    write_eigensystem(sigmaP, "g", "F", fmt=fmt)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    lambdas = zeros(9,1)
    # x-flip, y-flip, and z-flip
    lambdas[3] = (diff(Psi, s2) - diff(Psi, s3)) / (s2 - s3)
    lambdas[4] = (diff(Psi, s1) - diff(Psi, s3)) / (s1 - s3)
    lambdas[5] = (diff(Psi, s1) - diff(Psi, s2)) / (s1 - s2)

    # x-twist, y-twist, and z-twist
    lambdas[6] = (diff(Psi, s2) + diff(Psi, s3)) / (s2 + s3)
    lambdas[7] = (diff(Psi, s1) + diff(Psi, s3)) / (s1 + s3)
    lambdas[8] = (diff(Psi, s1) + diff(Psi, s2)) / (s1 + s2)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    Hw = zeros(3,3)

    # Diagonal entries
    Hw[0,0] = diff(Psi, s1, 2)
    Hw[1,1] = diff(Psi, s2, 2)
    Hw[2,2] = diff(Psi, s3, 2)

    # Off-diagonal entries
    Hw[0,1] = diff(diff(Psi, s1), s2)
    Hw[0,2] = diff(diff(Psi, s1), s3)
    Hw[1,2] = diff(diff(Psi, s2), s3)

    # Symmetrise
    Hw[1,0] = Hw[0,1]
    Hw[2,0] = Hw[0,2]
    Hw[2,1] = Hw[1,2]

    lambdas[0] = Hw[0,0]
    lambdas[1] = Hw[1,1]
    lambdas[2] = Hw[2,2]
    # ------------------------------------------------------------------#

    Hw = Hw.subs(s1+s2+s3, I1)
    Hw = Hw.subs(s1**2+s2**2+s3**2, I2)
    Hw = Hw.subs(s1*s2*s3, J)
    # Hw = simplify(Hw)

    # Get compact expressions for the eigenvalues
    if Hw[0,1] == 0 and Hw[0,2] == 0 and  Hw[1,2] == 0:
        for i in range(0, 9):
            lambdas[i] = lambdas[i].subs(s1*s2*s3, J)
            lambdas[i] = lambdas[i].subs(s1+s2+s3, I1)
            lambdas[i] = lambdas[i].subs(s1**2+s2**2+s3**2, I2)
            lambdas[i] = simplify(lambdas[i])

        write_eigensystem(lambdas, "full", "F", fmt=fmt)

    else:

        # eigenvalues = Hw.eigenvals().items()
        # # print(eigenvalues)
        # lambdas[0] = eigenvalues[0][0]
        # lambdas[1] = eigenvalues[1][0]
        # lambdas[2] = eigenvalues[2][0]

        write_eigensystem(Hw, "Hw", "F", fmt=fmt)

        # ------------------------------------------------------------------#
        for i in range(3, 9):

            lambdas[i] = lambdas[i].subs(s1*s2*s3, J)
            lambdas[i] = lambdas[i].subs(s1+s2+s3, I1)
            lambdas[i] = lambdas[i].subs(s1**2+s2**2+s3**2, I2)

            lambdas[i] = simplify(lambdas[i])

            lambdas[i] = lambdas[i].subs(s1*s2*s3, J)
            lambdas[i] = lambdas[i].subs(s1+s2+s3, I1)
            lambdas[i] = lambdas[i].subs(s1**2+s2**2+s3**2, I2)

        lambdas_is = zeros(6, 1)
        lambdas_is[:,0] = lambdas[3:,0]
        write_eigensystem(lambdas_is, "IS", "F", fmt=fmt)


