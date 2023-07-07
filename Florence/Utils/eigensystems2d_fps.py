from __future__ import print_function
from sympy import *
from write_eigensystems import write_eigensystem

def get_analytic_eigensystem2d_fps(Psi, fmt="python"):

    I1, I2, I3, J = symbols("I1 I2 I3 J")
    s1, s2 = symbols("s1 s2")

    Psi = Psi.subs(I1, s1 + s2)
    Psi = Psi.subs(I2, s1**2 + s2**2)
    Psi = Psi.subs(I3, s1 * s2)
    Psi = Psi.subs(J, s1 * s2)


    # ------------------------------------------------------------------#
    # First Piola-Kirchhoff stress tensor
    # [P = U * sigmaP * V^T] where U and V are obtained from SVD of F
    sigmaP = zeros(2,1)
    sigmaP[0] = diff(Psi, s1)
    sigmaP[1] = diff(Psi, s2)

    sigmaP = sigmaP.subs(s1*s2, J)
    sigmaP = sigmaP.subs(s1+s2, I1)
    sigmaP = sigmaP.subs(s1**2+s2**2, I2)

    write_eigensystem(sigmaP, "g", "F", fmt=fmt)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    lambdas = zeros(4,1)
    # flip
    lambdas[2] = (diff(Psi, s1) - diff(Psi, s2)) / (s1 - s2)

    # twist
    lambdas[3] = (diff(Psi, s1) + diff(Psi, s2)) / (s1 + s2)
    # ------------------------------------------------------------------#

    # ------------------------------------------------------------------#
    Hw = zeros(2,2)

    # Diagonal entries
    Hw[0,0] = diff(Psi, s1, 2)
    Hw[1,1] = diff(Psi, s2, 2)

    # Off-diagonal entries
    Hw[0,1] = diff(diff(Psi, s1), s2)

    # Symmetrise
    Hw[1,0] = Hw[0,1]

    lambdas[0] = Hw[0,0]
    lambdas[1] = Hw[1,1]
    # ------------------------------------------------------------------#

    # Hw = Hw.subs(I3, J)
    # Hw = Hw.subs(s1*s2, I3)
    Hw = Hw.subs(s1*s2, J)
    Hw = Hw.subs(s1+s2, I1)
    Hw = Hw.subs(s1**2+s2**2, I2)
    # A = simplify(A) # this is on but commented for complex beta

    # Get compact expressions for the eigenvalues
    if Hw[0,1] == 0:
        for i in range(0,4):
            lambdas[i] = lambdas[i].subs(s1*s2, J)
            lambdas[i] = lambdas[i].subs(s1+s2, I1)
            lambdas[i] = lambdas[i].subs(s1**2+s2**2, I2)
            lambdas[i] = simplify(lambdas[i])

        write_eigensystem(lambdas, "full", "F", fmt=fmt)

    else:

        # Hw = Hw.subs(I1, sigma0+sigma1+sigma2)
        # Hw = Hw.subs(I2, sigma0**2 + sigma1**2 + sigma2**2)
        # Hw = Hw.subs(J, sigma0*sigma1*sigma2)
        # Hw = simplify(Hw)

        # eigenvalues = Hw.eigenvals().items()
        # print(eigenvalues)
        # lambdas[0] = eigenvalues[0][0]
        # lambdas[1] = eigenvalues[1][0]

        write_eigensystem(Hw, "Hw", "F", fmt=fmt)

        # ------------------------------------------------------------------#
        for i in range(2, 4):

            lambdas[i] = lambdas[i].subs(s1*s2, J)
            lambdas[i] = lambdas[i].subs(s1+s2, I1)
            lambdas[i] = lambdas[i].subs(s1**2+s2**2, I2)

            lambdas[i] = simplify(lambdas[i])

        lambdas_is = zeros(2, 1)
        lambdas_is[:,0] = lambdas[2:,0]
        write_eigensystem(lambdas_is, "IS", "F", fmt=fmt)





