from __future__ import print_function
from sympy import *
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)

def write_eigensystem(A, wtype, formulation="F", fmt="python"):
    """
        wtype: g (for gradient), Hw, IS, full (for full tangent with decoupled eigenvalues), flip for C-based constitutive flip modes
        fmt: python, cxx, latex
        formulation: F, C
    """

    # Writer func
    def writefunc(x):
        if fmt == "latex":
            return latex(x)
        elif fmt == "cxx":
            return cxxcode(x, standard="C++17")
        elif fmt == "python":
            return x

    if formulation == "F":
        if wtype == "g":
            if A.shape[0] == 3:
                print("PK1 principal components:")
                if fmt == "python" or fmt == "cxx":
                    print("sigmaP[0] = ", writefunc(A[0]))
                    print("sigmaP[1] = ", writefunc(A[1]))
                    print("sigmaP[2] = ", writefunc(A[2]))
                else:
                    print("\\lambda_{P_{11}} = ", writefunc(A[0]))
                    print("\\lambda_{P_{22}} = ", writefunc(A[1]))
                    print("\\lambda_{P_{33}} = ", writefunc(A[2]))
            if A.shape[0] == 2:
                print("PK1 principal components:")
                if fmt == "python" or fmt == "cxx":
                    print("sigmaP[0] = ", writefunc(A[0]))
                    print("sigmaP[1] = ", writefunc(A[1]))
                else:
                    print("\\lambda_{P_{11}} = ", writefunc(A[0]))
                    print("\\lambda_{P_{22}} = ", writefunc(A[1]))

        elif wtype == "Hw":
            print("d x d Hessian (Hw) has coupled eigenvalues")
            print("d x d Hessian (Hw) matrix:")
            if fmt == "python" or fmt == "latex":
                if A.shape[1] == 3:
                    print("a11 = ", writefunc(A[0,0]))
                    print("a22 = ", writefunc(A[1,1]))
                    print("a33 = ", writefunc(A[2,2]))
                    print("a12 = ", writefunc(A[0,1]))
                    print("a13 = ", writefunc(A[0,2]))
                    print("a23 = ", writefunc(A[1,2]))
                elif A.shape[1] == 2:
                    print("a11 = ", writefunc(A[0,0]))
                    print("a22 = ", writefunc(A[1,1]))
                    print("a12 = ", writefunc(A[0,1]))

            elif fmt == "cxx":
                if A.shape[1] == 3:
                    print("const Real a11 = ", writefunc(A[0,0]), ";")
                    print("const Real a22 = ", writefunc(A[1,1]), ";")
                    print("const Real a33 = ", writefunc(A[2,2]), ";")
                    print("const Real a12 = ", writefunc(A[0,1]), ";")
                    print("const Real a13 = ", writefunc(A[0,2]), ";")
                    print("const Real a23 = ", writefunc(A[1,2]), ";")
                elif A.shape[1] == 2:
                    print("const Real a11 = ", writefunc(A[0,0]),";")
                    print("const Real a22 = ", writefunc(A[1,1]),";")
                    print("const Real a12 = ", writefunc(A[0,1]),";")

        elif wtype == "full":
            if A.shape[0] == 9 or A.shape[0] == 4:
                print("d x d Hessian (Hw) has decoupled eigenvalues")
                print("Full eigensystem: first 3 are for tangent stiffness and last 6 for initial stiffness:")
                for i in range(0, A.shape[0]):
                    if fmt == "python":
                        print("lamb"+str(i+1)+" = ", writefunc(A[i]))
                    if fmt == "cxx":
                        print("lambda_"+str(i+1)+" = ", writefunc(A[i]))
                    if fmt == "latex":
                        print("\\lambda_"+str(i+1)+" = ", writefunc(A[i]))

        elif wtype == "IS":
            if A.shape[0] == 6 or A.shape[0] == 2:
                offset = 3 if A.shape[0] == 2 else 4
                print("Initial stiffness eigenvalues:")
                for i in range(0, A.shape[0]):
                    if fmt == "python":
                        print("lamb"+str(i+offset)+" = ", writefunc(A[i]))
                    if fmt == "cxx":
                        print("lambda_"+str(i+offset)+" = ", writefunc(A[i]))
                    if fmt == "latex":
                        print("\\lambda_"+str(i+offset)+" = ", writefunc(A[i]))


    elif formulation == "C":
        if wtype == "g":
            if A.shape[0] == 3:
                print("PK2 principal components:")
                if fmt == "python" or fmt == "cxx":
                    print("sigmaS[0] = ", writefunc(A[0]))
                    print("sigmaS[1] = ", writefunc(A[1]))
                    print("sigmaS[2] = ", writefunc(A[2]))
                else:
                    print("\\lambda_{S_{11}} = ", writefunc(A[0]))
                    print("\\lambda_{S_{22}} = ", writefunc(A[1]))
                    print("\\lambda_{S_{33}} = ", writefunc(A[2]))
            if A.shape[0] == 2:
                print("PK2 principal components:")
                if fmt == "python" or fmt == "cxx":
                    print("sigmaS[0] = ", writefunc(A[0]))
                    print("sigmaS[1] = ", writefunc(A[1]))
                else:
                    print("\\lambda_{S_{11}} = ", writefunc(A[0]))
                    print("\\lambda_{S_{22}} = ", writefunc(A[1]))

        elif wtype == "Hw":
            print("d x d Hessian (Hw) has coupled eigenvalues")
            print("d x d Hessian (Hw) matrix:")
            if fmt == "python" or fmt == "latex":
                if A.shape[1] == 3:
                    print("a11 = ", writefunc(A[0,0]))
                    print("a22 = ", writefunc(A[1,1]))
                    print("a33 = ", writefunc(A[2,2]))
                    print("a12 = ", writefunc(A[0,1]))
                    print("a13 = ", writefunc(A[0,2]))
                    print("a23 = ", writefunc(A[1,2]))
                elif A.shape[1] == 2:
                    print("a11 = ", writefunc(A[0,0]))
                    print("a22 = ", writefunc(A[1,1]))
                    print("a12 = ", writefunc(A[0,1]))

            elif fmt == "cxx":
                if A.shape[1] == 3:
                    print("const Real a11 = ", writefunc(A[0,0]), ";")
                    print("const Real a22 = ", writefunc(A[1,1]), ";")
                    print("const Real a33 = ", writefunc(A[2,2]), ";")
                    print("const Real a12 = ", writefunc(A[0,1]), ";")
                    print("const Real a13 = ", writefunc(A[0,2]), ";")
                    print("const Real a23 = ", writefunc(A[1,2]), ";")
                elif A.shape[1] == 2:
                    print("const Real a11 = ", writefunc(A[0,0]),";")
                    print("const Real a22 = ", writefunc(A[1,1]),";")
                    print("const Real a12 = ", writefunc(A[0,1]),";")

        elif wtype == "flip":
            if A.shape[0] == 3 or A.shape[0] == 1:
                offset = 3 if A.shape[0] == 1 else 4
                print("The flip mode constitutive eigenvalues:")
                for i in range(0, A.shape[0]):
                    if fmt == "python":
                        print("lamb"+str(i+offset)+" = ", writefunc(A[i]))
                    elif fmt == "cxx":
                        print("lambda_"+str(i+offset)+" = ", writefunc(A[i]))
                    elif fmt == "latex":
                        print("\\lambda_"+str(i+offset)+" = ", writefunc(A[i]))








