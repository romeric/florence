from eigensystems2d_fps import get_analytic_eigensystem2d_fps
from eigensystems3d_fps import get_analytic_eigensystem3d_fps
from eigensystems2d_cps import get_analytic_eigensystem2d_cps
from eigensystems3d_cps import get_analytic_eigensystem3d_cps

def get_analytic_eigensystem(Psi, ndim, formulation, fmt="python"):

    if ndim == 2 and formulation == "F":
        get_analytic_eigensystem2d_fps(Psi, fmt=fmt)
    elif ndim == 3 and formulation == "F":
        get_analytic_eigensystem3d_fps(Psi, fmt=fmt)
    elif ndim == 2 and formulation == "C":
        get_analytic_eigensystem2d_cps(Psi, fmt=fmt)
    elif ndim == 3 and formulation == "C":
        get_analytic_eigensystem3d_cps(Psi, fmt=fmt)

