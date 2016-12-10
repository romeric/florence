import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
import numpy as np
from math import sqrt
from numpy import einsum
from Florence.Tensor import Voigt, trace, makezero, makezero3d
from Florence.LegendreTransform import LegendreTransform
from Florence import *
from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures



def check_dispatch():

    ndim=3
    ngauss = 5
    mu1,mu2,mu3,lamb,eps_1,eps_2,eps_3,eps_e = 1.,2.,3.,4.,5.,6.,7.,9.
    # material = IsotropicElectroMechanics_106(ndim,mu1=mu1,mu2=mu2,lamb=lamb,eps_1=eps_1,eps_2=eps_2)
    material = AnisotropicMooneyRivlin_1(ndim,mu1=mu1,mu2=mu2,mu3=mu3,lamb=lamb)

    F = np.random.rand(ngauss,ndim,ndim)
    D = np.random.rand(ngauss,ndim)
    E = np.random.rand(ngauss,ndim)
    N = np.random.rand(100,ndim)
    material.anisotropic_orientations = N
    StrainTensors = KinematicMeasures(F,"nonlinear")

    from _AnisotropicMooneyRivlin_1_ import KineticMeasures
    # from _Piezoelectric_100_ import KineticMeasures
    # from _IsotropicElectroMechanics_106_ import KineticMeasures

    stress_py = np.zeros_like(F)
    if ndim==3:
        hessian_py = np.zeros((ngauss,6,6))
    elif ndim==2:
        hessian_py = np.zeros((ngauss,3,3))

    for g in range(ngauss):
        stress_py[g,:,:] = material.CauchyStress(StrainTensors,elem=0,gcounter=g)
        hessian_py[g,:,:] = material.Hessian(StrainTensors,elem=0,gcounter=g)
    
    stress_cpp, hessian_cpp = KineticMeasures(material,F,N[0][:,None])
    makezero3d(stress_cpp)
    makezero3d(hessian_cpp)
    makezero3d(hessian_py)
    makezero3d(hessian_cpp)

    # print stress_py - stress_cpp
    # print hessian_py - hessian_cpp
    print np.allclose(hessian_cpp,hessian_py)
    print np.allclose(stress_cpp,stress_py)



check_dispatch()
