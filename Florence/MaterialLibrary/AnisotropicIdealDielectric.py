import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class AnisotropicIdealDielectric(Material):
    """Ideal dielectric material model for Laplace problems
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(AnisotropicIdealDielectric, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = 1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()
        self.nature = "linear"
        self.fields = "electrostatics"

        self.H_VoigtSize = self.ndim

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self, F, ElectricFieldx, elem=0):
        # from Florence.MaterialLibrary.LLDispatch._AnisotropicIdealDielectric_ import KineticMeasures
        # D, H_Voigt = KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)
        # return D, None, H_Voigt
        ndim = self.ndim
        e = self.e # dielectric tensor
        J = np.linalg.det(F)
        I = np.eye(ndim,ndim)
        et = np.tile(e,(ElectricFieldx.shape[0],1)).reshape(ElectricFieldx.shape[0],e.shape[0],e.shape[0])
        eJ = np.einsum("ijk,i->ijk",et,1./J)
        D = np.einsum("ijk,ik->ij",eJ,ElectricFieldx)[...,None]
        # Negative definite inverse is needed due to Legendre transform
        H_Voigt = -eJ

        return np.ascontiguousarray(D), None, np.ascontiguousarray(H_Voigt)

    def Hessian(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        e = self.e
        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        # self.dielectric_tensor = 1./eps_1*I
        # Negative definite inverse is needed due to Legendre transform
        self.dielectric_tensor = -e/J
        return self.dielectric_tensor

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        pass

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)
        e = self.e
        J = StrainTensors['J'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        D_exact = np.dot(e/J,E)
        return D_exact

    def Permittivity(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        e = self.e
        J = StrainTensors['J'][gcounter]
        # Negative definite inverse is needed due to Legendre transform
        self.dielectric_tensor = -e/J
        return self.dielectric_tensor

    def ElectrostaticMeasures(self,F,ElectricFieldx, elem=0):
        return self.KineticMeasures(F,ElectricFieldx, elem)