import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class IdealDielectric(Material):
    """Ideal dielectric material model for Laplace problems
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IdealDielectric, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = 1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()
        self.nature = "linear"
        self.fields = "electrostatics"

        if not hasattr(self,'eps_1'):
            if hasattr(self,'eps'):
                self.eps_1 = self.eps

        self.H_VoigtSize = self.ndim

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        # from Florence.MaterialLibrary.LLDispatch._IdealDielectric_ import KineticMeasures
        # D, H_Voigt = KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)
        # return D, None, H_Voigt
        ndim = self.ndim
        eps_1 = float(self.eps_1)
        J = np.linalg.det(F)
        I = np.eye(ndim,ndim)
        D = einsum("i,ij->ij",eps_1/J,ElectricFieldx)[...,None]
        # H_Voigt = np.broadcast_to(1./eps_1*I,(J.shape[0],ndim,ndim))
        # Negative definite inverse is needed due to Legendre transform
        H_Voigt = np.einsum("i,jk",-eps_1/J,I)

        return np.ascontiguousarray(D), None, np.ascontiguousarray(H_Voigt)

    def Hessian(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        eps_1 = self.eps_1
        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        # self.dielectric_tensor = 1./eps_1*I
        # Negative definite inverse is needed due to Legendre transform
        self.dielectric_tensor = -eps_1/J*I
        return self.dielectric_tensor

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        pass

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)
        eps_1 = self.eps_1
        J = StrainTensors['J'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        D_exact = eps_1/J*E
        return D_exact

    def Permittivity(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        eps_1 = self.eps_1
        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        # self.dielectric_tensor = 1./eps_1*I
        # Negative definite inverse is needed due to Legendre transform
        self.dielectric_tensor = -eps_1/J*I
        return self.dielectric_tensor

    def ElectrostaticMeasures(self,F,ElectricFieldx, elem=0):
        return self.KineticMeasures(F,ElectricFieldx, elem)