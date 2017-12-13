import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
#####################################################################################################
                        # Electrostatic model in terms of internal energy
                        # W(C,D) = 1/2/eps_2/J (D0*CD0)
#####################################################################################################


class IdealDielectric(Material):

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IdealDielectric, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = 1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()

        self.H_VoigtSize = self.ndim

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._IdealDielectric_ import KineticMeasures
        return KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)

    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):
        I = StrainTensors['I']
        eps_1 = self.eps_1
        self.dielectric_tensor = 1./eps_1*I
        return self.dielectric_tensor

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)

        # SANITY CHECK FOR IMPLICIT COMPUTATUTAION OF D
        eps_1 = self.eps_1
        J = StrainTensors['J'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        D_exact = eps_1/J*E
        return D_exact