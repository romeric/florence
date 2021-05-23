import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class DistortionModel(Material):
    """The fundamental DistortionModel model

        W_(C) = F:F / d / J**(2/d)

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(DistortionModel, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._MooneyRivlin_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8)
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))
            # J = .1

        trb = trace(b) + 0

        C_Voigt = 4./d * J**(-1) * einsum("ij,kl", (-1./d * trb * I + b), (-1./d) * J**(-2./d) * I ) +\
            4./d**2 * J**(-1) * J**(-2./d) * (-einsum("ij,kl", I, b) + 0.5 * trb * (einsum("ik,jl",I,I)+einsum("il,jk",I,I)) )

        # factor = np.exp(1./d * J**(-2./d) * trb - 1.)
        # sigma = 2./d * J**(-2./d - 1.) * (-1./d * trb * I + b)
        # C_Voigt = factor * (np.einsum("ij,kl", sigma, sigma) + C_Voigt)

        C_Voigt = Voigt(C_Voigt,1)

        # s = np.linalg.svd(C_Voigt)[1]
        # if np.any(s < 0):
        #     print(s)

        return C_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))
            # J = .1

        trb = trace(b) + 0

        sigma = 2./d * J**(-2./d - 1.) * (-1./d * trb * I + b)
        # sigma *= np.exp(1./d * J**(-2./d) * trb - 1)

        return sigma


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        d = self.ndim

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))
            # J = .1

        # Adding a dummy term to energy to disallow sign switch in backtracking linesearch
        # This energy is negative iff J is negative
        # energy  = (1./d * J**(-2./d) * trace(b) - 1.)
        # energy  = (1./d * J**(-2./d) * trace(b) - 0.) #+ 1e1
        energy  = (1./d * J**(-2./d) * trace(b)) + 1.

        return energy
