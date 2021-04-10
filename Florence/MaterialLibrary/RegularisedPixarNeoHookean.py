import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

class RegularisedPixarNeoHookean(Material):
    """The Neo-Hookean internal energy, used in production in Pixar.

        W(C) = mu/2*(C:I-3) - mu*(J-1) + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(RegularisedPixarNeoHookean, self).__init__(mtype, ndim, **kwargs)

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
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8)
        else:
            delta = 1e-4
        J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))
        # cr = J**2

        mu = self.mu
        lamb = self.lamb
        C_Voigt = 2 * (lamb * (1. - 1./J) -  mu / J) * self.d2CrdCdC(StrainTensors, gcounter) +\
            (mu+lamb)/J**3 * J * np.einsum("ij,kl", self.dCrdC(StrainTensors, gcounter), self.dCrdC(StrainTensors, gcounter))
        C_Voigt = Voigt(C_Voigt,1)
        # print(C_Voigt)
        # exit()


        self.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8)
        else:
            delta = 1e-4
        J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))
        # cr = J**2

        mu = self.mu
        lamb = self.lamb
        stress = mu/J*b + (lamb * (1. - 1./J) -  mu / J) * self.dCrdC(StrainTensors, gcounter)
        # print(stress)
        # exit()

        return stress


    # def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

    #     mu = self.mu
    #     lamb = self.lamb

    #     I = StrainTensors['I']
    #     J = StrainTensors['J'][gcounter]
    #     F = StrainTensors['F'][gcounter]
    #     C = np.dot(F.T,F)

    #     if np.isclose(J, 0) or J < 0:
    #         delta = np.sqrt(0.04 * J * J + 1e-8)
    #     else:
    #         delta = 1e-4
    #     # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

    #     energy  = mu/2.*(trace(C) - 3.) - mu*(J-1) + lamb/2.*(J-1.)**2

    #     return energy



    def dCrdC(self, StrainTensors, gcounter):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8)
        else:
            delta = 1e-4
        J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        c = J**2
        sqrt = np.sqrt
        dcrdc =  0.25*(sqrt(c) + sqrt(c + 4*delta**2))*(1/sqrt(c + 4*delta**2) + 1./sqrt(c))
        dcrdC = dcrdc * c # C-1 factor left out
        # push forward
        dcrdC *= I * 1. / J
        return dcrdC

    def d2CrdCdC(self, StrainTensors, gcounter):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8)
        else:
            delta = 1e-4
        J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        c = J**2
        sqrt = np.sqrt
        d2crdcdc = (sqrt(c) + sqrt(c + 4*delta**2))**2*(0.25*sqrt(c)*(c + 4*delta**2)**2 - 0.125*c*(c + 4*delta**2)**(3/2) + 0.125*(c + 4*delta**2)**(5/2))/(sqrt(c)*(c + 4*delta**2)**3)
        one = d2crdcdc * c
        # push
        one *= 1. / J * np.einsum("ij,kl", I, I)

        # two = self.dCrdC(StrainTensors, gcounter)
        dcrdc =  0.25*(sqrt(c) + sqrt(c + 4*delta**2))*(1/sqrt(c + 4*delta**2) + 1./sqrt(c))
        dcrdC = dcrdc * c # C-1 factor left out
        two = dcrdC * I
        two = 1./J * 0.5 * (np.einsum("ik,jl", two, two) + np.einsum("il,jk", two, two))
        return one - two








