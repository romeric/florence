import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

class NeoHookeanBSmith(Material):
    """The fundamental Neo-Hookean internal energy, described in B. Smith et. al.

        W(C) = mu/2*(C:I-3) + lamb/2*(J - alpha)**2 - mu/2*ln(C:I + 1)

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NeoHookeanBSmith, self).__init__(mtype, ndim, **kwargs)

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

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        b = StrainTensors['b'][gcounter]
        trb = np.trace(b)
        if self.ndim==2:
            trb += 1
        delta = 1.
        alpha = 1 + 3./4. * mu / lamb
        C_Voigt = 2. * mu / J / (trb + delta)**2 * np.einsum("ij,kl", b, b) + 2 * lamb * J * (1. - alpha/2./J) * np.einsum("ij,kl", I, I) -\
                    lamb * (J - alpha) * (np.einsum("ik,jl", I, I)  + np.einsum("il,jk", I, I) )
        C_Voigt = Voigt(C_Voigt,1)

        self.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb

        trb = np.trace(b)
        if self.ndim==2:
            trb += 1
        delta = 1.
        alpha = 1 + 3./4. * mu / lamb
        stress = mu / J * (1. - 1./(trb + delta)) * b + lamb * (J - alpha) * I

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        alpha = 1 + 3./4. * mu / lamb
        energy  = mu/2.*(trace(C) - 3.) - mu/2.*np.log(trace(C) + 1) + lamb/2.*(J-alpha)**2

        return energy


