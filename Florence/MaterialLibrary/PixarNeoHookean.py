import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

class PixarNeoHookean(Material):
    """The Neo-Hookean internal energy, used in production in Pixar.

        W(C) = mu/2*(C:I-3) - mu*(J-a) + lamb/2*(J-a)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(PixarNeoHookean, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        if not hasattr(self, "alpha"):
            self.alpha = 1.0

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
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        alpha = self.alpha
        C_Voigt = (lamb * (2*J-alpha) - mu) * np.einsum("ij,kl",I,I) + (mu - lamb * (J-alpha))  * (np.einsum("ik,jl",I,I) + np.einsum("il,jk",I,I))
        C_Voigt = Voigt(C_Voigt,1)

        self.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        alpha = self.alpha
        stress = mu/J*b + (lamb*(J - alpha) - mu)*I

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        energy  = mu/2.*(trace(C) - 3.) - mu*(J-1) + lamb/2.*(J-1.)**2

        return energy
