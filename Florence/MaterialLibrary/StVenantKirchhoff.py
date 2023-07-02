import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

class StVenantKirchhoff(Material):
    """The fundamental StVenantKirchhoff internal energy

        W(C) = mu/4*(C-I):(C-I) + lamb/4 tr(C-I)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(StVenantKirchhoff, self).__init__(mtype, ndim, **kwargs)

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
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb

        C_Voigt = lamb * np.einsum("ij,kl",I,I) + mu * (np.einsum("ik,jl",I,I) + np.einsum("il,jk",I,I))
        C_Voigt = 1./J * np.einsum("iI,jJ,kK,lL,IJKL",F,F,F,F,C_Voigt)
        # Same here
        # C_Voigt = lamb/J * np.einsum("ij,kl",b,b) + mu/J * (np.einsum("ik,jl",b,b) + np.einsum("il,jk",b,b))
        C_Voigt = Voigt(C_Voigt,1)

        self.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb
        E = 0.5 * (C - I)
        stress = lamb * trace(E) * I + 2. * mu * E
        stress = 1./J * np.dot(F,np.dot(stress,F.T))

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb

        energy  = mu/4*np.einsum("ij,ij",C-I,C-I) + lamb/8 * trace(C-I)**2

        return energy
