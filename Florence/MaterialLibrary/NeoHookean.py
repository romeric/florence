import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace

class NeoHookean(Material):
    """The fundamental Neo-Hookean internal energy, described in Bonet et. al.

        W(C) = mu/2*(C:I-3)- mu*lnJ + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NeoHookean, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"  

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)
        

    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):
        
        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]

        mu2 = self.mu/J- self.lamb*(J-1.0)
        lamb2 = self.lamb*(2*J-1.0) 

        H_Voigt = lamb2*self.vIijIkl+mu2*self.vIikIjl

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        mu = self.mu
        lamb = self.lamb
            
        return 1.0*mu/J*b + (lamb*(J-1.0)-mu/J)*I



    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        energy  = mu/2.*(trace(C) - 3.) - mu*np.log(J) + lamb/2.*(J-1.)**2

        return energy


