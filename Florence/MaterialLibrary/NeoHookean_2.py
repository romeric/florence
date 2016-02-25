import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace


#####################################################################################################
                                        # NeoHookean Material Model 2
#####################################################################################################


class NeoHookean_2(Material):
    """Material model for neo-Hookean with the following internal energy:

        W(C) = mu/2*(C:I)-mu*lnJ+lamba/2*(J-1)**2

        """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NeoHookean_2, self).__init__(mtype, ndim, **kwargs)

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"Nonlinear")
        self.Hessian(StrainTensors)
        

    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
        
        I = StrainTensors['I']
        detF = StrainTensors['J'][gcounter]

        mu2 = self.mu/detF- self.lamb*(detF-1.0)
        lamb2 = self.lamb*(2*detF-1.0) 

        H_Voigt = lamb2*self.vIijIkl+mu2*self.vIikIjl

        # MaterialArgs.H_VoigtSize = H_Voigt.shape[0]
        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        mu = self.mu
        lamb = self.lamb
            
        return 1.0*mu/J*b + (lamb*(J-1.0)-mu/J)*I


