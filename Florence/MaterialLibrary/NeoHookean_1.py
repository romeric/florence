import numpy as np
from Florence.Tensor import trace

#####################################################################################################
                                        # NeoHookean Material Model 2
#####################################################################################################


class NeoHookean_1(object):
    """NeoHookean model with the following energy

        W(C) = u/2*C:I -u*J + lambda *(J-1)**2

        """
    def __init__(self, ndim, MaterialArgs=None):
        super(NeoHookean_1, self).__init__()
        self.ndim = ndim
        self.nvar = self.ndim

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False 


    def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
        mu = MaterialArgs.mu
        lamb = MaterialArgs.lamb
        I = StrainTensors['I']
        detF = StrainTensors['J'][gcounter]

        mu2 = mu - lamb*(detF-1.0)
        lamb2 = lamb*(2*detF-1.0) - mu


        C_Voigt = lamb2*MaterialArgs.vIijIkl+mu2*MaterialArgs.vIikIjl

        MaterialArgs.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt


    def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        mu = MaterialArgs.mu
        lamb = MaterialArgs.lamb

        return (lamb*(J-1.0)-mu)*I+1.0*mu/J*b


