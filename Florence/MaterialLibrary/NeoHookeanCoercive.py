import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace

#####################################################################################################
                                        # NeoHookean Material Model 2
#####################################################################################################


class NeoHookeanCoercive(Material):
    """Material model for neo-Hookean with the following internal energy:

        W(C) = mu/2*(C:I)+mu*e**(1-J)/J+lamba/2*(J-1)**2

        """

    def __init__(self, ndim, **kwargs):
    # def __init__(self, ndim, MaterialArgs=None):
        mtype = type(self).__name__
        super(NeoHookeanCoercive, self).__init__(mtype, ndim, **kwargs)
        self.ndim = ndim
        self.nvar = self.ndim

        # self.mu = self.mu
        # self.lamb = self.lamb  - self.mu

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


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]

        H_Voigt = self.lamb*(2.*J-1)*self.vIijIkl - self.lamb*(J-1.)*self.vIikIjl + \
                1.0*self.mu*J*np.exp(1.-J)*self.vIijIkl + self.mu*(J+1.)/J*np.exp(1.-J)*self.vIikIjl

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # stress = 2.0*self.mu/J*b + self.lamb*(J-1.0)*I - 2.*J**(-2)*self.mu*np.exp(-J+1.)*I
        # stress = self.mu/J*b + self.lamb*(J-1.0)*I - J**(-2)*self.mu*np.exp(-J+1.)*I
        stress = 1.0*self.mu/J*b + self.lamb*(J-1.0)*I - self.mu*(J+1.)/J*np.exp(1.-J)*I

        return stress


