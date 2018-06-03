import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

class RegularisedNeoHookean(Material):
    """The fundamental Neo-Hookean internal energy, described in Bonet et. al.

        W(C) = mu/2*(C:I-3)- mu*lnJ + lamb/2*(J-1)**2 + W_elastic

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(RegularisedNeoHookean, self).__init__(mtype, ndim, **kwargs)
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

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        return


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        # Jb = 1.27168554933
        Jb = self.Jbar[elem]

        mu2 = self.mu/J- self.lamb*(J-Jb)
        lamb2 = self.lamb*(2*J-Jb)

        H_Voigt_n = lamb2*self.vIijIkl+mu2*self.vIikIjl

        # return H_Voigt

        H_Voigt_l = Voigt(self.lamb*einsum('ij,kl',I,I)+self.mu*(einsum('ik,jl',I,I)+einsum('il,jk',I,I)) ,1)

        alp = 0.1
        # H_Voigt = (1-alp)*H_Voigt_l + alp*H_Voigt_n
        H_Voigt = H_Voigt_n
        # return H_Voigt_l + H_Voigt_n
        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        strain = StrainTensors['strain'][gcounter]

        # mu = self.mu
        # lamb = self.lamb

        # return 1.0*mu/J*b + (lamb*(J-1.0)-mu/J)*I

        mu = self.mu
        lamb = self.lamb
        # Jb = 1.27168554933
        Jb = self.Jbar[elem]

        # CHECK IF THIS IS NECESSARY
        if self.ndim == 3:
            tre = trace(strain)
        elif self.ndim == 2:
            tre = trace(strain) + 1

        sigma_linear =  2.*mu*strain + lamb*tre*I
        simga_nonlinear = 1.0*mu/J*b + (lamb*(J-Jb)-mu/J)*I

        alp = 0.1
        # simga =  (1-alp)*simga_nonlinear + alp*sigma_linear
        simga = simga_nonlinear

        return simga



