import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt
#####################################################################################################
                                # NeoHookean
                                # W_n(C) = mu/2*C:I - mu*lnJ + lamb/2*(J-1)**2
                                # 0 - stands for outer product
#####################################################################################################


class NeoHookean_3(Material):
    """docstring for IsotropicElectroMechanics"""

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NeoHookean_3, self).__init__(mtype, ndim, **kwargs)

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
        self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

    def Hessian(self,StrainTensors, ElectricFieldx=0, elem=0, gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # Elasticity
        C = lamb*(2.*J-1.)*einsum("ij,kl",I,I) +(mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )
        H_Voigt = Voigt(C,1)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self, StrainTensors, ElectricFieldx, elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        return 1.0*mu/J*(b - I) + lamb*(J-1)*I


    def ElectricDisplacementx(self, StrainTensors, ElectricFieldx, elem=0, gcounter=0):
        return
