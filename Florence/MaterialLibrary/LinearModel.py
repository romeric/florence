import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace


class LinearModel(Material):
    """Classical linear elastic material model

        W = lamb/2.*trace(e)**2 + mu*e*e

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(LinearModel, self).__init__(mtype, ndim, **kwargs)


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
        return self.H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        strain = StrainTensors['strain'][gcounter]
        I = StrainTensors['I']

        mu = self.mu
        lamb = self.lamb

        if self.ndim == 3:
            tre = trace(strain)
        elif self.ndim == 2:
            tre = trace(strain) + 1

        return 2*mu*strain + lamb*tre*I


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        return np.zeros((self.ndim,1))


    def InternalEnergy(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        strain = StrainTensors['strain'][gcounter]

        if self.ndim == 3:
            tre = trace(strain)
        elif self.ndim == 2:
            tre = trace(strain) + 1

        energy = lamb/2.*tre**2 + mu*np.einsum("ij,ij",strain,strain)
        return energy
