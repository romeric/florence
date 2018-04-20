import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace

#####################################################################################################
                                # Isotropic Linear Model
#####################################################################################################


class LinearModel(Material):
    """docstring for LinearModel"""

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

        # CHECK IF THIS IS NECESSARY
        if self.ndim == 3:
            tre = trace(strain)
        elif self.ndim == 2:
            tre = trace(strain) + 1

        # USE FASTER TRACE FUNCTION
        return 2*mu*strain + lamb*tre*I


    def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
        return np.zeros((self.ndim,1))
