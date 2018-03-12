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

        #------------------------------------------------------------------------------------#
        # GET MATERIAL CONSTANTS
        # mu = MaterialArgs.mu
        # lamb = MaterialArgs.lamb

        # FOURTH ORDER ELASTICITY TENSOR
        # USING EINSUM
        # d = np.einsum
        # I = StrainTensors['I']
        # H_Voigt = Voigt(lamb*d('ij,kl',I,I)+mu*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
        # MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

        # return H_Voigt
        #------------------------------------------------------------------------------------#

        # MaterialArgs.H_VoigtSize = MaterialArgs.H_Voigt.shape[0]
        # return MaterialArgs.H_Voigt

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
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))
