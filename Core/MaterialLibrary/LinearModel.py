import numpy as np
# from Core.Supplementary.Tensors.Tensors import *
from Core.Supplementary.Tensors import *
# from Core.Supplementary.Tensors.Tensors_Sym import *

#####################################################################################################
                                # Isotropic Linear Model
#####################################################################################################


class LinearModel(object):
    """docstring for LinearModel"""

    def __init__(self, ndim, MaterialArgs=None):
        super(LinearModel, self).__init__()
        self.ndim = ndim
        self.nvar = self.ndim
        

    def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

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
        
        MaterialArgs.H_VoigtSize = MaterialArgs.H_Voigt.shape[0]


        return MaterialArgs.H_Voigt



    def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):


        strain = StrainTensors['strain'][gcounter]
        I = StrainTensors['I']

        mu = MaterialArgs.mu
        lamb = MaterialArgs.lamb

        # CHECK IF THIS IS NECESSARY
        if self.ndim == 3:
            tre = trace(strain)
        elif self.ndim == 2:
            tre = trace(strain) + 1

        # return 2*mu*strain + lamb*np.trace(strain)*I 
        # USE FASTER TRACE FUNCTION
        return 2*mu*strain + lamb*trace(strain)*I  
        

    def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))
