import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace

#####################################################################################################
                            # INCREMENTAL LINEAR ELASTIC ISOTROPIC MODEL
#####################################################################################################


class IncrementalLinearElastic(Material):
    """This is the linear elastic model with zero stresses and constant Hessian
        but the geometry for this model requires incremental updates at every x=x_k
        """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IncrementalLinearElastic, self).__init__(mtype,ndim,**kwargs)

    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
        # RETURN THE 4TH ORDER ELASTICITY TENSOR (VOIGT FORM)
        return self.H_Voigt


    def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # RETURN STRESSES
        return np.zeros((self.ndim,self.ndim)),np.zeros((self.ndim,self.ndim))


    def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))
