import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace


class IncrementalLinearElastic(Material):
    """This is the linear elastic model with zero stresses and constant Hessian
        but the geometry for this model requires incremental updates at every x=x_k
        """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IncrementalLinearElastic, self).__init__(mtype,ndim,**kwargs)
        self.energy_type = "internal_energy"
        self.nature = "linear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = False
        self.has_low_level_dispatcher = True

    def KineticMeasures(self, F, ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._LinearElastic_ import KineticMeasures
        # STRAIN TENSOR IS COMPUTED WITHIN THE LOW LEVEL VERSION OF THE MATERIAL
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):
        # RETURN THE 4TH ORDER ELASTICITY TENSOR (VOIGT FORM)
        return self.H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # RETURN STRESSES
        return np.zeros((self.ndim,self.ndim))


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))


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