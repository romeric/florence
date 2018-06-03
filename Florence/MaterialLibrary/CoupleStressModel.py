import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

__all__ = ["CoupleStressModel"]

class CoupleStressModel(Material):
    """The couple stress model based on linear strain and curvature tensors

        W_cs(e,x) = W_lin(e) + eta*x:x

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(CoupleStressModel, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.energy_type = "internal_energy"
        self.nature = "linear"
        self.fields = "couple_stress"

        # FOR STATICALLY CONDENSED FORMULATION
        if self.ndim==3:
            self.elasticity_tensor_size = 6
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.elasticity_tensor_size = 3
            self.H_VoigtSize = 3
        self.gradient_elasticity_tensor_size = ndim

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        # raise NotImplementedError("KineticMeasures for this formulation is not implemented yet")
        return None, self.elasticity_tensors, self.gradient_elasticity_tensors


    def Hessian(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        I = StrainTensors['I']

        self.elasticity_tensor = Voigt(lamb*einsum('ij,kl',I,I)+mu*(einsum('ik,jl',I,I)+einsum('il,jk',I,I)),1)
        self.gradient_elasticity_tensor = 2.*eta*I
        self.coupling_tensor0 = np.zeros((self.elasticity_tensor.shape[0],self.gradient_elasticity_tensor.shape[0]))

        # # BUILD HESSIAN
        # factor = 1.
        # H1 = np.concatenate((self.elasticity_tensor,factor*self.coupling_tensor0),axis=1)
        # H2 = np.concatenate((factor*self.coupling_tensor0.T,self.gradient_elasticity_tensor),axis=1)
        # H_Voigt = np.concatenate((H1,H2),axis=0)
        # return H_Voigt
        return None



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        I = StrainTensors['I']
        e = StrainTensors['strain'][gcounter]

        tre = trace(e)
        if self.ndim==2:
            tre +=1.

        sigma = lamb*tre*I + 2.0*mu*e
        return sigma


    def CoupleStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        GradU = F - I
        k = GradU - GradU.T
        k = np.array([k[0,1],k[0,2],k[1,2]])

        couple_stress = 2*eta*k
        return couple_stress


    def LagrangeMultiplierStress(self,S,ElectricFieldx=None,elem=0,gcounter=0):

        lm_stress = 0.5*S
        return lm_stress


    def TotalStress(self,StrainTensors,S,ElectricFieldx=None,elem=0,gcounter=0):

        cauchy_stress = self.CauchyStress(StrainTensors,ElectricFieldx,elem=elem,gcounter=gcounter)
        lm_stress     = self.LagrangeMultiplierStress(S,ElectricFieldx,elem=elem,gcounter=gcounter)
        lm_stress     = np.array([
            [0.,lm_stress[0],lm_stress[1]],
            [-lm_stress[0],0.,lm_stress[2]],
            [-lm_stress[1],-lm_stress[2],0.]
            ])
        return cauchy_stress + lm_stress
