import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt, UnVoigt

__all__ = ["IsotropicLinearFlexoelectricModel"]

class IsotropicLinearFlexoelectricModel(Material):
    """The couple stress model based on linear strain and curvature tensors

        W_cs(e,x) = W_lin(e) - E.P:e - 1/2. eps e.e + eta*x:x

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicLinearFlexoelectricModel, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.energy_type = "enthalpy"
        self.nature = "linear"
        self.fields = "flexoelectric"

        # FOR STATICALLY CONDENSED FORMULATION
        if self.ndim==3:
            self.elasticity_tensor_size = 9
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.elasticity_tensor_size = 5
            self.H_VoigtSize = 5
        self.gradient_elasticity_tensor_size = ndim

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        D = self.eps*ElectricFieldx.reshape(ElectricFieldx.shape[0],ElectricFieldx.shape[1],1) # CHECK
        sigma = np.zeros((ElectricFieldx.shape[0],self.ndim,self.ndim))
        return (D, sigma, self.H_Voigt, self.elasticity_tensors, self.gradient_elasticity_tensors,
            self.piezoelectric_tensors, self.flexoelectric_tensors, self.dielectric_tensors)


    def Hessian(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        eps = self.eps
        I = StrainTensors['I']

        self.elasticity_tensor = Voigt(lamb*einsum('ij,kl',I,I)+mu*(einsum('ik,jl',I,I)+einsum('il,jk',I,I)),1)
        self.gradient_elasticity_tensor = 2.*eta*I
        self.coupling_tensor0 = np.zeros((self.elasticity_tensor.shape[0],self.gradient_elasticity_tensor.shape[0]))

        # Piezoelectric tensor must be 6x3 [3D] or 3x2 [2D]
        self.piezoelectric_tensor = self.P
        # Dielectric tensor
        self.dielectric_tensor = -self.eps*I

        # # BUILD HESSIAN
        factor = -1.
        H1 = np.concatenate((self.elasticity_tensor,factor*self.piezoelectric_tensor),axis=1)
        H2 = np.concatenate((factor*self.piezoelectric_tensor.T,self.dielectric_tensor),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)
        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        piezoelectric_tensor = self.P
        flexoelectric_tensor = self.f
        I = StrainTensors['I']
        e = StrainTensors['strain'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        tre = trace(e)
        if self.ndim==2:
            tre +=1.


        # REQUIRES CHECKS
        if self.ndim == 3:
            sigma_f = np.dot(flexoelectric_tensor,E)
            sigma_f = np.array([
                [0.,-sigma_f[2],sigma_f[1]],
                [sigma_f[2],0.,-sigma_f[0]],
                [-sigma_f[1],sigma_f[0],0.],
                ])
        else:
            # THIRD COMPONENT WHICH DOES NOT EXITS?
            sigma_f = np.zeros((self.ndim,self.ndim))

        sigma = lamb*tre*I + 2.0*mu*e - UnVoigt(np.dot(piezoelectric_tensor,E)) + sigma_f
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


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,EulerW=0,elem=0,gcounter=0):
        # CHECK
        piezoelectric_tensor = self.P
        flexoelectric_tensor = self.f
        return self.eps*ElectricFieldx.reshape(self.ndim,1) +  np.dot(flexoelectric_tensor.T,EulerW)


    def InternalEnergy(self,StrainTensors,ElectricFieldx,EulerW=0,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eta = self.eta
        eps = self.eps

        e = StrainTensors['strain'][gcounter]
        E = ElectricFieldx.reshape(self.ndim)

        tre = trace(e)
        if self.ndim==2:
            tre +=1.

        self.strain_energy = mu*einsum("ij,ij",e,e) + lamb*tre**2
        self.electrical_energy = eps*mu*einsum("i,i",E,E)
        return self.strain_energy, self.electrical_energy
