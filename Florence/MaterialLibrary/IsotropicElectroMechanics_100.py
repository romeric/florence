import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
from copy import deepcopy


class IsotropicElectroMechanics_100(Material):
    """Simplest electromechanical model with no coupling in terms of internal energy 
                W(C,D0) = W_neo(C) + 1/2/eps_1* (D0*D0)
    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_100, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()
        self.nature = "nonlinear"
        self.fields = "electro_mechanics"  

        if self.ndim == 2:
            self.H_VoigtSize = 5
        elif self.ndim == 3:
            self.H_VoigtSize = 9

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False

    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]

        self.elasticity_tensor = lamb*(2.*J-1.)*einsum("ij,kl",I,I) + \
            (mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        self.coupling_tensor = np.zeros((self.ndim,self.ndim,self.ndim))

        self.dielectric_tensor = J/eps_1*np.linalg.inv(b)


        # TRANSFORM TENSORS TO THEIR ENTHALPY COUNTERPART
        E_Voigt, P_Voigt, C_Voigt = self.legendre_transform.InternalEnergyToEnthalpy(self.dielectric_tensor, 
            self.coupling_tensor, self.elasticity_tensor)
        # BUILD HESSIAN
        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
        H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        return 1.0*mu/J*(b - I) + lamb*(J-1)*I

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)

        # SANITY CHECK FOR IMPLICIT COMPUTATUTAION OF D
        # I = StrainTensors['I']
        # J = StrainTensors['J'][gcounter]
        # b = StrainTensors['b'][gcounter]
        # E = ElectricFieldx.reshape(self.ndim,1)
        # eps_1 = self.eps_1
        # D_exact = eps_1/J*np.dot(b,E)
        # # print np.linalg.norm(D - D_exact)
        # return D_exact
        # print D

        return D