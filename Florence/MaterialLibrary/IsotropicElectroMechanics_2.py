import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt


class IsotropicElectroMechanics_2(Material):
    """Isotropic Electromechanical Model 2
                W(C,E) = W_n(C) + c1*I: (E 0 E) + c2*C: (E 0 E)  
                W_n(C) = mu/2*C:I - mu*lnJ + lamb/2*(J-1)**2
                0 - stands for outer product
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_2, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "enthalpy"
        self.nature = "nonlinear"
        self.fields = "electro_mechanics"

        if self.ndim == 2:
            self.H_VoigtSize = 5
        elif self.ndim == 3:
            self.H_VoigtSize = 9

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False
        

    def Hessian(self,StrainTensors, ElectricFieldx=0, elem=0, gcounter=0):

        mu = self.mu
        lamb = self.lamb
        c1 = self.c1
        c2 = self.c2

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx
        # print(E.shape, b.shape, np.dot(b,ElectricFieldx).shape)
        be = np.dot(b,ElectricFieldx).reshape(self.ndim)

        # Elasticity
        C = lamb*(2.*J-1.)*einsum("ij,kl",I,I) +(mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )
        C_Voigt = Voigt(C,1)

        # Coupled Tensor (e - 3rd order)
        e_voigt = 2*c2/J*( einsum('ij,k',b,be) + einsum('i,jk',be,b) )
        e_voigt = Voigt(e_voigt,1)

        # Dielectric Tensor (Permittivity - 2nd order)
        # Permittivity = -2./J*np.dot((c1*I+c2*b),b)
        Permittivity = 2./J*np.dot((c1*I+c2*b),b) 

        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
        H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self, StrainTensors, ElectricFieldx, elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        c2 = self.c2

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx

        fe = np.dot(F,ElectricFieldx)

        return 1.0*mu/J*(b - I) + lamb*(J-1)*I + 2*c2*np.dot(fe,fe.T) 


    def ElectricDisplacementx(self, StrainTensors, ElectricFieldx, elem=0, gcounter=0):
        
        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        
        c1 = self.c1
        c2 = self.c2

        return -2.0/J*np.dot(b,np.dot((c1*I+c2*b),ElectricFieldx)).reshape(self.ndim,1)
        # return -2.0*np.dot(b,np.dot((c1*I+c2*b),ElectricFieldx))
