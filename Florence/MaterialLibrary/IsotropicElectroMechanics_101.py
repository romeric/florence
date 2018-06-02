import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class IsotropicElectroMechanics_101(Material):
    """
                Simplest electromechanical model in terms of internal energy 
                        W(C,D0) = W_neo(C) + 1/2/eps_1 (FD0*FD0)
    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_101, self).__init__(mtype, ndim, **kwargs)
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
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._IsotropicElectroMechanics_101_ import KineticMeasures
        return KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)


    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = ElectricDisplacementx.reshape(self.ndim)

        self.elasticity_tensor = lamb*(2.*J-1.)*einsum("ij,kl",I,I) + \
        (mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        self.coupling_tensor = J/eps_1*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I))

        self.dielectric_tensor = J/eps_1*I 

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
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        D = ElectricDisplacementx.reshape(self.ndim,1)

        return 1.0*mu/J*(b - I) + lamb*(J-1)*I + J/eps_1*np.dot(D,D.T)
        # return 1.0*mu*(b - I)  + 1./eps_1*np.dot(D,D.T)


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        # D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)

        eps_1 = self.eps_1
        J = StrainTensors['J'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        D_exact = eps_1/J*E
        # print np.linalg.norm(D - D_exact)
        return D_exact

        return D


    def Permittivity(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):
        eps_1 = self.eps_1
        I = StrainTensors['I']
        J = StrainTensors['J']
        self.dielectric_tensor = J/eps_1*I
        # Negative definite inverse is needed due to Legendre transform
        return -eps_1/J*I

    def ElectrostaticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._IsotropicElectroMechanics_101_ import KineticMeasures
        D, _, H_Voigt = KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)
        H_Voigt = np.ascontiguousarray(H_Voigt[:,-self.ndim:,-self.ndim:])
        return D, None, H_Voigt