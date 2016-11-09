import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
#####################################################################################################
                # Simplest electromechanical model in terms of internal energy 
                        # W(C,D0) = W_neo(C) + 1/2/eps_1 (FD0*FD0)
#####################################################################################################


class IsotropicElectroMechanics_101(Material):
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_101, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
        self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

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
        bb = np.dot(b,b)

        self.elasticity_tensor = lamb*(2.*J-1.)*einsum("ij,kl",I,I) + \
        (mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        self.coupling_tensor = J/eps_1*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I))

        self.dielectric_tensor = J/eps_1*I 

        if self.ndim == 2:
            self.H_VoigtSize = 5
        elif self.ndim == 3:
            self.H_VoigtSize = 9

        # TRANSFORM TENSORS TO THEIR ENTHALPY COUNTERPART
        E_Voigt, P_Voigt, C_Voigt = self.legendre_transform.InternalEnergyToEnthalpy(self.dielectric_tensor, 
            self.coupling_tensor, self.elasticity_tensor, in_voigt=False)
        # E_Voigt, P_Voigt, C_Voigt = self.legendre_transform.InternalEnergyToEnthalpy(self.dielectric_tensor, 
            # Voigt(self.coupling_tensor,1), Voigt(self.elasticity_tensor,1), in_voigt=True)

        # BUILD HESSIAN
        factor = 1.
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

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)

        # eps_1 = self.eps_1
        # J = StrainTensors['J'][gcounter]
        # E = ElectricFieldx.reshape(self.ndim,1)
        # D_exact = eps_1/J*E
        # print np.linalg.norm(D - D_exact)
        # return D_exact

        return D