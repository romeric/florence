import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform

class IsotropicElectroMechanics_201(Material):
    """Electromechanical model in terms of internal energy 
            W(C,D) = W_mn(C) + 1/2/eps_1 (FD0*FD0)
            W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2)*lnJ + lamb/2*(J-1)**2
    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_201, self).__init__(mtype, ndim, **kwargs)
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

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]

        self.elasticity_tensor = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
            2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
            lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

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

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        D = ElectricDisplacementx.reshape(self.ndim,1)

        if self.ndim==2:
            trb = trace(b) + 1
        else:
            trb = trace(b)

        simga_mech = 2.0*mu1/J*b + \
            2.0*mu2/J*(trb*b - np.dot(b,b)) -\
            2.0*(mu1+2*mu2)/J*I +\
            lamb*(J-1)*I
        sigma_electric = J/eps_1*np.dot(D,D.T)
        sigma = simga_mech + sigma_electric

        return sigma

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)


        # SANITY CHECK FOR IMPLICIT COMPUTATUTAION OF D
        # I = StrainTensors['I']
        # J = StrainTensors['J'][gcounter]
        # E = ElectricFieldx.reshape(self.ndim,1)
        # eps_1 = self.eps_1
        # D_exact = eps_1/J*E
        # # print np.linalg.norm(D - D_exact)
        # return D_exact

        return D
