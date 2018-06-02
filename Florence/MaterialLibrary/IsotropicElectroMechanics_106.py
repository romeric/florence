from __future__ import division
import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class IsotropicElectroMechanics_106(Material):
    """ Electromechanical model in terms of internal energy
    
                        W(C,D) = W_mn(C) + 1/2/eps_1 (D0*D0) + 1/2/eps_2/J (FD0*FD0)
                        W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2)*lnJ + lamb/2*(J-1)**2

    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_106, self).__init__(mtype, ndim, **kwargs)
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
        from Florence.MaterialLibrary.LLDispatch._IsotropicElectroMechanics_106_ import KineticMeasures
        return KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)


    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1
        eps_2 = self.eps_2

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]

        C_mech = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
            2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
            lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        C_elect = 1./eps_2*(0.5*DD*(einsum('ik,jl',I,I) + einsum('il,jk',I,I) + einsum('ij,kl',I,I) ) - \
                einsum('ij,k,l',I,Dx,Dx) - einsum('i,j,kl',Dx,Dx,I))

        self.elasticity_tensor = C_mech + C_elect

        self.coupling_tensor = 1./eps_2*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I) - einsum('ij,k',I,Dx))

        self.dielectric_tensor = J/eps_1*np.linalg.inv(b)  + 1./eps_2*I 

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
        eps_2 = self.eps_2

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        D = ElectricDisplacementx.reshape(self.ndim,1)
        DD = np.dot(D.T,D)[0,0]

        trb = trace(b)
        if self.ndim==2:
            trb += 1.

        simga_mech = 2.0*mu1/J*b + \
            2.0*mu2/J*(trb*b - np.dot(b,b)) -\
            2.0*(mu1+2*mu2)/J*I +\
            lamb*(J-1.)*I
        sigma_electric = 1./eps_2*(np.dot(D,D.T) - 0.5*DD*I)
        sigma = simga_mech + sigma_electric

        return sigma

    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, ElectricFieldx, elem, gcounter)

        # SANITY CHECK FOR IMPLICIT COMPUTATUTAION OF D
        # I = StrainTensors['I']
        # J = StrainTensors['J'][gcounter]
        # b = StrainTensors['b'][gcounter]
        # E = ElectricFieldx.reshape(self.ndim,1)
        # eps_1 = self.eps_1
        # eps_2 = self.eps_2
        # inverse = np.linalg.inv(J/eps_1*np.linalg.inv(b) + 1./eps_2*I)
        # D_exact = np.dot(inverse,E)
        # print np.linalg.norm(D - D_exact)
        # return D_exact

        return D



    def Permittivity(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1
        eps_2 = self.eps_2

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        self.dielectric_tensor = J/eps_1*np.linalg.inv(b)  + 1./eps_2*I 

        # TRANSFORM TENSORS TO THEIR ENTHALPY COUNTERPART
        H_Voigt = -np.linalg.inv(self.dielectric_tensor)

        return H_Voigt


    def ElectrostaticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._IsotropicElectroMechanics_106_ import KineticMeasures
        D, _, H_Voigt = KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx)
        H_Voigt = np.ascontiguousarray(H_Voigt[:,-self.ndim:,-self.ndim:])
        return D, None, H_Voigt