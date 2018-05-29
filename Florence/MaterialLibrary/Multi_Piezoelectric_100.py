import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
from math import sqrt

class Multi_Piezoelectric_100(Material):
    """ 
                Piezoelectric model in terms of internal energy 
                W(C,D) = W_mn(C) + 1/2/eps_1 (D0*D0) + 1/2/eps_2/J (FD0*FD0) 
                    + u3*(FD0/sqrt(u3*eps_3)+FN)*(FD0/sqrt(u3*eps_3)+FN) + u3*HN*HN - 2*sqrt(u3/eps_3)*D0*N
                W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2+u3)*lnJ + lamb/2*(J-1)**2 

    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(Multi_Piezoelectric_100, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()
        self.nature = "nonlinear"
        self.fields = "electro_mechanics"
        
        self.is_transversely_isotropic = True
        if self.ndim==3:
            self.H_VoigtSize = 9
        else:
            self.H_VoigtSize = 5

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        self.mu1 = self.mu1s[elem]
        self.mu2 = self.mu2s[elem]
        self.mu3 = self.mu3s[elem]
        self.lamb = self.lambs[elem]
        self.eps_1 = self.eps_1s[elem]
        self.eps_2 = self.eps_2s[elem]
        self.eps_3 = self.eps_3s[elem]

        from Florence.MaterialLibrary.LLDispatch._Piezoelectric_100_ import KineticMeasures
        return KineticMeasures(self,np.ascontiguousarray(F), ElectricFieldx, self.anisotropic_orientations[elem][:,None])

    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1s[elem]
        mu2 = self.mu2s[elem]
        mu3 = self.mu3s[elem]
        lamb = self.lambs[elem]
        eps_1 = self.eps_1s[elem]
        eps_2 = self.eps_2s[elem]
        eps_3 = self.eps_3s[elem]

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        D  = ElectricDisplacementx.reshape(self.ndim,1)

        FN = np.dot(F,N)[:,0]
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]

        # Iso + Aniso
        C_mech = 2.*mu2/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
                2.*(mu1+2*mu2+mu3)/J * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
                lamb*(2.*J-1.)*einsum('ij,kl',I,I) - lamb*(J-1.) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
                4.*mu3/J * ( einsum('ij,kl',I,outerHN) + einsum('ij,kl',outerHN,I) ) + \
                2.*mu3/J*innerHN*(2.0*einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) + \
                2.*mu3/J * ( einsum('il,j,k',I,HN,HN) + einsum('jl,i,k',I,HN,HN) + \
                einsum('ik,j,l',I,HN,HN) + einsum('jk,i,l',I,HN,HN) )

        C_elect = 1./eps_2*(0.5*DD*(einsum('ik,jl',I,I) + einsum('il,jk',I,I) + einsum('ij,kl',I,I) ) - \
                einsum('ij,k,l',I,Dx,Dx) - einsum('i,j,kl',Dx,Dx,I)) 

        self.elasticity_tensor = C_mech + C_elect

        
        self.coupling_tensor = 1./eps_2*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I) - einsum('ij,k',I,Dx)) + \
                2.*J*sqrt(mu3/eps_3)*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I)) + \
                2.*sqrt(mu3/eps_3)*(einsum('ik,j',I,FN) + einsum('i,jk',FN,I))


        self.dielectric_tensor = J/eps_1*np.linalg.inv(b)  + 1./eps_2*I + 2.*J*sqrt(mu3/eps_3)*I

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

        mu1 = self.mu1s[elem]
        mu2 = self.mu2s[elem]
        mu3 = self.mu3s[elem]
        lamb = self.lambs[elem]
        eps_1 = self.eps_1s[elem]
        eps_2 = self.eps_2s[elem]
        eps_3 = self.eps_3s[elem]

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)[:,0]
        HN = np.dot(H,N)[:,0]
        outerFN = einsum('i,j',FN,FN)
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]
        D0D = np.dot(D,D.T)

        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1.

        sigma_mech = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - np.dot(b,b)) - \
            2.*(mu1+2*mu2+mu3)/J*I + \
            lamb*(J-1)*I +\
            2*mu3/J*outerFN +\
            2*mu3/J*innerHN*I - 2*mu3/J*outerHN

        sigma_electric = 1./eps_2*(D0D - 0.5*DD*I) +\
            2.*J*sqrt(mu3/eps_3)*D0D + 2*sqrt(mu3/eps_3)*(einsum('i,j',Dx,FN) + einsum('i,j',FN,Dx))

        sigma = sigma_mech + sigma_electric

        return sigma


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        # THE ELECTRIC FIELD NEEDS TO BE MODFIED TO TAKE CARE OF CONSTANT TERMS
        mu1 = self.mu1s[elem]
        mu2 = self.mu2s[elem]
        mu3 = self.mu3s[elem]
        lamb = self.lambs[elem]
        eps_1 = self.eps_1s[elem]
        eps_2 = self.eps_2s[elem]
        eps_3 = self.eps_3s[elem]

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)
        HN = np.dot(H,N)
        E = ElectricFieldx.reshape(self.ndim,1)
        modElectricFieldx = (E - 2.*sqrt(mu3/eps_3)*FN + 2./J*sqrt(mu3/eps_3)*HN) 

        # D = self.legendre_transform.GetElectricDisplacement(self, StrainTensors, modElectricFieldx, elem, gcounter)
        
        # SANITY CHECK FOR IMPLICIT COMPUTATUTAION OF D
        inverse = np.linalg.inv(J/eps_1*np.linalg.inv(b) + 1./eps_2*I + 2.*J*sqrt(mu3/eps_3)*I)
        D_exact = np.dot(inverse, (E - 2.*sqrt(mu3/eps_3)*FN + 2./J*sqrt(mu3/eps_3)*HN) ) 
        # print np.linalg.norm(D - D_exact)
        return D_exact

        return D
