import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class Piezoelectric_100(Material):
    """ 
                Piezoelectric model in terms of internal energy 
                W(C,D) = W_mn(C) + 1/2/eps_1 (D0*D0) + 1/2/eps_2/J (FD0*FD0) 
                    + u3*(FD0/sqrt(u3*eps_3)+FN)*(FD0/sqrt(u3*eps_3)+FN) + u3*HN*HN - 2*sqrt(u3/eps_3)*D0*N
                W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2+u3)*lnJ + lamb/2*(J-1)**2 

    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(Piezoelectric_100, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "internal_energy"
        self.legendre_transform = LegendreTransform()
        
        self.is_anisotropic = True

        # self.N = np.array([[0,0,1.]])
        self.N = np.array([[0,1.]]).reshape(ndim,1)

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
        self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        lamb = self.lamb
        eps_1 = self.eps_1
        eps_2 = self.eps_2

        N = self.N

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        H = J*np.linalg.inv(F).T

        D  = ElectricDisplacementx.reshape(self.ndim,1)
        Dx = D.reshape(self.ndim)
        DD = np.dot(D.T,D)[0,0]

        HN = np.dot(H,N)
        # print N.shape, H.shape, np.dot(H,N)[:,0].shape, np.dot(HN.T,HN).shape
        # exit()
        HN = np.dot(H,N)
        HNHN = np.dot(HN.T,HN)[0,0]
        HNOHN = np.dot(HN,HN.T)
        HN = HN[:,0]

        C_mech = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
            2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
            lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) +\
            4.0*mu3*(-1./J*einsum('ij,k,l',I,HN,HN) -1./J*einsum('i, j,kl',HN,HN,I) + HNHN * (einsum('ij,kl',I,I) - \
                0.5*einsum('ik,jl',I,I) - 0.5*einsum('il,jk',I,I) ) + 0.5/J*einsum('il,jk',I,HNOHN) +\
                0.5/J*einsum('ik,jl',I,HNOHN) + 0.5/J*einsum('jl,ik',I,HNOHN) + 0.5/J*einsum('jk,il',I,HNOHN) )

        C_elect = 1./eps_2*(0.5*DD*(einsum('ik,jl',I,I) + einsum('il,jk',I,I) + einsum('ij,kl',I,I) ) - \
                einsum('ij,k,l',I,Dx,Dx) - einsum('i,j,kl',Dx,Dx,I))

        self.elasticity_tensor = C_mech + C_elect

        self.coupling_tensor = 1./eps_2*(einsum('ik,j',I,Dx) + einsum('i,jk',Dx,I) - einsum('ij,k',I,Dx))

        self.dielectric_tensor = J/eps_1*np.linalg.inv(b)  + 1./eps_2*I 

        if self.ndim == 2:
            self.H_VoigtSize = 5
        elif self.ndim == 3:
            self.H_VoigtSize = 9

        # TRANSFORM TENSORS TO THEIR ENTHALPY COUNTERPART
        # E_Voigt, P_Voigt, C_Voigt = self.legendre_transform.InternalEnergyToEnthalpy(self.dielectric_tensor, 
            # self.coupling_tensor, self.elasticity_tensor, in_voigt=False)
        E_Voigt, P_Voigt, C_Voigt = self.legendre_transform.InternalEnergyToEnthalpy(self.dielectric_tensor, 
            Voigt(self.coupling_tensor,1), Voigt(self.elasticity_tensor,1), in_voigt=True)

        # BUILD HESSIAN
        factor = 1.
        H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
        H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        lamb = self.lamb
        eps_2 = self.eps_2

        N = self.N

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        H = J*np.linalg.inv(F).T
        D = ElectricDisplacementx.reshape(self.ndim,1)
        DD = np.dot(D.T,D)[0,0]

        FN = np.dot(F,N)
        HN = np.dot(H,N)
        HNHN = np.dot(HN.T,HN)[0,0]
        HNOHN = np.dot(HN,HN.T)
        FNOFN = np.dot(FN,FN.T)
        HN = HN[:,0]

        simga_mech = 2.0*mu1/J*b + \
            2.0*mu2/J*(trace(b)*b - np.dot(b,b)) -\
            2.0*(mu1+2*mu2)/J*I +\
            lamb*(J-1)*I +\
            2.0*mu3/J*HNHN*I - 2.*mu3/J*HNOHN + \
            2.0*mu3/J*FNOFN
        sigma_electric = 1/eps_2*(np.dot(D,D.T) - 0.5*DD*I)
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





# FN*FN = NCN
# S = 2 N O N
# FD0 FD0 = D0 CD0
# S = 2 D0 O D0 -> sigma = 2J D O D