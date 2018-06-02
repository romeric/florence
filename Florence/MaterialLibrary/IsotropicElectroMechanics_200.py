import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material

class IsotropicElectroMechanics_200(Material):
    """Electromechanical model in terms of Helmoltz energy 
            W(C,D) = W_mn(C) - eps_1/2 (E0*C**(-1)*E0)
            W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2)*lnJ + lamb/2*(J-1)**2
    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_200, self).__init__(mtype, ndim, **kwargs)
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


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        E  = ElectricFieldx.reshape(self.ndim,1)
        Ex = E.reshape(E.shape[0])
        EE = np.dot(E,E.T)
        be = np.dot(b,ElectricFieldx).reshape(self.ndim,1)

        C_mech = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
            2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
            lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        C_elect = -eps_1/J* (einsum("ik,j,l",I,Ex,Ex) + einsum("il,j,k",I,Ex,Ex) + einsum("i,k,jl",Ex,Ex,I) + einsum("i,l,jk",Ex,Ex,I))

        C_Voigt = Voigt(C_mech + C_elect, 1)

        P_Voigt = eps_1/J*(einsum('ik,j',I,Ex)+einsum('i,jk',Ex,I))
        P_Voigt = Voigt(P_Voigt,1)
            
        E_Voigt = -eps_1/J*I

        # Build the Hessian
        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
        H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        if self.ndim==2:
            trb = trace(b) + 1
        else:
            trb = trace(b)

        simga_mech = 2.0*mu1/J*b + \
            2.0*mu2/J*(trb*b - np.dot(b,b)) -\
            2.0*(mu1+2*mu2)/J*I +\
            lamb*(J-1)*I
        sigma_electric = eps_1/J*np.dot(E,E.T)
        sigma = simga_mech + sigma_electric


        return sigma


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        eps_1 = self.eps_1
        
        D = eps_1/J*E 
        return D




























# import numpy as np
# from numpy import einsum
# from Florence.Tensor import trace, Voigt
# from .MaterialBase import Material

# class IsotropicElectroMechanics_200(Material):
#     """Electromechanical model in terms of Helmoltz energy 
#             W(C,D) = W_mn(C) + eps_1/2/J (E0*C**(-1)*E0)
#             W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2)*lnJ + lamb/2*(J-1)**2
#     """
    
#     def __init__(self, ndim, **kwargs):
#         mtype = type(self).__name__
#         super(IsotropicElectroMechanics_200, self).__init__(mtype, ndim, **kwargs)
#         # REQUIRES SEPARATELY
#         self.nvar = self.ndim+1
#         self.energy_type = "enthalpy"

#         # INITIALISE STRAIN TENSORS
#         from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
#         StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
#         self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

#     def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#         mu1 = self.mu1
#         mu2 = self.mu2
#         lamb = self.lamb
#         eps_1 = self.eps_1

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]

#         E  = ElectricFieldx.reshape(self.ndim,1)
#         Ex = E.reshape(E.shape[0])
#         EE = np.dot(E,E.T)
#         be = np.dot(b,ElectricFieldx).reshape(self.ndim,1)

#         C_mech = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
#             2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
#             lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

#         C_elect = -eps_1/J* (einsum("ik,j,l",I,Ex,Ex) + einsum("il,j,k",I,Ex,Ex) + einsum("i,k,jl",Ex,Ex,I) + einsum("i,l,jk",Ex,Ex,I))

#         C_Voigt = Voigt(C_mech + C_elect, 1)

#         P_Voigt = eps_1/J*(einsum('ik,j',I,Ex)+einsum('i,jk',Ex,I))
#         P_Voigt = Voigt(P_Voigt,1)
            
#         E_Voigt = -eps_1/J*I

#         # if elem==0 and gcounter==0:
#         #     from Florence.Tensor import makezero
#         #     makezero(E_Voigt,tol=1e-8)
#         #     print E_Voigt
#         # exit()

#         # Build the Hessian
#         factor = -1.
#         H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
#         H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
#         H_Voigt = np.concatenate((H1,H2),axis=0)

#         self.H_VoigtSize = H_Voigt.shape[0]

#         return H_Voigt



#     def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#         mu1 = self.mu1
#         mu2 = self.mu2
#         lamb = self.lamb
#         eps_1 = self.eps_1

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
#         E = ElectricFieldx.reshape(self.ndim,1)

#         if self.ndim==2:
#             trb = trace(b) + 1
#         else:
#             trb = trace(b)

#         simga_mech = 2.0*mu1/J*b + \
#             2.0*mu2/J*(trb*b - np.dot(b,b)) -\
#             2.0*(mu1+2*mu2)/J*I +\
#             lamb*(J-1)*I
#         sigma_electric = eps_1/J*np.dot(E,E.T)
#         sigma = simga_mech + sigma_electric


#         return sigma


#     def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
#         E = ElectricFieldx.reshape(self.ndim,1)
#         eps_1 = self.eps_1
        
#         D = eps_1/J*E 
#         return D
