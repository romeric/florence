from __future__ import division
import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material

class SteinmannModel(Material):
    """Steinmann's electromechanical model in terms of enthalpy

        W(C,E) = W_n(C) + c1*I:(E0 0 E0) + c2*C:(E0 0 E0) - eps_1/2*J*C**(-1):(E0 0 E0)
        W_n(C) = mu/2*(C:I-3) - mu*lnJ + lamb/2*(lnJ)**2

        Reference:
            D. K. Vu, P. Steinmann, and G. Possart, "Numerical modelling of non-linear electroelasticity",
            International Journal for Numerical Methods in Engineering, 70:685-704, (2007)   
    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(SteinmannModel, self).__init__(mtype, ndim, **kwargs)
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
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._SteinmannModel_ import KineticMeasures
        return KineticMeasures(self, np.ascontiguousarray(F), ElectricFieldx)

    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        c1 = self.c1
        c2 = self.c2
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]


        E = 1.0*ElectricFieldx.reshape(self.ndim,1)
        Ex = E.reshape(E.shape[0])
        EE = np.dot(E,E.T)
        be = np.dot(b,ElectricFieldx).reshape(self.ndim)

        C_Voigt = lamb/J*einsum('ij,kl',I,I) - (lamb*np.log(J) - mu)/J*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
            eps_1*( einsum('ij,kl',I,EE) + einsum('ij,kl',EE,I) - einsum('ik,jl',EE,I) - einsum('il,jk',EE,I) - \
                einsum('ik,jl',I,EE) - einsum('il,jk',I,EE) ) + \
            eps_1*(np.dot(E.T,E)[0,0])*0.5*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) - einsum('ij,kl',I,I) )

        C_Voigt = Voigt(C_Voigt,1)

        P_Voigt = eps_1*( einsum('ik,j',I,Ex) + einsum('jk,i',I,Ex) - einsum('ij,k',I,Ex)) +\
            2.0*c2/J*( einsum('ik,j',b,be) + einsum('i,jk',be,b) )
        
        P_Voigt = Voigt(P_Voigt,1)
            

        E_Voigt = -eps_1*I +  2.0*c1/J*b + 2.0*c2/J*np.dot(b,b)

        # Build the Hessian
        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
        H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        c1 = self.c1
        c2 = self.c2
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        be = np.dot(b,E)

        stress = 1.0*mu/J*(b-I) + lamb/J*np.log(J)*I + \
            eps_1*(np.dot(E,E.T) - 0.5*np.dot(E.T,E)[0,0]*I) +\
            2.0*c2/J*np.dot(be,be.T)

        return stress


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
        mu = self.mu
        lamb = self.lamb
        c1 = self.c1
        c2 = self.c2
        eps_1 = self.eps_1      

        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        bb =  np.dot(b,b)
        be = np.dot(b,E)
        
        D = eps_1*E - 2.0*c1/J*be - 2.0*c2/J*np.dot(bb,E)

        return D























# import numpy as np
# from Florence.Tensor import trace
# #####################################################################################################
#                                 # Isotropic Steinmann Model
# #####################################################################################################


# class Steinmann(object):
#     """docstring for Steinmann"""
#     def __init__(self, ndim):
#         super(Steinmann, self).__init__()
#         self.ndim = ndim
#     def Get(self):
#         self.nvar = self.ndim+1
#         self.modelname = 'Steinmann'
#         return self.nvar, self.modelname

#     def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#         # Using Einstein summation (using numpy einsum call)
#         d = np.einsum

#         # Get material constants (5 in this case)
#         mu = MaterialArgs.mu
#         lamb = MaterialArgs.lamb
#         c1 = MaterialArgs.c1
#         c2 = MaterialArgs.c2
#         varepsilon_1 = MaterialArgs.eps_1

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]

#         # Update Lame constants
#         mu2 = mu - lamb*(J-1.0)
#         lamb2 = lamb*(2.0*J-1.0) - mu

#         E = 1.0*ElectricFieldx
#         Ex = E.reshape(E.shape[0])
#         EE = np.dot(E,E.T)
#         be = np.dot(b,ElectricFieldx).reshape(ndim)

#         # Fourth order elasticity tensor
#         # C_Voigt = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) +\
#         #   varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) -AikBjl(EE,I)-AilBjk(EE,I)-AilBjk(I,EE)-AikBjl(I,EE) ) +\
#         #   varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(AikBjl(I,I) + AilBjk(I,I))-0.5*AijBkl(I,I))
#         # C_Voigt=0.5*(C_Voigt+C_Voigt.T)

#         C_Voigt = Voigt(            
#             lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) +\
#             varepsilon_1*(d('ij,kl',I,EE) + d('ij,kl',EE,I) - d('ik,jl',EE,I)-d('il,jk',EE,I)-d('il,jk',I,EE)-d('ik,jl',I,EE) ) +\
#             varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(d('ik,jl',I,I) + d('il,jk',I,I))-0.5*d('ij,kl',I,I))
#             ,1
#             )

        
#         # Coupled Tensor (e - 3rd order)
#         # Note that the actual piezoelectric tensor is symmetric wrt to the last two indices
#         # Actual tensor (varepsilon_1 bit) is: e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) 
#         # We need to make its Voigt_form symmetric with respect to (j,k) instead of (i,j) 
#         # e_voigt = 1.0*varepsilon_1*(AijUk(I,Ex)+AikUj(I,Ex)-UiAjk(Ex,I)).T +\
#         # (2.0*c2/J)*(AikUj(b,be)+AijUk(b,be)).T

#         e_voigt = Voigt(
#             1.0*varepsilon_1*(d('ij,k',I,Ex)+d('ik,j',I,Ex)-d('i,jk',Ex,I)) +\
#             (2.0*c2/J)*(d('ik,j',b,be)+d('ij,k',b,be))
#             ,1
#             )
            

#         # Dielectric Tensor (Permittivity - 2nd order)
#         Permittivity = -varepsilon_1*I +\
#         (2.0*c1/J)*b +\
#         (2.0*c2/J)*np.dot(b,b)

#         # Build the Hessian
#         factor = -1.
#         H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
#         H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
#         H_Voigt = np.concatenate((H1,H2),axis=0)

#         MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#         return H_Voigt



#     def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#         c2 = MaterialArgs.c2

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
#         E = ElectricFieldx

#         mu = MaterialArgs.mu
#         lamb = MaterialArgs.lamb
#         varepsilon_1 = MaterialArgs.eps_1

#         be = np.dot(b,ElectricFieldx)

#         return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I + varepsilon_1*(np.dot(E,E.T)-0.5*np.dot(E.T,E)[0,0]*I) +\
#         (2.0*c2/J)*np.dot(be,be.T)


#     def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
#         c1 = MaterialArgs.c1
#         c2 = MaterialArgs.c2
#         varepsilon_1 = MaterialArgs.eps_1       

#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
#         bb =  np.dot(b,b)
        
#         return varepsilon_1*ElectricFieldx -\
#         (2.0*c1/J)*np.dot(b,ElectricFieldx) -\
#         (2.0*c2/StrainTensors.J)*np.dot(bb,ElectricFieldx).reshape(StrainTensors.b.shape[0],1)
