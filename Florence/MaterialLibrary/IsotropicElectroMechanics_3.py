import numpy as np
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
#####################################################################################################
                                # Simplest Electromechanical Helmoltz energy
#####################################################################################################


class IsotropicElectroMechanics_3(Material):
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_3, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "enthalpy"

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
        self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        # Using Einstein summation (using numpy einsum call)
        d = np.einsum

        mu = self.mu
        lamb = self.lamb
        varepsilon_1 = self.eps_1
        eps_2 = self.eps_2

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # Update Lame constants
        mu2 = mu - lamb*(J-1.0)
        lamb2 = lamb*(2.0*J-1.0) - mu

        E  = ElectricFieldx.reshape(self.ndim,1)
        Ex = E.reshape(E.shape[0])
        EE = np.dot(E,E.T)
        be = np.dot(b,ElectricFieldx).reshape(self.ndim,1)

        C_Voigt = Voigt(            
            lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) +\
            varepsilon_1*(d('ij,kl',I,EE) + d('ij,kl',EE,I) - d('ik,jl',EE,I)-d('il,jk',EE,I)-d('il,jk',I,EE)-d('ik,jl',I,EE) ) +\
            varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(d('ik,jl',I,I) + d('il,jk',I,I))-0.5*d('ij,kl',I,I))
            ,1
            )

        e_voigt = Voigt(varepsilon_1*(d('ij,k',I,Ex)+d('ik,j',I,Ex)-d('i,jk',Ex,I)) ,1)
        # print(e_voigt)
            
        # Dielectric Tensor (Permittivity - 2nd order)
        Permittivity = -varepsilon_1*I - 2.*eps_2/J*(2*np.dot(be,be.T)+np.dot(be.T,be)*I)

        # Build the Hessian
        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
        H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        varepsilon_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)
        Ex = E.reshape(self.ndim)

        be = np.dot(b,ElectricFieldx)

        return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I + varepsilon_1*(np.dot(E,E.T)-0.5*np.dot(Ex.T,Ex)*I) 


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        varepsilon_1 = self.eps_1
        eps_2 = self.eps_2      
        
        be = np.dot(b,E)
        ebe = np.dot(E.T,be)[0,0]
        # print(be.shape,ebe.shape)
        # print((2.*varepsilon_1/J*(ebe)*be).shape, (varepsilon_1*ElectricFieldx).shape)
        # exit()

        D = varepsilon_1*E + 2.*eps_2/J*(ebe)*be
        return D.flatten()



# import numpy as np
# from numpy import einsum
# from .MaterialBase import Material
# from Florence.Tensor import trace, Voigt
# #####################################################################################################
#                                 # Isotropic Electromechanical Model 2
#                                 # W(C,E) = W_n(C) + eps_1/2*(E*E)**2 
#                                 # W_n(C) = mu/2*C:I - mu*lnJ + lamb/2*(J-1)**2
#                                 # 0 - stands for outer product
# #####################################################################################################


# class IsotropicElectroMechanics_3(Material):
#     """docstring for IsotropicElectroMechanics"""

#     def __init__(self, ndim, **kwargs):
#         mtype = type(self).__name__
#         super(IsotropicElectroMechanics_3, self).__init__(mtype, ndim, **kwargs)

#         # INITIALISE STRAIN TENSORS
#         from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
#         StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"nonlinear")
#         self.Hessian(StrainTensors,np.zeros((self.ndim,1)))

#     def Hessian(self,StrainTensors, ElectricFieldx=0, elem=0, gcounter=0):

#         mu = self.mu
#         lamb = self.lamb
#         eps_1 = self.eps_1

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
#         E = ElectricFieldx.reshape(self.ndim,1)


#         # Elasticity
#         C = lamb*(2.*J-1.)*einsum("ij,kl",I,I) +(mu/J - lamb*(J-1))*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )
#         C_Voigt = Voigt(C,1)

#         # Coupled Tensor (e - 3rd order)
#         e_voigt = Voigt(np.zeros((self.ndim,self.ndim,self.ndim)),1)

#         # Dielectric Tensor (Permittivity - 2nd order)
#         be = np.dot(b,ElectricFieldx)
#         # be0be = np.dot(be,be.T)
#         # beibe = np.dot(be.T,be)
#         # print(np.dot(be.T,be))
#         # Permittivity = - 2.*eps_1/J*(2*np.dot(be,be.T)+np.dot(be.T,be)*I)
#         Permittivity = - 2.*eps_1/J*(2*np.dot(be,be.T)+np.dot(be.T,be)*I)

#         factor = -1.
#         H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
#         H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
#         H_Voigt = np.concatenate((H1,H2),axis=0)

#         self.H_VoigtSize = H_Voigt.shape[0]

#         # print(Permittivity)
#         # print(ElectricFieldx)

#         return H_Voigt


#     def CauchyStress(self, StrainTensors, ElectricFieldx, elem=0,gcounter=0):

#         mu = self.mu
#         lamb = self.lamb

#         I = StrainTensors['I']
#         F = StrainTensors['F'][gcounter]
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]
        
#         return 1.0*mu/J*(b - I) + lamb*(J-1)*I 


#     def ElectricDisplacementx(self, StrainTensors, ElectricFieldx, elem=0, gcounter=0):
        
#         I = StrainTensors['I']
        # J = StrainTensors['J'][gcounter]
        # b = StrainTensors['b'][gcounter]
        # E = ElectricFieldx.reshape(self.ndim,1)
        
#         eps_1 = self.eps_1

        # be = np.dot(b,E)
        # ebe = np.dot(E.T,be)[0,0]

#         return -2.*eps_1/J*(ebe)*be