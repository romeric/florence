import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt


class IsotropicElectroMechanics_1(Material):
    """docstring for IsotropicElectroMechanics"""

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_1, self).__init__(mtype, ndim, **kwargs)

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
        varepsilon_1 = self.eps_1

        detF = StrainTensors['J'][gcounter]


        mu2 = mu - lamb*(detF-1.0)
        lamb2 = lamb*(2.0*detF-1.0) - mu

        delta = StrainTensors['I']
        E = 1.0*ElectricFieldx

        Ex = E.reshape(E.shape[0])
        EE = np.outer(E,E)
        innerEE = np.dot(E,E.T)

        I = delta
        # C = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) + varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) - \
        # 2.*AikBjl(EE,I)-2.0*AilBjk(I,EE) ) + varepsilon_1*(np.dot(E.T,E)[0,0])*(AikBjl(I,I)-0.5*AijBkl(I,I))

        # ORIGINAL
        # C = lamb2*AijBkl(I,I) +mu2*(AikBjl(I,I)+AilBjk(I,I)) +\
        #     varepsilon_1*(AijBkl(I,EE) + AijBkl(EE,I) -AikBjl(EE,I)-AilBjk(EE,I)-AilBjk(I,EE)-AikBjl(I,EE) ) +\
        #     varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(AikBjl(I,I) + AilBjk(I,I))-0.5*AijBkl(I,I))
        # C=0.5*(C+C.T)
        # C_Voigt = C

        C = lamb2*einsum("ij,kl",I,I) +mu2*(einsum("ik,jl",I,I)+einsum("il,jk",I,I)) +\
            varepsilon_1*(einsum("ij,kl",I,EE) + einsum("ij,kl",EE,I) - einsum("ik,jl",EE,I)- einsum("il,jk",I,EE) -\
            einsum("il,jl",I,EE)- einsum("ik,jl",I,EE) ) +\
            varepsilon_1*(innerEE)*(0.5*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )-0.5* einsum("ij,kl",I,I) )
        C_Voigt = Voigt(C,1)


        # Computing the hessian
        # Elasticity tensor (C - 4th order tensor)
        # C[i,j,k,l] += lamb2*delta[i,j]*delta[k,l]+2.0*mu2*(delta[i,k]*delta[j,l]) #

        b = StrainTensors['b'][gcounter]
        be = np.dot(b,ElectricFieldx).reshape(self.ndim,1)
        # Coupled Tensor (e - 3rd order)

        # e[k,i,j] += (-2.0*varepsilon_1/detF)*(be[j]*b[i,k] + be[i]*b[j,k]) #
        # e[i,j,k] += 1.0*varepsilon_1*( E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) ##
        # e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k]) ##

        # Note that the actual piezoelectric tensor is symmetric wrt to the last two indices
        # Actual tensor is: e[k,i,j] += 1.0*varepsilon_1*(E[i]*delta[j,k] + E[j]*delta[i,k] - delta[i,j]*E[k])
        # We need to make its Voigt_form symmetric with respect to (j,k) instead of (i,j)

        # ORIGINAL
        # e_voigt = 1.0*varepsilon_1*(AijUk(I,Ex)+AikUj(I,Ex)-UiAjk(Ex,I)).T

        e_voigt = 1.0*varepsilon_1*( einsum('ij,k',I,Ex) + einsum('ik,j',I,Ex) - einsum('i,jk',Ex,I) ).T
        e_voigt = Voigt(np.ascontiguousarray(e_voigt),1)

        # Dielectric Tensor (Permittivity - 2nd order)
        Permittivity = -varepsilon_1*delta ##

        # bb =  np.dot(StrainTensors.b,StrainTensors.b) #
        # Permittivity = -(2.0*varepsilon_1/detF)*bb #


        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*e_voigt),axis=1)
        H2 = np.concatenate((factor*e_voigt.T,Permittivity),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        self.H_VoigtSize = H_Voigt.shape[0]


        # return H_Voigt, C, e, Permittivity
        return H_Voigt


    def CauchyStress(self, StrainTensors, ElectricFieldx, elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx

        mu = self.mu
        lamb = self.lamb
        varepsilon_1 = self.eps_1

        be = np.dot(b,ElectricFieldx)

        return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I + varepsilon_1*(np.dot(E,E.T)-0.5*np.dot(E.T,E)*I) ##
        # return 1.0*mu/J*b+(lamb*(J-1.0)-mu)*I - (2.0*varepsilon_1/J)*np.dot(be,be.T)


    def ElectricDisplacementx(self, StrainTensors, ElectricFieldx, elem=0, gcounter=0):

        varepsilon_1 = self.eps_1
        return varepsilon_1*ElectricFieldx[:,None] ##

        # J = StrainTensors['J'][gcounter]
        # b = StrainTensors['b'][gcounter]
        # bb =  np.dot(b,b)
        # return (2.0*varepsilon_1/StrainTensors.J)*np.dot(bb,ElectricFieldx).reshape(StrainTensors.b.shape[0],1)

