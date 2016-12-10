import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material

class IsotropicElectroMechanics_3(Material):
    """Isotropic electromechanical model in terms of Helmoltz energy
        with one nonlinear electrostatic invariant

            W(C,E) = W_n(C) - eps_1/2*J*C**(-1):(E 0 E) + eps_2/2*(E*E)**2 
            W_n(C) = mu/2*(C:I-3) - mu*lnJ + lamb/2*(lnJ)**2
    
        where 0 stands for dyadic/outer product

    """
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_3, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1
        self.energy_type = "enthalpy"

        if self.ndim == 2:
            self.H_VoigtSize = 5
        elif self.ndim == 3:
            self.H_VoigtSize = 9

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx, elem=0):
        from Florence.MaterialLibrary.LLDispatch._IsotropicElectroMechanics_3_ import KineticMeasures
        return KineticMeasures(self, np.ascontiguousarray(F), ElectricFieldx)

    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eps_1 = self.eps_1
        eps_2 = self.eps_2

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

        P_Voigt = eps_1*( einsum('ik,j',I,Ex) + einsum('jk,i',I,Ex) - einsum('ij,k',I,Ex)) 
        
        P_Voigt = Voigt(P_Voigt,1)
            
        E_Voigt = -eps_1*I + 2.*eps_2/J*(2*np.dot(be,be.T)+np.dot(be.T,be)*I)

        # Build the Hessian
        factor = -1.
        H1 = np.concatenate((C_Voigt,factor*P_Voigt),axis=1)
        H2 = np.concatenate((factor*P_Voigt.T,E_Voigt),axis=1)
        H_Voigt = np.concatenate((H1,H2),axis=0)

        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        eps_1 = self.eps_1

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        stress = 1.0*mu/J*(b-I) + lamb/J*np.log(J)*I + \
            eps_1*(np.dot(E,E.T) - 0.5*np.dot(E.T,E)[0,0]*I) 

        return stress


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        E = ElectricFieldx.reshape(self.ndim,1)

        varepsilon_1 = self.eps_1
        eps_2 = self.eps_2      
        
        be = np.dot(b,E)
        ebe = np.dot(E.T,be)[0,0]

        D = varepsilon_1*E - 2.*eps_2/J*(ebe)*be
        return D