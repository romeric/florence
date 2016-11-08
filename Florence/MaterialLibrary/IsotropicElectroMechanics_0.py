import numpy as np
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
#####################################################################################################
                                # Simplest Electromechanical Helmoltz energy
#####################################################################################################


class IsotropicElectroMechanics_0(Material):
    """docstring for Steinmann"""
    
    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(IsotropicElectroMechanics_0, self).__init__(mtype, ndim, **kwargs)
        # REQUIRES SEPARATELY
        self.nvar = self.ndim+1

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

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # Update Lame constants
        mu2 = mu - lamb*(J-1.0)
        lamb2 = lamb*(2.0*J-1.0) - mu

        E  = ElectricFieldx.reshape(self.ndim,1)
        Ex = E.reshape(E.shape[0])
        EE = np.dot(E,E.T)
        be = np.dot(b,ElectricFieldx).reshape(self.ndim)

        C_Voigt = Voigt(            
            lamb2*d('ij,kl',I,I)+mu2*(d('ik,jl',I,I)+d('il,jk',I,I)) +\
            varepsilon_1*(d('ij,kl',I,EE) + d('ij,kl',EE,I) - d('ik,jl',EE,I)-d('il,jk',EE,I)-d('il,jk',I,EE)-d('ik,jl',I,EE) ) +\
            varepsilon_1*(np.dot(E.T,E)[0,0])*(0.5*(d('ik,jl',I,I) + d('il,jk',I,I))-0.5*d('ij,kl',I,I))
            ,1
            )

        e_voigt = Voigt(varepsilon_1*(d('ij,k',I,Ex)+d('ik,j',I,Ex)-d('i,jk',Ex,I)) ,1)
        # print(e_voigt)
            
        # Dielectric Tensor (Permittivity - 2nd order)
        Permittivity = -varepsilon_1*I 

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
        varepsilon_1 = self.eps_1      
        return varepsilon_1*ElectricFieldx