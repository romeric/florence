import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt, UnVoigt


class BonetTranservselyIsotropicHyperElastic(Material):
    """A compressible transervely isotropic model based on Bonet 1998.
        Material model is not polyconvex
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(BonetTranservselyIsotropicHyperElastic, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = True
        self.is_nonisotropic = True

        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"
        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        # CHECK IF FIBRE ORIENTATION IS SET
        if self.anisotropic_orientations is None:
            raise ValueError("Fibre orientation for non-isotropic material is not available")

        # Get material constants (5 in this case)
        E = self.E
        E_A = self.E_A
        G_A = self.G_A
        v = self.nu
        mu = self.mu

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem,:,None]
        FN = np.dot(F,N)[:,0]

        alpha = E/8.0/(1.+v)
        beta = E/8.0/(1.+v)
        eta_1 = 4.*alpha - G_A
        lamb = - (3*E)/(2*(v + 1)) - (E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A))
        eta_2 = E/(4*(v + 1)) - (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) + \
                    (E*(- E*v**2 + E_A))/(4*(v + 1)*(2*E*v**2 + E_A*v - E_A))
        gamma = (E_A**2*(v - 1))/(8*(2*E*v**2 + E_A*v - E_A)) - G_A/2 + \
                    (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) - \
                    (E*(- E*v**2 + E_A))/(8*(v + 1)*(2*E*v**2 + E_A*v - E_A))

        ut = 2*alpha + 4*beta


        H_Voigt = 2.*beta/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
                lamb*(2.*J-1.) *einsum('ij,kl',I,I) + \
                (ut/J - lamb*(J-1.) ) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
                4.*eta_2/J*( einsum('ij,k,l',b,FN,FN) + einsum('i,j,kl',FN,FN,b)  ) + \
                8.*gamma/J*( einsum('i,j,k,l',FN,FN,FN,FN) ) - \
                eta_1/J*( einsum('jk,i,l',b,FN,FN) + einsum('ik,j,l',b,FN,FN)  +
                        einsum('jl,i,k',b,FN,FN) + einsum('il,j,k',b,FN,FN) )


        H_Voigt = Voigt(H_Voigt ,1)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        # N = np.array([-1.,0.]).reshape(2,1)
        N = self.anisotropic_orientations[elem,:,None]
        FN = np.dot(F,N)
        # HN = np.dot(H,N)[:,0]
        innerFN = np.dot(FN.T,FN)[0][0]
        outerFN = np.dot(FN,FN.T)
        # innerFN = einsum('i,i',FN,FN)
        # outerFN = einsum('i,j',FN,FN)
        bFN = np.dot(b,FN)

        E = self.E
        E_A = self.E_A
        G_A = self.G_A
        v = self.nu
        mu = self.mu

        alpha = E/8.0/(1.+v)
        beta = E/8.0/(1.+v)
        eta_1 = 4.*alpha - G_A
        lamb = - (3*E)/(2*(v + 1)) - (E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A))
        eta_2 = E/(4*(v + 1)) - (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) + \
                    (E*(- E*v**2 + E_A))/(4*(v + 1)*(2*E*v**2 + E_A*v - E_A))
        gamma = (E_A**2*(v - 1))/(8*(2*E*v**2 + E_A*v - E_A)) - G_A/2 + \
                    (E_A*E*v)/(4*(2*E*v**2 + E_A*v - E_A)) - \
                    (E*(- E*v**2 + E_A))/(8*(v + 1)*(2*E*v**2 + E_A*v - E_A))

        ut = 2*alpha + 4*beta



        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1


        stress = 2.*alpha/J*b + 2.*beta/J*(trb*b - np.dot(b,b)) - ut/J*I + lamb*(J-1.)*I + \
            2.*eta_1/J*outerFN + 2.*eta_2/J*(innerFN-1.)*b + 2.*eta_2/J*(trb-3.)*outerFN + \
            4.*gamma/J*(innerFN-1.)*outerFN - eta_1/J *(np.dot(bFN,FN.T)+np.dot(FN,bFN.T))



        return stress
