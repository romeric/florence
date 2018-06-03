import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

#####################################################################################################
                                # Anisotropic MooneyRivlin Model
#####################################################################################################


class AnisotropicMooneyRivlin_0(Material):
    """A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
        The energy is given by:

            W(C) =  gamma * ( alpha*(C:I) + beta*(G:I) ) +
                    eta*(1-alpha)*( (N C N)**2 + N G N) - ut*J + lambda/2*(J-1)**2

            ut = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta  # for the stress to be
                zero at the origin

        the parameter "gamma" controls the amount of anisotropy and the vector N(ndim,1) is
        the direction of anisotropy

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(AnisotropicMooneyRivlin_0, self).__init__(mtype, ndim, **kwargs)
        self.nvar = self.ndim
        self.is_transversely_isotropic = True
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        else:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)[:,0]
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        # FIX GAMMA
        gamma = 0.5
        # gamma = 1.0
        alpha = mu/2./gamma
        beta  = mu/2./gamma
        eta   = mu/3.
        ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
        lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


        H_Voigt = 2.*gamma*beta/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
                4.*(1-gamma)*eta/J * einsum('i,j,k,l',FN,FN,FN,FN) + \
                4.*(1-gamma)*eta/J * ( innerHN * einsum('ij,kl',I,I) - \
                0.5*innerHN * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
                einsum('ij,k,l',I,HN,HN) - einsum('i,j,kl',HN,HN,I) ) + \
                2.*(1-gamma)*eta/J * ( einsum('il,j,k',I,HN,HN) + einsum('jl,i,k',I,HN,HN) + \
                einsum('ik,j,l',I,HN,HN) + einsum('jk,i,l',I,HN,HN) ) - \
                ut*einsum('ij,kl',I,I) + ut * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
                lamb*(2.*J-1.)*einsum('ij,kl',I,I) - lamb*(J-1.) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )

        H_Voigt = Voigt(H_Voigt ,1)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        mu = self.mu
        lamb = self.lamb

        # FIX GAMMA
        gamma = 0.5
        # gamma = 1.0
        alpha = mu/2./gamma
        beta  = mu/2./gamma
        eta   = mu/3.
        ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
        lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1


        stress = 2.*gamma*alpha/J*b + 2.*gamma*beta/J*(trb*b - np.dot(b,b)) + \
                 2.*(1.- gamma)*eta/J*np.dot(FN.T,FN)[0][0]*np.dot(FN,FN.T) + \
                 2.*(1.- gamma)*eta/J*(innerHN*I - outerHN) - \
                 ut*I + lamb*(J-1.)*I

        return stress


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))


