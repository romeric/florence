import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt


class MooneyRivlin_1(Material):
    """ Polyconvex compressible MooneyRivlin material model based on the energy:

            W = alpha*C:I+beta*G:I+lambda/2*(J-1)**2-4*beta*J-2*alpha*lnJ - (3*alpha-beta)

        where at the origin (alpha + beta) = mu/2
        """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MooneyRivlin_1, self).__init__(mtype,ndim,**kwargs)
        
        self.alpha = self.mu/4.0
        self.beta = self.mu/4.0

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        # Actual version
        # d = np.einsum
        # H_Voigt = Voigt( 4.0*beta/J*d('ij,kl',b,b) - 2.0*beta/J*( d('ik,jl',b,b) + d('il,jk',b,b) ) +\
        #   (lamb+4.0*beta+4.0*alpha/J)*d('ij,kl',I,I) + 2.0*(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*d('ij,kl',I,I) -\
        #   1.0*(lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1) 

        # # Simplified version
        # H_Voigt = 2.0*beta/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
        #   (lamb*(2.0*J-1.0) -4.0*beta)*einsum('ij,kl',I,I) - \
        #   (lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )

        # Further simplified version
        H_Voigt = 2.0*self.beta/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
            (lamb*(2.0*J-1.0) -4.0*self.beta)*self.Iijkl - \
            (lamb*(J-1.0) -4.0*self.beta -2.0*self.alpha/J)*self.Iikjl

        H_Voigt = Voigt(H_Voigt,1) 


        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]


        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1

        # stress = 2.0*alpha/J*b+2.0*beta/J*(trace(b)*b - np.dot(b,b)) + (lamb*(J-1.0)-4.0*beta-2.0*alpha/J)*I 
        stress = 2.0*self.alpha/J*b+2.0*self.beta/J*(trb*b - np.dot(b,b)) + (lamb*(J-1.0)-4.0*self.beta-2.0*self.alpha/J)*I 
        # print stress
        return stress


    def ElectricDisplacementx(self,StrainTensors,ElectricFieldx):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))
