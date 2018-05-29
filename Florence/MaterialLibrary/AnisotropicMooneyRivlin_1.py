from __future__ import division
import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt, makezero


class AnisotropicMooneyRivlin_1(Material):
    """A compressible transervely isotropic Mooney-Rivlin model with the energy given by:

            W(C) =  u1*(C:I) + u2*(G:I) - 2*(u1+2*u2+u3)*lnJ + lamb/2*(J-1)**2 + u3 FN.FN + u3 HN.HN

        where G=(H^T*H) and H=cofactor(F). The stress is zero at the origin

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(AnisotropicMooneyRivlin_1, self).__init__(mtype, ndim, **kwargs)
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
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._AnisotropicMooneyRivlin_1_ import KineticMeasures
        return KineticMeasures(self, F, self.anisotropic_orientations[elem][:,None])



    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        H_Voigt = 2.*mu2/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
                2.*(mu1+2.*mu2+mu3)/J * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
                lamb*(2.*J-1.)*einsum('ij,kl',I,I) - lamb*(J-1.) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
                4.*mu3/J * ( einsum('ij,kl',I,outerHN) + einsum('ij,kl',outerHN,I) ) + \
                2.*mu3/J*innerHN*(2.0*einsum('ij,kl',I,I) - einsum('ik,jl',I,I) - einsum('il,jk',I,I) ) + \
                2.*mu3/J * ( einsum('ik,jl',I,outerHN) + einsum('il,jk',I,outerHN) + \
                einsum('ik,jl',outerHN,I) + einsum('il,jk',outerHN,I) )  
                # 2.*mu3/J * ( einsum('il,j,k',I,HN,HN) + einsum('jl,i,k',I,HN,HN) + \
                # einsum('ik,j,l',I,HN,HN) + einsum('jk,i,l',I,HN,HN) ) 

        H_Voigt = Voigt(H_Voigt ,1)              
        
        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)[:,0]
        HN = np.dot(H,N)[:,0]
        outerFN = einsum('i,j',FN,FN)
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        mu1 = self.mu1
        mu2 = self.mu2
        mu3 = self.mu3
        lamb = self.lamb

        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1

        stress = 2.*mu1/J*b + \
            2.*mu2/J*(trb*b - np.dot(b,b)) - \
            2.*(mu1+2.*mu2+mu3)/J*I + \
            lamb*(J-1)*I +\
            2.*mu3/J*outerFN +\
            2.*mu3/J*innerHN*I - 2.*mu3/J*outerHN

        return stress