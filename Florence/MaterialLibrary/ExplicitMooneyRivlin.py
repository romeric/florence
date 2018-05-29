import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform


class ExplicitMooneyRivlin(Material):
    """The fundamental MooneyRivlin model from Gil and Ortigosa et. al.

        W_mn(C) = u1*C:I+u2*G:I - 2*(u1+2*u2)*lnJ + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(ExplicitMooneyRivlin, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = True
        # self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._ExplicitMooneyRivlin_ import KineticMeasures
        return KineticMeasures(self,F), None


    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        C_Voigt = 2.*mu2/J*(2*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b)) +\
            2.*(mu1+2.*mu2)/J*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) + \
            lamb*(2.*J-1.)*einsum("ij,kl",I,I) - lamb*(J-1)*( einsum("ik,jl",I,I)+einsum("il,jk",I,I) )

        C_Voigt = Voigt(C_Voigt,1)


        return C_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        trb = trace(b)
        if self.ndim==2:
            trb +=1.

        sigma = 2.0*mu1/J*b + \
            2.0*mu2/J*(trb*b - np.dot(b,b)) -\
            2.0*(mu1+2*mu2)/J*I +\
            lamb*(J-1)*I

        return sigma


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        C = np.dot(F.T,F)
        G = np.dot(H.T,H)

        energy  = mu1*(einsum('ij,ij',C,I) - 3.) + mu2*(einsum('ij,ij',G,I) - 3.) - 2.*(mu1+2.*mu2)*np.log(J) + lamb/2.*(J-1)**2

        return energy
