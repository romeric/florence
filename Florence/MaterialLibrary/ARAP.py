import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
from scipy.linalg import polar


class ARAP(Material):
    """The fundamental ARAP model

        W_arap(F) = (F - R)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(ARAP, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._MooneyRivlin_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        u, s, vh = np.linalg.svd(F, full_matrices=True)
        if self.ndim == 2:
            s1 = s[0]
            s2 = s[1]
            T = np.array([[0.,-1],[1,0.]])
            s1s2 = s1 + s2
            if (s1s2 < 2.0):
                s1s2 = 2.0
            lamb = 2. / (s1s2)
            T = 1./np.sqrt(2) * np.dot(u, np.dot(T, vh.T))
            C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb * np.einsum("ij,kl", T, T)
            C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            C_Voigt = np.ascontiguousarray(C_Voigt)

        elif self.ndim == 3:
            s1 = s[0]
            s2 = s[1]
            s3 = s[2]
            T1 = np.array([[0.,0.,0.],[0.,0.,-1],[0.,1.,0.]])
            T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
            T2 = np.array([[0.,0.,-1],[0.,0.,0.],[1.,0.,0.]])
            T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
            T3 = np.array([[0.,-1,0.],[1.,0.,0.],[0.,0.,0.]])
            T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))
            s1s2 = s1 + s2
            s1s3 = s1 + s3
            s2s3 = s2 + s3
            # if (s1s2 < 2.0):
            #     s1s2 = 2.0
            # if (s1s3 < 2.0):
            #     s1s3 = 2.0
            # if (s2s3 < 2.0):
            #     s2s3 = 2.0
            lamb1 = 2. / (s1s2)
            lamb2 = 2. / (s1s3)
            lamb3 = 2. / (s2s3)

            C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb3 * np.einsum("ij,kl", T1, T1) - \
                - 2. * lamb2 * np.einsum("ij,kl", T2, T2) - 2. * lamb1 * np.einsum("ij,kl", T3, T3)
            C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            C_Voigt = np.ascontiguousarray(C_Voigt)


            # s1 = s[0]
            # s2 = s[1]
            # s3 = s[2]
            # T1 = np.array([[0.,-1.,0.],[1.,0.,0],[0.,0.,0.]])
            # T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
            # T2 = np.array([[0.,0.,0.],[0.,0., 1],[0.,-1,0.]])
            # T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
            # T3 = np.array([[0.,0.,1.],[0.,0.,0.],[-1,0.,0.]])
            # T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))
            # s1s2 = s1 + s2
            # s1s3 = s1 + s3
            # s2s3 = s2 + s3
            # # if (s1s2 < 2.0):
            # #     s1s2 = 2.0
            # # if (s1s3 < 2.0):
            # #     s1s3 = 2.0
            # # if (s2s3 < 2.0):
            # #     s2s3 = 2.0
            # lamb1 = 2. / (s1s2)
            # lamb2 = 2. / (s1s3)
            # lamb3 = 2. / (s2s3)

            # C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb1 * np.einsum("ij,kl", T1, T1) - \
            #     - 2. * lamb3 * np.einsum("ij,kl", T2, T2) - 2. * lamb2 * np.einsum("ij,kl", T3, T3)
            # C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            # C_Voigt = np.ascontiguousarray(C_Voigt)

        C_Voigt = Voigt(C_Voigt,1)

        # C_Voigt += 0.95*self.vIikIjl
        # print(C_Voigt)

        return C_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb
        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]


        u, s, vh = np.linalg.svd(F, full_matrices=True)
        R = u.dot(vh)
        # s1 = s[0]
        # s2 = s[1]

        # R,U = polar(F)
        sigma = 2. * (F - R)
        sigma = 1./J * np.dot(sigma, F.T)
        # print(sigma)


        # sigma += 0.95/J*(b - I)

        return sigma


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        # R,U = polar(F)
        u, s, vh = np.linalg.svd(F, full_matrices=True)
        R = u.dot(vh)
        energy  = einsum("ij,ij",F - R,F - R)

        return energy
