import numpy as np
from numpy import einsum
from Florence.Tensor import trace, Voigt, makezero
from .MaterialBase import Material
from Florence.LegendreTransform import LegendreTransform
from scipy.linalg import polar


def svd_rv(F, full_matrices=True):

    det = np.linalg.det

    if F.shape[0] == 3:
        U, Sigma, V = np.linalg.svd(F, full_matrices=True)
        # reflection matrix
        L = np.eye(3,3);
        L[2,2] = det(np.dot(U, V.T))

        # see where to pull the reflection out of
        detU = det(U);
        detV = det(V);
        if (detU < 0 and detV > 0):
          U = np.dot(U, L)
        elif (detU > 0 and detV < 0):
          V = np.dot(V, L)

        # push the reflection to the diagonal
        Sigma = np.dot(Sigma, L)
        return U, Sigma, V
    else:
        U, Sigma, V = np.linalg.svd(F, full_matrices=True)
        # reflection matrix
        L = np.eye(2,2);
        L[1,1] = det(np.dot(U, V.T))

        # see where to pull the reflection out of
        detU = det(U);
        detV = det(V);
        if (detU < 0 and detV > 0):
          U = np.dot(U, L)
        elif (detU > 0 and detV < 0):
          V = np.dot(V, L)

        # push the reflection to the diagonal
        Sigma = np.dot(Sigma, L)
        return U, Sigma, V


# svd = np.linalg.svd
svd = svd_rv

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
        lamb = self.lamb
        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        det = np.linalg.det
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        # print(det(u),det(vh))
        # exit()
        if self.ndim == 2:
            s1 = s[0]
            s2 = s[1]
            T = np.array([[0.,-1],[1,0.]])
            s1s2 = s1 + s2
            if (s1s2 < 2.0):
                s1s2 = 2.0
            lamb = 2. / (s1s2)
            T = 1./np.sqrt(2) * np.dot(u, np.dot(T, vh.T))
            # C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb * np.einsum("ij,kl", T, T)
            # C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)

            C_Voigt = 2.0 * ( einsum("ik,jl",I,I)) - 2. * lamb * np.einsum("ij,kl", T, T)
            # C_Voigt = 2.0 * ( einsum("ij,kl",I,I)) - 2. * lamb * np.einsum("ij,kl", T, T)
            C_Voigt = 1./ J * np.einsum("jJ,iJkL,lL->ijkl", F, C_Voigt, F)
            # Exclude the stress term from this
            R = u.dot(vh.T)
            sigma = 2. * (F - R)
            sigma = 1./J * np.dot(sigma, F.T)
            C_Voigt -= np.einsum("ij,kl", sigma, I)
            # print(C_Voigt)
            # print(np.linalg.norm(T.flatten()))
            # print(Voigt(np.einsum("ij,kl", T, T),1))
            # print(np.einsum("ij,kl", T, T))
            # exit()

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
            if (s1s2 < 2.0):
                s1s2 = 2.0
            if (s1s3 < 2.0):
                s1s3 = 2.0
            if (s2s3 < 2.0):
                s2s3 = 2.0
            lamb1 = 2. / (s1s2)
            lamb2 = 2. / (s1s3)
            lamb3 = 2. / (s2s3)

            # C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb3 * np.einsum("ij,kl", T1, T1) - \
            #     - 2. * lamb2 * np.einsum("ij,kl", T2, T2) - 2. * lamb1 * np.einsum("ij,kl", T3, T3)
            # C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            # C_Voigt = np.ascontiguousarray(C_Voigt)

            C_Voigt = 2.0 * ( einsum("ik,jl",I,I)) - 2. * lamb3 * np.einsum("ij,kl", T1, T1) - \
                - 2. * lamb2 * np.einsum("ij,kl", T2, T2) - 2. * lamb1 * np.einsum("ij,kl", T3, T3)
            C_Voigt = 1./ J * np.einsum("jJ,iJkL,lL->ijkl", F, C_Voigt, F)
            C_Voigt = np.ascontiguousarray(C_Voigt)

            R = u.dot(vh.T)
            sigma = 2. * (F - R)
            sigma = 1./J * np.dot(sigma, F.T)
            C_Voigt -= np.einsum("ij,kl",sigma,I)


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

            # # C_Voigt = 1.0 * ( einsum("ik,jl",I,I)+einsum("il,jk",I,I) ) - 2. * lamb1 * np.einsum("ij,kl", T1, T1) - \
            # #     - 2. * lamb3 * np.einsum("ij,kl", T2, T2) - 2. * lamb2 * np.einsum("ij,kl", T3, T3)
            # # C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            # # C_Voigt = np.ascontiguousarray(C_Voigt)


            # C_Voigt = 2.0 * ( einsum("ik,jl",I,I)) - 2. * lamb1 * np.einsum("ij,kl", T1, T1) - \
            #     - 2. * lamb3 * np.einsum("ij,kl", T2, T2) - 2. * lamb2 * np.einsum("ij,kl", T3, T3)
            # C_Voigt = 1./ J * np.einsum("kI,lJ,iIjJ->iklj", F, F, C_Voigt)
            # C_Voigt = np.ascontiguousarray(C_Voigt)

            # R = u.dot(vh.T)
            # sigma = 2. * (F - R)
            # sigma = 1./J * np.dot(sigma, F.T)
            # C_Voigt -= np.einsum("ij,kl",sigma,I)


        C_Voigt = Voigt(C_Voigt,1)
        makezero(C_Voigt)

        # C_Voigt = np.eye(3,3)*2

        # C_Voigt += 0.05*self.vIikIjl
        # print(C_Voigt)
        # s = svd(C_Voigt)[1]
        # print(s)

        return C_Voigt



    def CauchyStress(self,StrainTensors,ElectricDisplacementx,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        R = u.dot(vh.T)
        # s1 = s[0]
        # s2 = s[1]
        # print(F)

        # R,U = polar(F)
        sigma = 2. * (F - R)
        sigma = 1./J * np.dot(sigma, F.T)
        # print(sigma)


        # sigma += 0.05/J*(b - I)

        return sigma


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        # R,U = polar(F)
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        R = u.dot(vh.T)
        energy  = einsum("ij,ij",F - R,F - R)

        return energy
