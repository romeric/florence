import numpy as np
from numpy import einsum
from Florence.VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace
from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from Florence.MaterialLibrary.MaterialBase import Material
from Florence.Tensor import trace, Voigt, makezero, issymetric


def vec(H):
    ndim = H.shape[0]
    if H.ndim == 4:
        # print(H.shape)
        # H = np.einsum("ijlk",H)
        x = H.flatten().reshape(ndim**2,ndim**2)
        return x
    else:
        # return H.flatten()
        return H.T.flatten() # careful - ARAP needs this



def svd_rv(F, full_matrices=True):

    det = np.linalg.det

    if F.shape[0] == 3:
        U, Sigma, V = np.linalg.svd(F, full_matrices=True)
        V = V.T
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
        return U, Sigma, V.T # for sym ARAP
    else:
        U, Sigma, V = np.linalg.svd(F, full_matrices=True)
        V = V.T
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
        return U, Sigma, V.T
        # return U, Sigma, V

# svd = np.linalg.svd
svd = svd_rv


def FillConstitutiveBF(B,SpatialGradient,F,ndim,nvar):

    # SpatialGradient = np.dot(F,SpatialGradient)

    if ndim == 2:
        B[::ndim,0] = SpatialGradient[0,:]
        B[::ndim,2] = SpatialGradient[1,:]
        B[1::ndim,1] = SpatialGradient[0,:]
        B[1::ndim,3] = SpatialGradient[1,:]
    else:
        B[::ndim,0] = SpatialGradient[0,:]
        B[::ndim,3] = SpatialGradient[1,:]
        B[::ndim,6] = SpatialGradient[2,:]

        B[1::ndim,1] = SpatialGradient[0,:]
        B[1::ndim,4] = SpatialGradient[1,:]
        B[1::ndim,7] = SpatialGradient[2,:]

        B[2::ndim,2] = SpatialGradient[0,:]
        B[2::ndim,5] = SpatialGradient[1,:]
        B[2::ndim,8] = SpatialGradient[2,:]


def dJdF(F):
    if F.shape[0] == 2:
        return np.array([ [F[1,1], -F[1,0]], [-F[0,1], F[0,0]] ])
    else:
        f0 = F[:,0]
        f1 = F[:,1]
        f2 = F[:,2]
        final = np.zeros((3,3))
        final[:,0] = np.cross(f1,f2);
        final[:,1] = np.cross(f2,f0);
        final[:,2] = np.cross(f0,f1);
        makezero(final)
        return final

def d2JdFdF(F):
    if F.shape[0] == 2:
        H = np.eye(4,4); H = np.fliplr(H); H[1,2] = -1; H[2,1] = -1;
        return H
    else:
        f0 = F[:,0]
        f1 = F[:,1]
        f2 = F[:,2]

        F0 = np.array([
            [     0  , -f0[2],   f0[1]],
            [  f0[2] ,      0,  -f0[0]],
            [  -f0[1],  f0[0],      0]])

        F1 = np.array([
            [     0, -f1[2],  f1[1]],
            [f1[2] ,     0 , -f1[0]],
            [-f1[1],  f1[0],      0]])

        F2 = np.array([
            [     0, -f2[2],  f2[1]],
            [f2[2] ,     0 , -f2[0]],
            [-f2[1],  f2[0],      0]])

        Z = np.zeros((3,3))
        H = np.vstack((
                np.hstack((Z,-F2,F1)),
                np.hstack((F2,Z,-F0)),
                np.hstack((-F1,F0,Z))
            ))
        return H


# delta = 1e-3
# def Jr(J):
#     return 0.5 * (J + np.sqrt(J**2 + delta**2))

# def dJrdF(F):
#     J = np.linalg.det(F)
#     return 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)

# def d2JrdFdF(F):
#     J = np.linalg.det(F)
#     djdf = dJdF(F)
#     gJ = vec(djdf)
#     dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * djdf
#     gJr = vec(dJrdF)
#     HJr = 0.5 * (1 + J / np.sqrt(J**2 + delta**2)) * d2JdFdF(F) + 0.5 * (delta**2 / (J**2 + delta**2)**(3./2.)) * np.outer(gJ,gJ)
#     return HJr


def LocallyInjectiveFunction(J):
    return 1. / (J**3 - 3 * J**2 + 3 * J)

def LocallyInjectiveGradient(J, gJ):
    return 3.*(-J**2 + 2*J - 1)/(J**2*(J**2 - 3*J + 3)**2) * gJ

def LocallyInjectiveHessian(J, gJ, HJ):
    H1 = 6.*(2*J**4 - 8*J**3 + 12*J**2 - 9*J + 3)/(J**3*(J**6 - 9*J**5 + 36*J**4 - 81*J**3 + 108*J**2 - 81*J + 27)) * np.outer(gJ,gJ)
    H1 += 3.*(-J**2 + 2*J - 1)/(J**2*(J**2 - 3*J + 3)**2) * HJ
    makezero(H1, 1e-12)
    return H1


class NeoHookeanF(Material):
    """The fundamental Neo-Hookean internal energy, described in Ogden et. al.

        W(C) = mu/2*(C:I-3)- mu*lnJ + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NeoHookeanF, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        b = StrainTensors['b'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb

        # BECAREFUL IJKL OF F,F or invF,invF is not symmetric
        # For F based formulation do we need to bring everything to reference domain, partial pull back?

        # Symmetric formulation based on K. Theodore arrangements
        gJ = vec(dJdF(F))
        HJ = d2JdFdF(F)
        d2 = self.ndim**2
        H = mu * np.eye(d2,d2) + (mu + lamb * (1. - np.log(J)))/J**2 * np.outer(gJ,gJ) + (lamb * np.log(J) - mu) / J * HJ

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        stress = mu*F + (lamb*np.log(J) - mu) * dJdF(F) / J

        return stress



class PixarNeoHookeanF(Material):
    """The Neo-Hookean internal energy, described in Smith et. al.

        W(C) = mu/2*(C:I-3)- mu*(J-1) + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(PixarNeoHookeanF, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4


        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        # Symmetric formulation based on K. Theodore arrangements
        gJ = vec(dJdF(F))
        HJ = d2JdFdF(F)

        # gJ = vec(dJrdF(F))
        # HJ = d2JrdFdF(F)

        d2 = self.ndim**2
        H = mu * np.eye(d2,d2) + lamb * np.outer(gJ,gJ) + (lamb * (J-1) - mu) * HJ

        # H += 0.28 * LocallyInjectiveHessian(J, gJ, HJ)

        # H += (1 - J**(-2)) * HJ + 2. / J**(3) * np.outer(gJ,gJ)

        # s = np.linalg.svd(H)[1]
        # if np.any(s < 0):
        #     print(s)

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))


        djdf = dJdF(F)
        # djdf = dJrdF(F)

        mu = self.mu
        lamb = self.lamb
        stress = mu*F + (lamb*(J - 1.) - mu) * djdf

        # stress += 0.28 * LocallyInjectiveGradient(J, dJdF(F))

        # stress += (1 - J**(-2)) * dJdF(F)

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            # J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        # J = Jr(J)

        energy  = mu/2.*(trace(C) - 3.) - mu*(J-1) + lamb/2.*(J-1.)**2

        return energy



class MIPSF(Material):
    """The Neo-Hookean internal energy, described in Smith et. al.

        W(C) = mu/2*(C:I-3)- mu*(J-1) + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPSF, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # the smaller minJ the more chance to untangle something
        # self.delta = 1e-3
        # self.minJ = minJ
        minJ = self.minJ
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # print(self.delta)

        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        delta = self.delta

        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        if np.isclose(Jr, 0):
            Jr = 1e-10

        # Symmetric formulation based on K. Theodore arrangements
        gJ = vec(dJdF(F))
        HJ = d2JdFdF(F)
        f = vec(F)

        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)
        gJr = vec(dJrdF)
        HJr = 0.5 * (1 + J / np.sqrt(J**2 + delta**2)) * HJ + 0.5 * (delta**2 / (J**2 + delta**2)**(3./2.)) * np.outer(gJ,gJ)

        d2 = self.ndim**2
        H = 2. / d / Jr**(2./d) * np.eye(d2,d2) - 4. / d**2 * Jr**(-2./d-1.) * (np.outer(gJr,f) + np.outer(f,gJr)) +\
            2. * trc / d**2 * (2./d + 1.) * Jr**(-2./d - 2.) * np.outer(gJr,gJr) -\
            2. * trc / d**2 * Jr**(-2./d-1.) * HJr

        # Neffs
        # H += (0.4*Jr**10 + 0.6)/Jr**7 * np.outer(gJr,gJr) + 0.1*(Jr**10 - 1)/Jr**6 * HJr
        # Garanzha
        # H += (1.0/Jr**3 * np.outer(gJr,gJr) + (0.5 - 0.5/Jr**2) * HJr) * self.lamb
        # standard
        # H += self.lamb * (np.outer(gJr,gJr) + (Jr - 1.) * HJr)

        H += self.lamb * LocallyInjectiveHessian(Jr, gJr, HJr)

        # s = np.linalg.svd(H)[1]
        # if np.any(s < 0):
        #     print(s)

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        delta = self.delta

        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        if np.isclose(Jr, 0):
            # print("Small Jr", J, Jr)
            Jr = 1e-10

        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)

        stress = 2. / d / Jr**(2./d) * F - 2. * trc / d**2 * Jr**(-2./d-1.)  * dJrdF

        # Neffs
        # stress += 0.1*(Jr**10 - 1)/Jr**6 * dJrdF
        # Garanzha
        # stress += self.lamb * (0.5 - 0.5/Jr**2) * dJrdF
        # standard
        # stress += self.lamb * (Jr - 1.) * dJrdF

        stress += self.lamb * LocallyInjectiveGradient(Jr, dJrdF)

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        delta = self.delta

        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        if np.isclose(Jr, 0):
            Jr = 1e-10

        energy  = (1./d * Jr**(-2./d) * trc - 1.)

        # Neffs
        # energy += 0.02*(Jr**5 + 1./Jr**5 - 2.)
        # Garanzha
        # energy += (0.5*Jr - 1.0 + 0.5/Jr) * self.lamb
        # standard
        # energy += self.lamb * 0.5 * (Jr - 1.)**2

        energy += self.lamb * LocallyInjectiveFunction(Jr)

        return energy



class ARAPF(Material):
    """The fundamental ARAP model

        W_arap(F) = (F - R)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(ARAPF, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

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

        det = np.linalg.det
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T

        # s = np.dot(vh, np.dot(np.diag(s), vh.T))
        # print(u,s,vh)
        # exit()
        if self.ndim == 2:
            s1 = s[0]
            s2 = s[1]
            T = np.array([[0.,-1],[1,0.]])
            T = 1./np.sqrt(2.) * np.dot(u, np.dot(T, vh.T))
            t =  vec(T)
            H = np.eye(4,4)
            I_1 = s.sum()
            filtered = 2.0 / I_1 if I_1 >= 2.0 else 1.0
            # filtered = 1.0
            H -= filtered * np.outer(t,t)
            H *= 2.


        elif self.ndim == 3:
            s0 = s[0]
            s1 = s[1]
            s2 = s[2]
            T1 = np.array([[0.,-1.,0.],[1.,0.,0],[0.,0.,0.]])
            T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
            T2 = np.array([[0.,0.,0.],[0.,0., 1],[0.,-1.,0.]])
            T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
            T3 = np.array([[0.,0.,1.],[0.,0.,0.],[-1,0.,0.]])
            T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))

            s0s1 = s0 + s1
            s0s2 = s0 + s2
            s1s2 = s1 + s2
            if (s0s1 < 2.0):
                s0s1 = 2.0
            if (s0s2 < 2.0):
                s0s2 = 2.0
            if (s1s2 < 2.0):
                s1s2 = 2.0
            lamb1 = 2. / (s0s1)
            lamb2 = 2. / (s0s2)
            lamb3 = 2. / (s1s2)

            t1 = vec(T1)
            t2 = vec(T2)
            t3 = vec(T3)

            H = 2. * np.eye(9,9);
            # H = H - (4. / (s0 + s1)) * np.outer(t1 , t1);
            # H = H - (4. / (s1 + s2)) * np.outer(t2 , t2);
            # H = H - (4. / (s0 + s2)) * np.outer(t3 , t3);

            H = H - (2 * lamb1) * np.outer(t1 , t1);
            H = H - (2 * lamb3) * np.outer(t2 , t2);
            H = H - (2 * lamb2) * np.outer(t3 , t3);

            # print(H)
            # exit()

        # H += 0.1 * LocallyInjectiveHessian(J, vec(dJdF(F)), d2JdFdF(F))

        # s = np.linalg.svd(C_Voigt)[1]
        # print(s)
        # exit()

        C_Voigt = H
        self.H_VoigtSize = H.shape[0]

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

        sigma = 2. * (F - R)

        # S = np.dot(vh, np.dot(np.diag(s), vh.T))
        # sigma = 2. * R.dot(S - I)
        # print(sigma)
        # exit()

        # sigma += 0.1 * LocallyInjectiveGradient(J, dJdF(F))

        return sigma


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        R = u.dot(vh.T)
        energy  = einsum("ij,ij",F - R,F - R)

        return energy




class SymmetricARAPF(Material):
    """The fundamental ARAP model

        W_arap(F) = (F - R)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(SymmetricARAPF, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

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
        # u, s, vh = np.linalg.svd(F, full_matrices=True)
        vh = vh.T
        # print(u)
        # print(s)
        # print(vh)
        # exit()

        R = u.dot(vh.T)
        S = np.dot(vh, np.dot(np.diag(s), vh.T))
        g = vec(dJdF(F))

        J2 = J**2
        J3 = J**3
        J4 = J**4

        f = vec(F)
        r = vec(R)
        HJ = d2JdFdF(F)

        if self.ndim == 2:

            # HJ = np.eye(4,4); HJ = np.fliplr(HJ); HJ[1,2] = -1; HJ[2,1] = -1;
            I1 = trace(S)
            I2 = trace(b)

            T = np.array([[0.,-1],[1,0.]])
            T = 1./np.sqrt(2.) * np.dot(u, np.dot(T, vh.T))
            t =  vec(T)

            H = 2. * (1 + 1 / J2) * np.eye(4,4)
            H -= 4. / J3 * (np.outer(g,f) + np.outer(f,g))
            H += 2. / J2 * (np.outer(g,r) + np.outer(r,g))
            H += 2. / J2 * (I1 - I2 / J) * HJ
            H += 2. / J3 * (3. * I2 / J - 2. * I1) * np.outer(g,g)
            H -= 4. / I1 * (1. + 1. / J) * np.outer(t,t)


        elif self.ndim == 3:

            # sigma0 = S[0,0]
            # sigma1 = S[1,1]
            # sigma2 = S[2,2]

            # sigma0 = s[0]
            # sigma1 = s[1]
            # sigma2 = s[2]

            # sqrt = np.sqrt
            # # def sqrt(x): return x

            # lambdas = [
            #     (sigma0 ** 4 * sigma1 ** 3 + sigma0 ** 3 * sigma1 ** 4 - 2 * sigma0 ** 3 * sigma1 ** 3 + sigma0 ** 3 * sigma1 - sigma0 ** 3 + sigma0 * sigma1 ** 3 - sigma1 ** 3) / (sigma0 ** 3 * sigma1 ** 3 * (sigma0 + sigma1)),
            #     (sigma1 ** 4 * sigma2 ** 3 + sigma1 ** 3 * sigma2 ** 4 - 2 * sigma1 ** 3 * sigma2 ** 3 + sigma1 ** 3 * sigma2 - sigma1 ** 3 + sigma1 * sigma2 ** 3 - sigma2 ** 3) / (sigma1 ** 3 * sigma2 ** 3 * (sigma1 + sigma2)),
            #     (sigma0 ** 4 * sigma2 ** 3 + sigma0 ** 3 * sigma2 ** 4 - 2 * sigma0 ** 3 * sigma2 ** 3 + sigma0 ** 3 * sigma2 - sigma0 ** 3 + sigma0 * sigma2 ** 3 - sigma2 ** 3) / (sigma0 ** 3 * sigma2 ** 3 * (sigma0 + sigma2)),
            #     (sigma0 ** 3 * sigma1 ** 3 - sigma0 ** 2 * sigma1 + sigma0 ** 2 - sigma0 * sigma1 ** 2 + sigma0 * sigma1 + sigma1 ** 2) / (sigma0 ** 3 * sigma1 ** 3),
            #     (sigma1 ** 3 * sigma2 ** 3 - sigma1 ** 2 * sigma2 + sigma1 ** 2 - sigma1 * sigma2 ** 2 + sigma1 * sigma2 + sigma2 ** 2) / (sigma1 ** 3 * sigma2 ** 3),
            #     (sigma0 ** 3 * sigma2 ** 3 - sigma0 ** 2 * sigma2 + sigma0 ** 2 - sigma0 * sigma2 ** 2 + sigma0 * sigma2 + sigma2 ** 2) / (sigma0 ** 3 * sigma2 ** 3),
            #     (sigma0 ** 4 - 2 * sigma0 + 3) / sigma0 ** 4,
            #     (sigma1 ** 4 - 2 * sigma1 + 3) / sigma1 ** 4,
            #     (sigma2 ** 4 - 2 * sigma2 + 3) / sigma2 ** 4
            #     ]

            # qs = [[0, 0, 0, 0, 0, 0, 1, 0, 0],
            #     [sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0, 0, 0],
            #     [0, 0, -sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0],
            #     [-sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 1, 0],
            #     [0, -sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0, 0],
            #     [0, 0, sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0],
            #     [0, sqrt(2) / 2, 0, 0, sqrt(2) / 2, 0, 0, 0, 0],
            #     [0, 0, 0, 0, 0, 0, 0, 0, 1]
            #     ]
            # qs = np.array(qs).T

            # H = np.zeros((9,9))
            # # mm = 1./np.sqrt(2.)
            # mm=1.
            # for i in range(9):
            #     # Qi = qs[i].reshape(3,3)
            #     # Qi = mm * np.dot(u, np.dot(Qi, vh.T))
            #     # qi = vec(Qi)
            #     qi = qs[i]
            #     # print(lambdas[i])
            #     H += lambdas[i] * np.outer(qi,qi)
            #     # H += np.max([lambdas[i], 0.]) * np.outer(qi,qi)
            # # print(H)
            # return H


            def DFDF(index):
              # i = np.mod(index - 1, 3);
              # j = np.floor((index - 1) / 3);
              i = np.mod(index, 3);
              j = np.floor((index) / 3);
              i = int(i)
              j = int(j)

              DF = np.zeros((3,3));
              DF[i,j] = 1;
              return DF

            def IIC_Hessian(F):
                H = np.zeros((9,9))
                for i in range(9):
                    DF = DFDF(i);
                    # print(DF)
                    # exit()
                    # A = 4 * (DF * F' * F + F * F' * DF + F * DF' * F);
                    A = 4 * (DF.dot(F.T.dot(F)) + F.dot(F.T.dot(DF)) + F.dot(DF.T.dot(F)))
                    # print(A)
                    column = A.T.reshape(9);
                    H[:,i] = column;
                return H

            def IIC_Star_Hessian(F):
                IC = np.trace(F.T.dot(F))
                H = 2 * IC * np.eye(9,9);
                f = vec(F);
                H = H + 4 * np.outer(f,f);
                # print(H)

                IIC_H = IIC_Hessian(F);
                H = H - 2 * (IIC_H / 4.);
                return H


            # F = np.arange(9) + 1; F = F.reshape(3,3).T; F[2,2] = 50; F=F.astype(float)
            # # print(F)
            # xx = IIC_Star_Hessian(F)
            # # xx = IIC_Hessian(F)
            # # xx = IIC_Hessian(F)
            # print(xx)
            # exit()


            C = F.T.dot(F)
            IC = trace(C)
            IIC = trace(C.dot(C))
            IIStarC = 0.5 * (IC**2 - IIC)
            dIIStarC = 2 * IC * F - 2 * np.dot(F,np.dot(F.T,F))
            t = vec(dIIStarC)
            d2IIStarC = IIC_Star_Hessian(F)

            H  = 2. * np.eye(9,9)
            H -= 2. / J3 * (np.outer(g,t) + np.outer(t,g))
            H += 6. * IIStarC / J4 * np.outer(g,g)
            H += 1. / J2 * d2IIStarC
            H -= 2. * IIStarC / J3 * d2JdFdF(F)
            # print(H)

            s0 = s[0]
            s1 = s[1]
            s2 = s[2]
            # s0 = S[0,0]
            # s1 = S[1,1]
            # s2 = S[2,2]
            T1 = np.array([[0.,-1.,0.],[1.,0.,0],[0.,0.,0.]])
            T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
            T2 = np.array([[0.,0.,0.],[0.,0., 1],[0.,-1.,0.]])
            T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
            T3 = np.array([[0.,0.,1.],[0.,0.,0.],[-1,0.,0.]])
            T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))

            s0s1 = s0 + s1
            s0s2 = s0 + s2
            s1s2 = s1 + s2
            # if (s0s1 < 2.0):
            #     s0s1 = 2.0
            # if (s0s2 < 2.0):
            #     s0s2 = 2.0
            # if (s1s2 < 2.0):
            #     s1s2 = 2.0
            lamb1 = 2. / (s0s1)
            lamb2 = 2. / (s0s2)
            lamb3 = 2. / (s1s2)

            t1 = vec(T1)
            t2 = vec(T2)
            t3 = vec(T3)

            dRdF  = (lamb1) * np.outer(t1 , t1);
            dRdF += (lamb3) * np.outer(t2 , t2);
            dRdF += (lamb2) * np.outer(t3 , t3);
            # print(dRdF)

            IS  = trace(S)
            IIS = trace(b)
            IIStarS = 0.5 * (IS*IS - IIS)

            newH = (1 + IS / J) * dRdF + (1 / J) * np.outer(r,r)
            newH = newH - (IS / J2) * (np.outer(g,r) + np.outer(r,g))
            newH = newH + (2.0 * IIStarS / J3) * np.outer(g,g)
            newH = newH - (IIStarS / J2) * HJ
            newH = newH + (1 / J2) * (np.outer(g,f) + np.outer(f,g))
            newH = newH - (1/J) * np.eye(9,9);
            H = H - 2.0 * newH
            H = H / 2.
            makezero(H)

            # print(H)
            # exit()

        # s = np.linalg.svd(C_Voigt)[1]
        # print(s)
        # exit()

        C_Voigt = H
        self.H_VoigtSize = H.shape[0]

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
        S = np.dot(vh, np.dot(np.diag(s), vh.T))

        J2 = J**2
        J3 = J**3

        I1 = trace(S)
        I2 = trace(b)

        if self.ndim == 2:
            sigma = 2. * (1. + 1. / J2) * F - 2. * (1. + 1. / J) * R + (2. / J2) * (I1 - I2 / J) * dJdF(F)
        else:
            djdf = dJdF(F)
            C = np.dot(F.T,F)
            IC = trace(C);
            IIC = trace(C.dot(C));
            IIStarC = 0.5 * (IC*IC - IIC);
            IS = trace(S);
            # IIS = trace(S.dot(S));
            IIS = I2
            IIStarS = 0.5 * (IS*IS - IIS);

            # % here's symmetric dirichlet
            dIIStarC = 2 * IC * F - 2. * b.dot(F)
            P = 2 * F + dIIStarC / J2 - (2. / J3) * IIStarC * djdf;
            P = P - 2 * ((1 + IS / J) * R - (IIStarS / J2) * djdf - (1. / J) * F);
            P = P/2.;
            sigma = P
            makezero(sigma,1e-12)
            # print(sigma)

        return sigma




class Corotational(Material):
    """The fundamental ARAP model

        W_arap(F) = (F - R)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(Corotational, self).__init__(mtype, ndim, **kwargs)
        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

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

        det = np.linalg.det
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T


        R = u.dot(vh.T)
        S = np.dot(vh, np.dot(np.diag(s), vh.T))
        # print()
        IS = trace(S)
        kterm = IS - self.ndim
        if self.ndim == 2:
            r = 1./np.sqrt(2) * vec(R)
            s1 = s[0]
            s2 = s[1]
            T = np.array([[0.,-1],[1,0.]])
            T = 1./np.sqrt(2) * np.dot(u, np.dot(T, vh.T))
            t =  vec(T)
            L = np.array([[0.,1],[1,0.]])
            L = 1./np.sqrt(2) * np.dot(u, np.dot(L, vh.T))
            l = vec(L)
            P = np.array([[1.,0],[0,-1.]])
            P = 1./np.sqrt(2) * np.dot(u, np.dot(P, vh.T))
            p = vec(P)

            H = 2. * np.eye(4,4)
            H += 2. * (kterm - 2) / (s1 + s2) * np.outer(t,t)
            H += 2 * np.outer(r,r)


            # # I_1 = S.sum()
            # I_1 = s1 + s2
            # H = 2. * mu * np.eye(4,4)
            # # H = 2. * mu * np.outer(p,p)
            # # H += 2. * mu * np.outer(l,l)
            # # print(H)
            # # exit()
            # H += (0. * mu + 2 * lamb * (I_1 - 2 - 2 * mu) / I_1) * np.outer(t,t)
            # H += (0. * mu + 2 * lamb) * np.outer(r,r)
            # # print(H)
            # # exit()


        elif self.ndim == 3:
            r = 1./np.sqrt(3) * vec(R)
            s0 = s[0]
            s1 = s[1]
            s2 = s[2]
            T1 = np.array([[0.,-1.,0.],[1.,0.,0],[0.,0.,0.]])
            T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
            T2 = np.array([[0.,0.,0.],[0.,0., 1],[0.,-1.,0.]])
            T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
            T3 = np.array([[0.,0.,1.],[0.,0.,0.],[-1,0.,0.]])
            T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))

            s0s1 = s0 + s1
            s0s2 = s0 + s2
            s1s2 = s1 + s2
            if (s0s1 < 2.0):
                s0s1 = 2.0
            if (s0s2 < 2.0):
                s0s2 = 2.0
            if (s1s2 < 2.0):
                s1s2 = 2.0
            lamb1 = 2. / (s0s1)
            lamb2 = 2. / (s0s2)
            lamb3 = 2. / (s1s2)

            t1 = vec(T1)
            t2 = vec(T2)
            t3 = vec(T3)

            H = 2. * np.eye(9,9)
            H += 2. * (kterm - 2) / (s0s1) * np.outer(t1,t1)
            H += 2. * (kterm - 2) / (s1s2) * np.outer(t2,t2)
            H += 2. * (kterm - 2) / (s0s2) * np.outer(t3,t3)
            H += 3 * np.outer(r,r)


        C_Voigt = H
        self.H_VoigtSize = H.shape[0]

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
        S = np.dot(vh, np.dot(np.diag(s), vh.T))
        IS = trace(S)

        sigma = 2. * (F - R) + (IS - self.ndim) * R
        # sigma = 2. * mu * (F - R) + 1 * lamb * (IS - 2) * R
        # print(sigma)
        # exit()

        return sigma










__all__ = ["FBasedDisplacementFormulation"]

class FBasedDisplacementFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False, quadrature_degree=None):

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(FBasedDisplacementFormulation, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces,
            compute_post_quadrature=compute_post_quadrature)

        self.fields = "mechanics"
        self.nvar = self.ndim

        self.GetQuadraturesAndFunctionSpaces(mesh, variables_order=variables_order,
            quadrature_rules=quadrature_rules, quadrature_type=quadrature_type,
            function_spaces=function_spaces, compute_post_quadrature=compute_post_quadrature,
            equally_spaced_bases=equally_spaced_bases, quadrature_degree=quadrature_degree)



    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, TotalPot):

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

        # if True:
        if False:
            LagrangeElemCoords = self.GetIdealElement(elem, fem_solver, function_space, LagrangeElemCoords)

        # COMPUTE THE STIFFNESS MATRIX
        stiffnessel, t = self.GetLocalStiffness(function_space,material,
                LagrangeElemCoords,EulerElemCoords,fem_solver,elem)

        I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed is False:
            # COMPUTE THE MASS MATRIX
            if material.has_low_level_dispatcher:
                massel = self.__GetLocalMass__(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
            else:
                massel = self.GetLocalMass(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)


        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)
        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed is False:
            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem



    def GetElementalMatricesInVectorForm(self, elem, function_space, mesh, material, fem_solver, Eulerx, TotalPot):

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

        # COMPUTE THE TRACTION VECTOR
        t = self.GetLocalTraction(function_space,material,
            LagrangeElemCoords,EulerElemCoords,fem_solver,elem)

        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed is False:
            # COMPUTE THE MASS MATRIX
            if material.has_low_level_dispatcher:
                # massel = self.__GetLocalMass__(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
                massel = self.__GetLocalMass_Efficient__(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
            else:
                # massel = self.GetLocalMass(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
                massel = self.GetLocalMass_Efficient(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)

            if fem_solver.analysis_subtype == "explicit" and fem_solver.mass_type == "lumped":
                massel = self.GetLumpedMass(massel)


        return t, f, massel



    def GetLocalStiffness(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get stiffness matrix of the system"""

        nvar = self.nvar
        ndim = self.ndim
        nodeperelem = function_space.Bases.shape[0]

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = function_space.Jm
        AllGauss = function_space.AllGauss

        # ALLOCATE
        stiffness = np.zeros((nodeperelem*nvar,nodeperelem*nvar),dtype=np.float64)
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        B = np.zeros((nodeperelem*nvar,material.H_VoigtSize),dtype=np.float64)

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # ParentGradientX = [np.eye(3,3)]
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
        SpatialGradient = np.einsum('ikj', MaterialGradient)
        # COMPUTE ONCE detJ
        detJ = np.einsum('i,i->i', AllGauss[:,0], det(ParentGradientX))
        # detJ = np.einsum('i,i,i->i', AllGauss[:,0], det(ParentGradientX), StrainTensors['J'])

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = material.Hessian(StrainTensors,None,elem,counter)

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:], StrainTensors['F'][counter],
                CauchyStressTensor, H_Voigt, requires_geometry_update=fem_solver.requires_geometry_update)

            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        makezero(stiffness, 1e-12)
        # print(stiffness)
        # print(tractionforce)
        # exit()
        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, F, CauchyStressTensor, H_Voigt,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveBF(B,SpatialGradient,F,self.ndim,self.nvar)

        BDB = B.dot(H_Voigt.dot(B.T))

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            # TotalTraction = GetTotalTraction(CauchyStressTensor)
            TotalTraction = vec(CauchyStressTensor)
            # TotalTraction = vec(CauchyStressTensor.T)
            t = np.dot(B,TotalTraction)[:,None]

        return BDB, t


    def GetEnergy(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get virtual energy of the system. For dynamic analysis this is handy for computing conservation of energy.
            The routine computes the global form of virtual internal energy i.e. integral of "W(C,G,C)"". This can be
            computed purely in a Lagrangian configuration.
        """

        if True:
        # if False:
            LagrangeElemCoords = self.GetIdealElement(elem, fem_solver, function_space, LagrangeElemCoords)

        nvar = self.nvar
        ndim = self.ndim
        nodeperelem = function_space.Bases.shape[0]

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = function_space.Jm
        AllGauss = function_space.AllGauss

        internal_energy = 0.

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        detJ = np.einsum('i,i->i', AllGauss[:,0], det(ParentGradientX))

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):
            # COMPUTE THE INTERNAL ENERGY AT THIS GAUSS POINT
            energy = material.InternalEnergy(StrainTensors,elem,counter)
            # INTEGRATE INTERNAL ENERGY
            internal_energy += energy*detJ[counter]

        return internal_energy
