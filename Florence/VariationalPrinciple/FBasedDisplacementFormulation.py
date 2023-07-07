# -*- coding: utf-8 -*-
#
# EVERYTHING HERE IS EXPERMENTAL AND NOT FULLY PART OF FLORENCE - JUST ACADEMIC PROTOTYPES
#
import numpy as np
from numpy import einsum
import scipy as sp
import itertools
from Florence.VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace
from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from Florence.MaterialLibrary.MaterialBase import Material
from Florence.Tensor import trace, Voigt, makezero, issymetric
norm = np.linalg.norm
outer = np.outer
# np.set_printoptions(precision=16)

def vec(H):
    ndim = H.shape[0]
    if H.ndim == 4:
        x = H.flatten().reshape(ndim**2,ndim**2)
        return x
    else:
        return H.T.flatten() # careful - ARAP needs this


def cross(A, B) :
    C = np.zeros((3 ,3))

    C[0 ,0] = A[1 ,1]*B[2 ,2] - A[1 ,2]*B[2 ,1] - A[2 ,1]*B[1 ,2] + A [2 ,2]* B[1 ,1]
    C[0 ,1] = A[1 ,2]*B[2 ,0] - A[1 ,0]*B[2 ,2] + A[2 ,0]*B[1 ,2] - A [2 ,2]* B[1 ,0]
    C[0 ,2] = A[1 ,0]*B[2 ,1] - A[1 ,1]*B[2 ,0] - A[2 ,0]*B[1 ,1] + A [2 ,1]* B[1 ,0]
    C[1 ,0] = A[0 ,2]*B[2 ,1] - A[0 ,1]*B[2 ,2] + A[2 ,1]*B[0 ,2] - A [2 ,2]* B[0 ,1]
    C[1 ,1] = A[0 ,0]*B[2 ,2] - A[0 ,2]*B[2 ,0] - A[2 ,0]*B[0 ,2] + A [2 ,2]* B[0 ,0]
    C[1 ,2] = A[0 ,1]*B[2 ,0] - A[0 ,0]*B[2 ,1] + A[2 ,0]*B[0 ,1] - A [2 ,1]* B[0 ,0]
    C[2 ,0] = A[0 ,1]*B[1 ,2] - A[0 ,2]*B[1 ,1] - A[1 ,1]*B[0 ,2] + A [1 ,2]* B[0 ,1]
    C[2 ,1] = A[0 ,2]*B[1 ,0] - A[0 ,0]*B[1 ,2] + A[1 ,0]*B[0 ,2] - A [1 ,2]* B[0 ,0]
    C[2 ,2] = A[0 ,0]*B[1 ,1] - A[0 ,1]*B[1 ,0] - A[1 ,0]*B[0 ,1] + A [1 ,1]* B[0 ,0]

    return C



def levi_civita(dim):
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim))
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x] = np.linalg.det(mat)
    return arr



def Get_FxIxF(F):
    # This is not the same as what appears in SIGGRAPH paper supplemental
    # This is same as Fastor only last minute (the last einsum indices change from iIjJ to IiJj)
    # to make it compatible with the vec function
    ndim = F.shape[0]
    E = levi_civita(ndim)

    # For I, both iiIjJ and IiJj flatten to diagonal matrix
    I = einsum("ij,IJ->iIjJ", np.eye(ndim), np.eye(ndim))

    # Note that if we did this that is iIjJ to IiJj then IxF would be the same as d2JdFdF
    # IxF = einsum("jpq,JPQ,iIpP,qQ->IiJj", E, E, I_, F)

    IxF = einsum("jpq,JPQ,iIpP,qQ->iIjJ", E, E, I, F)
    FxIxF = einsum("ipq,IPQ,qQjJ,pP->IiJj", E, E, IxF, F) # only this changes from Fastor style iIjJ to IiJj
    fxIxf = vec(FxIxF) # to make this work as expected

    return fxIxf


def cbrts(z):
    # Get all 3 cube roots of a function
    z = complex(z)
    x = z.real
    y = z.imag
    resArg = [ (np.arctan2(y,x)+2*np.pi*n)/3. for n in range(1,4) ]
    return np.array([  abs(z)**(1./3) * (np.cos(a) + np.sin(a)*1j) for a in resArg ])

# @jit(nopython=True)
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


def dRdF(u, s, vh, regularise=False):

    ndim = u.shape[0]

    if ndim == 3:

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
        if regularise:
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

        gR  = (lamb1) * np.outer(t1 , t1)
        gR += (lamb3) * np.outer(t2 , t2)
        gR += (lamb2) * np.outer(t3 , t3)

    elif ndim == 2:

        s1 = s[0]
        s2 = s[1]
        T = np.array([[0.,-1],[1,0.]])
        T = 1./np.sqrt(2.) * np.dot(u, np.dot(T, vh.T))
        t =  vec(T)
        I_1 = s.sum()
        filtered = 2.0 / I_1
        if regularise:
            filtered = 2.0 / I_1 if I_1 >= 2.0 else 1.0
        gR = filtered * np.outer(t,t)

    return gR



def GetEigenMatrices(u, vh):
    """ Eigenmatrices for F-based formulations
    """

    ndim = u.shape[0]

    if ndim == 3:
        # Scale modes
        D1 = np.array([[1.,0,0],[0,0,0],[0,0,0]])
        D1 = np.dot(u, np.dot(D1, vh.T))
        d1 = vec(D1)
        D2 = np.array([[0.,0,0],[0,1.,0],[0.,0,0]])
        D2 = np.dot(u, np.dot(D2, vh.T))
        d2 = vec(D2)
        D3 = np.array([[0.,0,0],[0,0.,0],[0,0,1.]])
        D3 = np.dot(u, np.dot(D3, vh.T))
        d3 = vec(D3)

        # Flip modes
        L1 = np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        L1 = 1./np.sqrt(2) * np.dot(u, np.dot(L1, vh.T))
        l1 = vec(L1)
        L2 = np.array([[0.,0.,1.],[0.,0., 0],[1.,0.,0.]])
        L2 = 1./np.sqrt(2) * np.dot(u, np.dot(L2, vh.T))
        l2 = vec(L2)
        L3 = np.array([[0.,1.,0.],[1.,0.,0.],[0,0.,0.]])
        L3 = 1./np.sqrt(2) * np.dot(u, np.dot(L3, vh.T))
        l3 = vec(L3)

        # Twist modes
        T1 = np.array([[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
        T1 = 1./np.sqrt(2) * np.dot(u, np.dot(T1, vh.T))
        t1 = vec(T1)
        T2 = np.array([[0.,0.,-1.],[0.,0., 0],[1.,0.,0.]])
        T2 = 1./np.sqrt(2) * np.dot(u, np.dot(T2, vh.T))
        t2 = vec(T2)
        T3 = np.array([[0.,-1.,0.],[1.,0.,0.],[0,0.,0.]])
        T3 = 1./np.sqrt(2) * np.dot(u, np.dot(T3, vh.T))
        t3 = vec(T3)

        return d1, d2, d3, l1, l2, l3, t1, t2, t3

    elif ndim == 2:

        # Scale modes
        D1 = np.array([[1.,0],[0,0.]])
        D1 = np.dot(u, np.dot(D1, vh.T))
        d1 = vec(D1)

        D2 = np.array([[0.,0],[0,1.]])
        D2 = np.dot(u, np.dot(D2, vh.T))
        d2 = vec(D2)

        # Flip mode
        L = np.array([[0.,1],[1,0.]])
        L = 1./np.sqrt(2.) * np.dot(u, np.dot(L, vh.T))
        l = vec(L)

        # Twist mode
        T = np.array([[0.,-1],[1,0.]])
        T = 1./np.sqrt(2.) * np.dot(u, np.dot(T, vh.T))
        t = vec(T)

        return d1, d2, l, t


def GetInitialStiffnessPolyconvex(sigmaH, sigmaJ, F, stabilise=False, eps=1e-6):

    ndim = F.shape[0]
    if not stabilise:
        if ndim == 3:
            return d2JdFdF(sigmaH + sigmaJ * F)
        elif ndim == 2:
            if np.linalg.norm(sigmaH) > 1e-6:
                raise NotImplementedError("Polyconvex initial stiffness for 2D in this format not yet implemented")
            return sigmaJ * d2JdFdF(F)
    else:
        # eigs, vecs = sp.linalg.eigh(d2JdFdF(sigmaH + sigmaJ * F))
        # eigs[eigs < 0] = eps
        # xx = vecs.dot(np.diag(eigs).dot(vecs.T))
        # return xx

        if ndim == 3:

            # Get SVD of F
            [U, S, Vh] = svd(F, full_matrices=True); V = Vh.T
            s1 = S[0]
            s2 = S[1]
            s3 = S[2]

            [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(U, V)

            # Get eigenvalues of the following matrix
            Hw = sigmaJ * np.array([
                [0 , s3,  s2],
                [s3,  0,  s1],
                [s2, s1,   0],
                ])
            eigs, vecs = sp.linalg.eigh(Hw)

            # # Alternatively the 3 roots can be obtained as:
            # tt = complex(J**2 - I2**3 / 27., 0)
            # tt = np.sqrt(tt)
            # u = cbrts(J + tt); u.sort()
            # v = cbrts(J - tt); v.sort()
            # roots = (u + v).real
            # # Or through numpy polynomial
            # # import numpy.polynomial.polynomial as poly
            # # roots = poly.polyroots((2. * J, -I2, 0., 1.))
            # # the following does not work as python only returns principal root
            # # cbrt = lambda x: x**(1./3.)
            # # x1 = cbrt(J + tt) + cbrt(J - tt)
            # # x2 = x1 * complex(-1., -np.sqrt(3.)) / 2.
            # # x3 = x1 * complex(-1.,  np.sqrt(3.)) / 2.

            lamb4 = -sigmaJ * s1
            lamb5 = -sigmaJ * s2
            lamb6 = -sigmaJ * s3
            lamb7 =  sigmaJ * s1
            lamb8 =  sigmaJ * s2
            lamb9 =  sigmaJ * s3

            lamb1 = eigs[0]
            lamb2 = eigs[1]
            lamb3 = eigs[2]

            lamb1 = max(lamb1, eps)
            lamb2 = max(lamb2, eps)
            lamb3 = max(lamb3, eps)
            lamb4 = max(lamb4, eps)
            lamb5 = max(lamb5, eps)
            lamb6 = max(lamb6, eps)
            lamb7 = max(lamb7, eps)
            lamb8 = max(lamb8, eps)
            lamb9 = max(lamb9, eps)

            # Build sigmaJ * IxF
            ds = np.array([d1,d2,d3]).T
            HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                + lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])
            sigmaJIxF = ds.dot(HwSPD.dot(ds.T)) + \
                lamb4 * outer(l1,l1) + lamb5 * outer(l2,l2) + lamb6 * outer(l3,l3) +\
                lamb7 * outer(t1,t1) + lamb8 * outer(t2,t2) + lamb9 * outer(t3,t3)

            # vec1 = vecs[:,0]
            # vec2 = vecs[:,1]
            # vec3 = vecs[:,2]
            # e1 = vec1[0] * d1 + vec1[1] * d2 + vec1[2] * d3
            # e2 = vec2[0] * d1 + vec2[1] * d2 + vec2[2] * d3
            # e3 = vec3[0] * d1 + vec3[1] * d2 + vec3[2] * d3
            # sigmaJIxF = lamb1 * outer(e1,e1) + lamb2 * outer(e2,e2) + lamb3 * outer(e3,e3) +\
            #     lamb4 * outer(l1,l1) + lamb5 * outer(l2,l2) + lamb6 * outer(l3,l3) +\
            #     lamb7 * outer(t1,t1) + lamb8 * outer(t2,t2) + lamb9 * outer(t3,t3)

            # Get SVD of sigmaH
            [U, S, Vh] = svd(sigmaH, full_matrices=True); V = Vh.T

            s1 = S[0]
            s2 = S[1]
            s3 = S[2]

            [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(U, V)

            # Get eigenvalues of the following matrix
            Hw = np.array([
                [0 , s3,  s2],
                [s3,  0,  s1],
                [s2, s1,   0],
                ])
            eigs, vecs = sp.linalg.eigh(Hw)

            lamb1 = eigs[0]
            lamb2 = eigs[1]
            lamb3 = eigs[2]
            lamb4 = -s1
            lamb5 = -s2
            lamb6 = -s3
            lamb7 =  s1
            lamb8 =  s2
            lamb9 =  s3

            lamb1 = max(lamb1, eps)
            lamb2 = max(lamb2, eps)
            lamb3 = max(lamb3, eps)
            lamb4 = max(lamb4, eps)
            lamb5 = max(lamb5, eps)
            lamb6 = max(lamb6, eps)
            lamb7 = max(lamb7, eps)
            lamb8 = max(lamb8, eps)
            lamb9 = max(lamb9, eps)

            # Build IxsigmaH
            ds = np.array([d1,d2,d3]).T
            HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                + lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])
            IxsigmaH = ds.dot(HwSPD.dot(ds.T)) + \
                lamb4 * outer(l1,l1) + lamb5 * outer(l2,l2) + lamb6 * outer(l3,l3) +\
                lamb7 * outer(t1,t1) + lamb8 * outer(t2,t2) + lamb9 * outer(t3,t3)

            # vec1 = vecs[:,0]
            # vec2 = vecs[:,1]
            # vec3 = vecs[:,2]
            # e1 = vec1[0] * d1 + vec1[1] * d2 + vec1[2] * d3
            # e2 = vec2[0] * d1 + vec2[1] * d2 + vec2[2] * d3
            # e3 = vec3[0] * d1 + vec3[1] * d2 + vec3[2] * d3

            # IxsigmaH = lamb1 * outer(e1,e1) + lamb2 * outer(e2,e2) + lamb3 * outer(e3,e3) +\
            #     lamb4 * outer(l1,l1) + lamb5 * outer(l2,l2) + lamb6 * outer(l3,l3) +\
            #     lamb7 * outer(t1,t1) + lamb8 * outer(t2,t2) + lamb9 * outer(t3,t3)

            # Sum the two
            initial_stiffness = sigmaJIxF + IxsigmaH

            return initial_stiffness

        elif ndim == 2:

            # Get SVD of F
            [U, S, Vh] = svd(F, full_matrices=True); V = Vh.T
            s1 = S[0]
            s2 = S[1]

            [d1, d2, l, t] = GetEigenMatrices(U, V)

            # Or alternatively
            # Hw = np.array([
            #     [0 , 1],
            #     [1,  0],
            #     ])
            # eigs, vecs = sp.linalg.eigh(Hw)

            vecs = 1./np.sqrt(2.) * np.array([[1.,-1.],[1.,1.]])

            lamb1 =  sigmaJ
            lamb2 = -sigmaJ
            lamb3 = -sigmaJ
            lamb4 =  sigmaJ

            lamb1 = max(lamb1, eps)
            lamb2 = max(lamb2, eps)
            lamb3 = max(lamb3, eps)
            lamb4 = max(lamb4, eps)

            # Build sigmaJ * IxF
            ds = np.array([d1,d2]).T
            HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
            sigmaJIxF = ds.dot(HwSPD.dot(ds.T)) + lamb3 * outer(l,l) +  lamb4 * outer(t,t)

            # Get SVD of sigmaH
            if np.linalg.norm(sigmaH) > 1e-10:
                [U, S, Vh] = svd(sigmaH, full_matrices=True); V = Vh.T
                s1 = S[0]
                s2 = S[1]

                [d1, d2, l, t] = GetEigenMatrices(U, V)

                vecs = 1./np.sqrt(2.) * np.array([[1.,-1.],[1.,1.]])

                lamb1 =  1.
                lamb2 = -1.
                lamb3 = -1.
                lamb4 =  1.

                lamb1 = max(lamb1, eps)
                lamb2 = max(lamb2, eps)
                lamb3 = max(lamb3, eps)
                lamb4 = max(lamb4, eps)

                # Build sigmaJ * IxF
                ds = np.array([d1,d2]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
                IxsigmaH = ds.dot(HwSPD.dot(ds.T)) + lamb3 * outer(l,l) +  lamb4 * outer(t,t)

                # Sum the two
                initial_stiffness = sigmaJIxF + IxsigmaH

            initial_stiffness = sigmaJIxF

            return initial_stiffness




def FillConstitutiveBF(B,SpatialGradient,ndim,nvar):
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

        mu = self.mu
        lamb = self.lamb

        # BECAREFUL IJKL OF F,F or invF,invF is not symmetric
        # For F based formulation do we need to bring everything to reference domain, partial pull back?

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

        mu = self.mu
        lamb = self.lamb
        stress = mu*F + (lamb*np.log(J) - mu) * dJdF(F) / J

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        energy  = mu/2.*(trace(C) - 3.) - mu*np.log(J) + lamb/2.*np.log(J)**2

        return energy



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

        d2 = self.ndim**2
        H = mu * np.eye(d2,d2) + lamb * np.outer(gJ,gJ) + (lamb * (J-1) - mu) * HJ

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

        djdf = dJdF(F)
        stress = mu*F + (lamb*(J - 1.) - mu) * djdf

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

        energy  = mu/2.*(trace(C) - 3.) - mu*(J-1) + lamb/2.*(J-1.)**2

        return energy



class MIPSF(Material):
    """The MIPS energy

        W(F) = F:F/d/Jr^(2/d)

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
        minJ = self.minJ
        # self.delta = np.sqrt(1e-12 + min(minJ, 0.)**2 * 0.04)
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04) * 2. # embed factot 4 in the definition of delta
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2) # superbad it seems like
        self.delta = 0.

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

        trc = np.trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        delta = self.delta

        posJ = np.sqrt(J**2 + delta**2)
        Jr = 0.5 * (J + posJ)
        # Jr = J if J >= 0 else 1e-8 * J**2 / (J**2 + delta**2)**(1./2.)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)

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

        # stress = 2. / d / Jr**(2./d) * F - 2. * trc / d**2 * Jr**(-2./d-1.)  * dJrdF
        # H1 = np.exp(1./d * Jr**(-2./d) * trc) * np.outer(vec(stress), vec(stress))
        # H  = np.exp(1./d * Jr**(-2./d) * trc) * H + H1

        # Neffs
        # H += self.lamb * ((0.4*Jr**10 + 0.6)/Jr**7 * np.outer(gJr,gJr) + 0.1*(Jr**10 - 1)/Jr**6 * HJr)
        # Garanzha
        # print(self.lamb,self.delta)
        H += (1.0/Jr**3 * np.outer(gJr,gJr) + (0.5 - 0.5/Jr**2) * HJr) * self.lamb
        # standard
        # H += self.lamb * (np.outer(gJr,gJr) + (Jr - 1.) * HJr)

        # H += self.lamb * LocallyInjectiveHessian(Jr, gJr, HJr)

        if True:
        # if False:
            if self.ndim == 2:

                I2 = trc;
                I3 = J

                # Compute the rotation variant SVD of F
                u, s, vh = svd(F, full_matrices=True)
                vh = vh.T
                # R = u.dot(vh.T)
                # S = np.dot(vh, np.dot(np.diag(s), vh.T))
                S = np.dot(u, np.dot(np.diag(s), vh.T))
                I1 = trace(S)

                s1 = s[0]
                s2 = s[1]

                [d1, d2, l, t] = GetEigenMatrices(u, vh)

                # # These are eigenvalues of MIPS for when J=Jr (no regularisation)
                # lamb1 =  -1. / I3 + I2 * (I2 - alpha) / 2. / I3**3
                # lamb2 =  -1. / I3 + I2 * (I2 + alpha) / 2. / I3**3
                # lamb3 =   1. / I3 + I2 / 2. / I3**2
                # lamb4 =   1. / I3 - I2 / 2. / I3**2

                # alpha  = np.sqrt(I2**2 - 3*I3**2)
                # beta  = (s1**2 - s2**2 + alpha)/I3

                # When J != Jr (regularisation)
                sqrt = np.sqrt
                delta2 = delta**2
                s = posJ
                lamb1 =  (I2*J*s1**2 + I2*J*s2**2 + I2*s*s1**2 + I2*s*s2**2 - I2*sqrt(4*J**4 + 8*J**3*s - 20*J**2*s**2 + J**2*s1**4 - 2*J**2*s1**2*s2**2 + J**2*s2**4 - 24*J*s**3 + 2*J*s*s1**4 - 4*J*s*s1**2*s2**2 + 2*J*s*s2**4 + 36*s**4 + s**2*s1**4 - 2*s**2*s1**2*s2**2 + s**2*s2**4) - 8*J*s**2 + 4*s**3)/(2*s**3*(J + s))
                lamb2 =  (I2*J*s1**2 + I2*J*s2**2 + I2*s*s1**2 + I2*s*s2**2 + I2*sqrt(4*J**4 + 8*J**3*s - 20*J**2*s**2 + J**2*s1**4 - 2*J**2*s1**2*s2**2 + J**2*s2**4 - 24*J*s**3 + 2*J*s*s1**4 - 4*J*s*s1**2*s2**2 + 2*J*s*s2**4 + 36*s**4 + s**2*s1**4 - 2*s**2*s1**2*s2**2 + s**2*s2**4) - 8*J*s**2 + 4*s**3)/(2*s**3*(J + s))
                lamb3 =  (I2 + 2*s)/(s*(J + s))
                lamb4 =  (-I2 + 2*s)/(s*(J + s))

                beta =  (I3*s1**2 - I3*s2**2 + s*s1**2 - s*s2**2 + sqrt(4*I3**4 + 8*I3**3*s - 20*I3**2*s**2 + I3**2*s1**4 - 2*I3**2*s1**2*s2**2 + I3**2*s2**4 - 24*I3*s**3 + 2*I3*s*s1**4 - 4*I3*s*s1**2*s2**2 + 2*I3*s*s2**4 + 36*s**4 + s**2*s1**4 - 2*s**2*s1**2*s2**2 + s**2*s2**4))/(2*(-I3**2 - I3*s + 3*s**2))

                # Project to SPD if needed
                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)

                # Scaling modes do not decouple for MIPS
                gamma  = np.sqrt(1. + beta**2) # normaliser

                # Coupled scaling modes
                e1 = 1. / gamma * (beta * d1 + d2)
                e2 = 1. / gamma * (d1 - beta * d2)

                d1 = e1
                d2 = e2

                H = lamb1 * np.outer(d1, d1) + lamb2 * np.outer(d2, d2) + lamb3 * np.outer(l, l) + lamb4 * np.outer(t, t)

                # print(H)
                # print(H1)
                # exit()

            else:

                I2 = trc
                I3 = J

                # Compute the rotation variant SVD of F
                u, s, vh = svd(F, full_matrices=True)
                vh = vh.T
                S = np.dot(u, np.dot(np.diag(s), vh.T))
                I1 = np.sum(s)

                s1 = s[0]
                s2 = s[1]
                s3 = s[2]

                [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(u, vh)

                # Note: fractional powers work different - if negative base (J) is encountered they explode
                # But make sure symbolically generated values are fractionalsed i.e. add dots(.)
                # singular values based
                # A = np.array([[2.*(2*s1**2 + 5*s2**2 + 5*s3**2)/(27.*s1**2*(s1*s2*s3)**(2./3)), 4*s3*(-2*s1**2 - 2.*s2**2 + s3**2)/(27.*(s1*s2*s3)**(5./3)), 4*s2*(-2*s1**2 + s2**2 - 2*s3**2)/(27.*(s1*s2*s3)**(5./3))],
                #     [4*s3*(-2*s1**2 - 2*s2**2 + s3**2)/(27.*(s1*s2*s3)**(5./3)), 2*(5*s1**2 + 2*s2**2 + 5*s3**2)/(27.*s2**2*(s1*s2*s3)**(2./3)), 4*s1*(s1**2 - 2*s2**2 - 2*s3**2)/(27.*(s1*s2*s3)**(5./3))],
                #     [4*s2*(-2*s1**2 + s2**2 - 2*s3**2)/(27.*(s1*s2*s3)**(5./3)), 4*s1*(s1**2 - 2*s2**2 - 2*s3**2)/(27.*(s1*s2*s3)**(5./3)), 2*(5*s1**2 + 5*s2**2 + 2*s3**2)/(27.*s3**2*(s1*s2*s3)**(2./3))]])

                # invariants based
                A = np.array([[2.*(5.*I2 - 3.*s1**2)/(27.*I3**(2./3.)*s1**2), (-8*I2*s3 + 12*s3**3)/(27*I3**(5./3)), (-8*I2*s2 + 12*s2**3)/(27*I3**(5./3))],
                    [(-8.*I2*s3 + 12*s3**3)/(27*I3**(5./3)), 2*(5*I2 - 3*s2**2)/(27*I3**(2./3)*s2**2), (-8*I2*s1 + 12*s1**3)/(27*I3**(5./3))],
                    [(-8.*I2*s2 + 12*s2**3)/(27*I3**(5./3)), (-8*I2*s1 + 12*s1**3)/(27*I3**(5./3)), 2*(5*I2 - 3*s3**2)/(27*I3**(2./3)*s3**2)]])

                # print(A)
                eigs, vecs = sp.linalg.eigh(A)
                # eigs, vecs = sp.linalg.eigh(A, driver="syev")
                # vecs = vecs.T
                vec1 = vecs[:,0]
                vec2 = vecs[:,1]
                vec3 = vecs[:,2]

                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 = eigs[2]

                # # singular value based - both work in this case
                # lamb4 =  2.*s1*(-s1**2 -   s2**2 + 3*s2*s3 - s3**2)/(9.*(s1*s2*s3)**(5./3))
                # lamb5 =  2.*s2*(-s1**2 + 3*s1*s3 -   s2**2 - s3**2)/(9.*(s1*s2*s3)**(5./3))
                # lamb6 =  2.*s3*(-s1**2 + 3*s1*s2 -   s2**2 - s3**2)/(9.*(s1*s2*s3)**(5./3))
                # lamb7 =  2.*s1*( s1**2 +   s2**2 + 3*s2*s3 + s3**2)/(9.*(s1*s2*s3)**(5./3))
                # lamb8 =  2.*s2*( s1**2 + 3*s1*s3 +   s2**2 + s3**2)/(9.*(s1*s2*s3)**(5./3))
                # lamb9 =  2.*s3*( s1**2 + 3*s1*s2 +   s2**2 + s3**2)/(9.*(s1*s2*s3)**(5./3))

                # invariant based
                lamb4 =  -2.*I2*s1/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))
                lamb5 =  -2.*I2*s2/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))
                lamb6 =  -2.*I2*s3/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))
                lamb7 =   2.*I2*s1/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))
                lamb8 =   2.*I2*s2/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))
                lamb9 =   2.*I2*s3/(9.*J**(5./3.)) + 2./(3.*J**(2./3.))

                e1 = vec1[0] * d1 + vec1[1] * d2 + vec1[2] * d3
                e2 = vec2[0] * d1 + vec2[1] * d2 + vec2[2] * d3
                e3 = vec3[0] * d1 + vec3[1] * d2 + vec3[2] * d3

                # Or equivalently
                # e1 = np.dot(u, np.dot(np.diag(vec1),vh.T))
                # e2 = np.dot(u, np.dot(np.diag(vec2),vh.T))
                # e3 = np.dot(u, np.dot(np.diag(vec3),vh.T))
                # e1 = vec(e1)
                # e2 = vec(e2)
                # e3 = vec(e3)

                # No need for normalisation
                # e1 /= np.linalg.norm(e1)
                # e2 /= np.linalg.norm(e2)
                # e3 /= np.linalg.norm(e3)

                # Project to SPD if needed
                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)
                    lamb5 = max(lamb5, eps)
                    lamb6 = max(lamb6, eps)
                    lamb7 = max(lamb7, eps)
                    lamb8 = max(lamb8, eps)
                    lamb9 = max(lamb9, eps)

                H1 = lamb1 * outer(e1,e1) + lamb2 * outer(e2,e2) + lamb3 * outer(e3,e3) +\
                    lamb4 * outer(t1,t1) + lamb5 * outer(t2,t2) + lamb6 * outer(t3,t3) +\
                    lamb7 * outer(l1,l1) + lamb8 * outer(l2,l2) + lamb9 * outer(l3,l3)

                H = H1

        # Numerically project to SPD
        # if True:
        if False:
            vals, v = np.linalg.eigh(H)
            vals[vals < 0.] = 0.
            H = np.dot(v, np.dot(np.diag(vals),np.linalg.inv(v)))

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = float(self.ndim)
        delta = self.delta

        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        # Jr = J if J >= 0 else 1e-8 * J**2 / (J**2 + delta**2)**(1./2.)
        if np.isclose(Jr, 0):
            # print("Small Jr", J, Jr)
            Jr = max(1e-8, Jr)

        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)

        stress = 2. / d / Jr**(2./d) * F - 2. * trc / d**2 * Jr**(-2./d-1.)  * dJrdF
        # stress *= np.exp(1./d * Jr**(-2./d) * trc)

        # print(self.lamb)
        # Neffs
        # stress += self.lamb * 0.1 * (Jr**10 - 1)/Jr**6 * dJrdF
        # Garanzha
        stress += self.lamb * (0.5 - 0.5/Jr**2) * dJrdF
        # standard
        # stress += self.lamb * (Jr - 1.) * dJrdF

        # stress += self.lamb * LocallyInjectiveGradient(Jr, dJrdF)

        # if J < 0:
        #     stress = 1. / delta * F
        #     # stress = delta * F
        #     # print(delta)

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        delta = self.delta

        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        # Jr = J if J >= 0 else 1e-8 * J**2 / (J**2 + delta**2)**(1./2.)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)

        # energy  = (1./d * Jr**(-2./d) * trc - 1.)
        energy  = (1./d * Jr**(-2./d) * trc) # avoid sign switches
        # energy  = np.exp(1./d * Jr**(-2./d) * trc)

        # Neffs
        # energy += self.lamb * 0.02 * (Jr**5 + 1./Jr**5 - 2.)
        # Garanzha
        # energy += 0.5 * (Jr + 1./Jr - 2.) * self.lamb
        energy += 0.5 * (Jr + 1./Jr) * self.lamb  # avoid sign switches
        # standard
        # energy += self.lamb * 0.5 * (Jr - 1.)**2

        # energy += self.lamb * LocallyInjectiveFunction(Jr)
        # if (energy <= 0.):
            # print(energy)

        # if J < 0:
        #     energy = trc / 2. / delta
        #     # energy = trc / 2. * delta

        return energy





class MIPSF2(Material):
    """The MIPS energy

        W(F) = Fr:Fr/d/Jr^(2/d)

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPSF2, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # the smaller minJ the more chance to untangle something
        minJ = self.minJ
        # self.delta = np.sqrt(1e-12 + min(minJ, 0.)**2 * 0.04)
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04) * 2. # embed factot 4 in the definition of delta
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2) # superbad it seems like
        # self.delta = .7

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

        posJ = np.sqrt(J**2 + delta**2)
        Jr = 0.5 * (J + posJ)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)

        # gJ = vec(dJdF(F))
        # HJ = d2JdFdF(F)
        # f = vec(F)

        # dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)
        # gJr = vec(dJrdF)
        # HJr = 0.5 * (1 + J / np.sqrt(J**2 + delta**2)) * HJ + 0.5 * (delta**2 / (J**2 + delta**2)**(3./2.)) * np.outer(gJ,gJ)

        # d2 = self.ndim**2
        # H = 2. / d / Jr**(2./d) * np.eye(d2,d2) - 4. / d**2 * Jr**(-2./d-1.) * (np.outer(gJr,f) + np.outer(f,gJr)) +\
        #     2. * trc / d**2 * (2./d + 1.) * Jr**(-2./d - 2.) * np.outer(gJr,gJr) -\
        #     2. * trc / d**2 * Jr**(-2./d-1.) * HJr

        # H1 = J / Jr**2 * HJr - 2. * J / Jr**3 * np.outer(gJr,gJr) - 1. / Jr * HJ + 1. / Jr**2 * (np.outer(gJ,gJr) + np.outer(gJr,gJ))
        # H1 *= 2.
        # H += H1

        # if self.ndim == 2 and False:
        if self.ndim == 2 and True:

            I2 = trc
            I3 = J

            # Compute the rotation variant SVD of F
            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            # R = u.dot(vh.T)
            # S = np.dot(vh, np.dot(np.diag(s), vh.T))
            S = np.dot(u, np.dot(np.diag(s), vh.T))
            I1 = trace(S)

            s1 = s[0]
            s2 = s[1]

            [d1, d2, l, t] = GetEigenMatrices(u, vh)

            sqrt = np.sqrt
            delta2 = delta**2
            s = posJ
            II_F = I2

            a = 2*(J**4 + 2*J**3*s - 11*J**2*s**2 - 12*J*s**3 + 18*s**4) + 4*Jr**2*(I2**2 - 2*J**2)
            alpha = sqrt(I2**2*a - 8*I2*J**5 - 16*I2*J**4*s + 72*I2*J**3*s**2 - 4*I2*J**3*s1**4 - 4*I2*J**3*s2**4 + 64*I2*J**2*s**3 - 8*I2*J**2*s*s1**4 - 8*I2*J**2*s*s2**4 - 160*I2*J*s**4 + 4*I2*J*s**2*s1**4 + 4*I2*J*s**2*s2**4 + 48*I2*s**5 + 8*I2*s**3*s1**4 + 8*I2*s**3*s2**4 + 8*J**6 + 16*J**5*s - 56*J**4*s**2 + 4*J**4*s1**4 + 4*J**4*s2**4 - 32*J**3*s**3 + 8*J**3*s*s1**4 + 8*J**3*s*s2**4 + 144*J**2*s**4 - 12*J**2*s**2*s1**4 - 12*J**2*s**2*s2**4 - 96*J*s**5 - 16*J*s**3*s1**4 - 16*J*s**3*s2**4 + 16*s**6 + 16*s**4*s1**4 + 16*s**4*s2**4)

            beta =  (2*(I2*Jr - J**2 - J * s + 2 * s**2) * (s1**2 - s2**2) - alpha)/(2*((I2*(3*s**2 - J*s - J**2) + 2*(J**3 + J**2*s - 3*J*s**2 + s**3))))

            lamb1 = 1. / Jr + (2 * (I2 * Jr + 2*s**2) * (I2 - 2 * J) + alpha) / (4 * s**3 * Jr)
            lamb2 = 1. / Jr + (2 * (I2 * Jr + 2*s**2) * (I2 - 2 * J) - alpha) / (4 * s**3 * Jr)
            lamb3 = 2. / Jr + ( I2 - 2 * J)/(2 * s * Jr)
            lamb4 =           (-I2 + 2 * J)/(2 * s * Jr)

            # Project to SPD if needed
            if self.stabilise_tangents:
                eps = self.tangent_stabiliser_value
                lamb1 = max(lamb1, eps)
                lamb2 = max(lamb2, eps)
                lamb3 = max(lamb3, eps)
                lamb4 = max(lamb4, eps)

            # Scaling modes do not decouple for MIPS
            gamma  = np.sqrt(1. + beta**2) # normaliser

            # Coupled scaling modes
            e1 = 1. / gamma * (beta * d1 + d2)
            e2 = 1. / gamma * (d1 - beta * d2)

            d1 = e1
            d2 = e2

            H = lamb1 * np.outer(d1, d1) + lamb2 * np.outer(d2, d2) + lamb3 * np.outer(l, l) + lamb4 * np.outer(t, t)

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        s = np.sqrt(J**2 + self.delta**2)
        Jr = 0.5 * (J + s)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)
        d = self.ndim

        delta = self.delta
        I2 = trace(F.T.dot(F))
        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)
        stress = 2. / d / Jr**(2./d) * F - 2. * I2 / d**2 * Jr**(-2./d-1.)  * dJrdF

        # stress += 2. * (J * dJrdJ - Jr) / Jr**2 * dJdF(F)
        # stress +=  2. * J / Jr**2 * dJrdF - 2. / Jr * dJdF(F)
        stress +=  J / Jr**2 * dJrdF - 1. / Jr * dJdF(F)
        # stress /= 2.

        H = dJdF(F)
        stress = 1. / Jr * (F - H) + 1. / Jr / s * (J - I2 / 2.) * H

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        d = self.ndim
        delta = self.delta
        s = np.sqrt(J**2 + self.delta**2)
        Jr = 0.5 * (J + s)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)
        # IIFr = trace(Fr.T.dot(Fr))
        I2 = trace(F.T.dot(F))

        # energy  = (1./d * Jr**(-2./d) * IIFr)
        # energy  = (1./d * Jr**(-2./d) * I2) + 2. * (1 - J / Jr)
        # energy  = (1./d * Jr**(-2./d) * I2) - 2. * J / Jr + 2.
        energy  = (1./d * Jr**(-2./d) * I2) + 1 - J / Jr


        return energy






class MIPSF3(Material):
    """The MIPS energy

        W(F) = Fr:Fr/d/Jr^(2/d)

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPSF3, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # the smaller minJ the more chance to untangle something
        minJ = self.minJ
        # self.delta = np.sqrt(1e-12 + min(minJ, 0.)**2 * 0.04)
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04) * 2. # embed factot 4 in the definition of delta
        # self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2) # superbad it seems like
        # self.delta = .7
        # self.maxBeta = self.maxBeta

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

        # self.delta = np.sqrt(1e-8 + min(self.minJ, 0.)**2 * 0.04)
        d = self.ndim
        # delta = self.delta
        beta = self.maxBeta

        II_F = trace(F.T.dot(F))

        # Compute the rotation variant SVD of F
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        S = np.dot(u, np.dot(np.diag(s), vh.T))
        # I1 = trace(S)
        I1 = s.sum()


        Jr = J + I1 * beta + beta**2
        if np.isclose(Jr, 0) or Jr < 0.:
            Jr = max(1e-8, Jr)
        # if Jr < 0:
            # print(Jr, beta)

        # if self.ndim == 2 and False:
        if self.ndim == 2 and True:

            I2 = II_F
            I3 = J

            s1 = s[0]
            s2 = s[1]

            [d1, d2, l, t] = GetEigenMatrices(u, vh)

            sqrt = np.sqrt
            tau =  (2*beta*s1 - 2*beta*s2 + s1**2 - s2**2 + sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4))/(I1*beta + J + beta**2)

            lamb1 =  (I1**2*beta**2 + 2*I1*beta**3 + I1*beta*s1**2 + I1*beta*s2**2 - I1*beta*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4) + I2*beta**2 + I2*beta*s1 + I2*beta*s2 + I2*s1**2/2 + I2*s2**2/2 - I2*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4)/2 - J**2 - 2*J*beta**2 - 2*J*beta*s1 - 2*J*beta*s2 + beta**4 + beta**2*s1**2 + beta**2*s2**2 - beta**2*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4))/(I1**3*beta**3 + 3*I1**2*J*beta**2 + 3*I1**2*beta**4 + 3*I1*J**2*beta + 6*I1*J*beta**3 + 3*I1*beta**5 + J**3 + 3*J**2*beta**2 + 3*J*beta**4 + beta**6)
            lamb2 =  (I1**2*beta**2 + 2*I1*beta**3 + I1*beta*s1**2 + I1*beta*s2**2 + I1*beta*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4) + I2*beta**2 + I2*beta*s1 + I2*beta*s2 + I2*s1**2/2 + I2*s2**2/2 + I2*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4)/2 - J**2 - 2*J*beta**2 - 2*J*beta*s1 - 2*J*beta*s2 + beta**4 + beta**2*s1**2 + beta**2*s2**2 + beta**2*sqrt(I1**2*beta**2 + 2*I1*J*beta + 2*I1*beta**3 - J**2 - 6*J*beta**2 - 4*J*beta*s1 - 4*J*beta*s2 + beta**4 + 4*beta**2*s1**2 + 4*beta**2*s2**2 + 4*beta*s1**3 + 4*beta*s2**3 + s1**4 + s2**4))/(I1**3*beta**3 + 3*I1**2*J*beta**2 + 3*I1**2*beta**4 + 3*I1*J**2*beta + 6*I1*J*beta**3 + 3*I1*beta**5 + J**3 + 3*J**2*beta**2 + 3*J*beta**4 + beta**6)
            lamb3 =  (2*I1*beta + I2/2 + J + 2*beta**2)/(I1*beta + J + beta**2)**2
            lamb4 =  (I1 + 2*beta)*(-I2 + 2*J)/(2*I1*(I1*beta + J + beta**2)**2)

            # Project to SPD if needed
            if self.stabilise_tangents:
                eps = self.tangent_stabiliser_value
                lamb1 = max(lamb1, eps)
                lamb2 = max(lamb2, eps)
                lamb3 = max(lamb3, eps)
                lamb4 = max(lamb4, eps)

            # Scaling modes do not decouple for MIPS
            gamma  = np.sqrt(1. + tau**2) # normaliser

            # Coupled scaling modes
            e1 = 1. / gamma * (tau * d1 + d2)
            e2 = 1. / gamma * (d1 - tau * d2)

            d1 = e1
            d2 = e2

            H = lamb1 * np.outer(d1, d1) + lamb2 * np.outer(d2, d2) + lamb3 * np.outer(l, l) + lamb4 * np.outer(t, t)

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        d = self.ndim
        # delta = self.delta
        beta = self.maxBeta

        # Compute the rotation variant SVD of F
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        R = u.dot(vh.T)
        S = np.dot(u, np.dot(np.diag(s), vh.T))
        # I1 = trace(S)
        I1 = s.sum()

        Jr = J + I1 * beta + beta**2
        if np.isclose(Jr, 0) or Jr < 0.:
            Jr = max(1e-8, Jr)

        I2 = trace(F.T.dot(F))

        H = dJdF(F)
        stress = 1. / Jr * (F - H) + 1. / Jr**2 * (J - I2 / 2.) * (H + beta * R)
        # stress = 1. / Jr * (F - H) + 1. / Jr**2 * (J - I2 / 2.) * (H + 0. * R)

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        d = self.ndim
        delta = self.delta
        beta = self.maxBeta

        # Compute the rotation variant SVD of F
        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        S = np.dot(u, np.dot(np.diag(s), vh.T))
        # I1 = trace(S)
        I1 = s.sum()
        Jr = J + I1 * beta + beta**2

        if np.isclose(Jr, 0) or Jr < 0.:
            Jr = max(1e-8, Jr)
        # IIFr = trace(Fr.T.dot(Fr))
        I2 = trace(F.T.dot(F))

        energy  = (1./d * Jr**(-2./d) * I2) + 1 - J / Jr


        return energy












class SymmetricDirichlet(Material):
    """ Symmetric Dirichlet model

        W(F) = 1/2*(F:F) + 1/2*(F**(-1):F**(-1))

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(SymmetricDirichlet, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # the smaller minJ the more chance to untangle
        minJ = self.minJ
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = 0.
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

        delta = self.delta
        posJ = np.sqrt(J**2 + delta**2)
        Jr = 0.5 * (J + posJ)
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)

        gJ = vec(dJdF(F))
        HJ = d2JdFdF(F)
        f = vec(F)

        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)
        gJr = vec(dJrdF)
        HJr = 0.5 * (1 + J / np.sqrt(J**2 + delta**2)) * HJ + 0.5 * (delta**2 / (J**2 + delta**2)**(3./2.)) * np.outer(gJ,gJ)

        d2 = self.ndim**2
        if self.ndim == 2:
            H = (1 + 1 / Jr**2) * np.eye(d2,d2) - 2 / Jr**3 * (np.outer(f,gJr) + np.outer(gJr,f)) + 3 * trc / Jr**4 * np.outer(gJr,gJr) -\
                trc / Jr**3 * HJr
        else:

            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            R = u.dot(vh.T)
            S = np.dot(vh, np.dot(np.diag(s), vh.T))
            r = vec(R)
            I1 = trace(S)
            Is = (I1**2 - trc) / Jr
            dIsdF = 2/Jr * (I1 * R - F) - Is / Jr * dJrdF

            gR = dRdF(u, s, vh, False)

            d2IsdFdF = 2. / Jr * (np.outer(r,r) + I1 * gR - np.eye(d2,d2)) - 2. / Jr**2 * np.outer(I1 * r - f, gJr) -\
                1 / Jr * np.outer(gJr,vec(dIsdF)) + Is / Jr**2 * np.outer(gJr,gJr) - Is / Jr * HJr

            H  = np.eye(d2,d2) + 1 / 4. * np.outer(vec(dIsdF),vec(dIsdF)) + Is / 4. * d2IsdFdF - 1 / Jr * gR + 1 / Jr**2 * np.outer(r, gJr) +\
                1/Jr**2 * np.outer(gJr, r) - 2 * I1 / Jr**3 * np.outer(gJr, gJr) + I1 / Jr**2 * HJr

        if True:
        # if False:
            if self.ndim == 2:
                # Compute the rotation variant SVD of F
                u, s, vh = svd_rv(F, full_matrices=True)
                vh = vh.T
                # R = u.dot(vh.T)
                # S = np.dot(vh, np.dot(np.diag(s), vh.T))
                S = np.dot(u, np.dot(np.diag(s), vh.T))
                # I1 = trace(S)
                I1 = s.sum()

                [d1, d2, l, t] = GetEigenMatrices(u, vh)

                s1 = s[0];
                s2 = s[1];

                I2 = trc;
                # lamb1 =  3*I2*s2**2/J**4 + 1 - 3/J**2 ;
                # lamb2 =  3*I2*s1**2/J**4 + 1 - 3/J**2 ;
                # lamb3 =  (I2 + J**3 + J)/J**3 ;
                # lamb4 =  (-I2 + J**3 + J)/J**3 ;

                # lamb1 = 1 + 3. / s1**4
                # lamb2 = 1 + 3. / s2**4
                # lamb3 = 1 + 1. / J**2 + I2/J**3
                # lamb4 = 1 + 1. / J**2 - I2/J**3

                lamb1 =  (4*I2*s2**2*(J**2 - posJ**2 + 3*posJ*(J + posJ)) - 16*J*posJ**2*(J + posJ) + posJ**3*(J + posJ)**3 + 4*posJ**3*(J + posJ))/(posJ**3*(J + posJ)**3)
                lamb2 =  (4*I2*s1**2*(J**2 - posJ**2 + 3*posJ*(J + posJ)) - 16*J*posJ**2*(J + posJ) + posJ**3*(J + posJ)**3 + 4*posJ**3*(J + posJ))/(posJ**3*(J + posJ)**3)
                lamb3 =  4*I2/(posJ*(J + posJ)**2) + 1 + 4/(J + posJ)**2
                lamb4 =  -4*I2/(posJ*(J + posJ)**2) + 1 + 4/(J + posJ)**2

                # Project to SPD if needed
                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)

                H = lamb1 * np.outer(d1, d1) + lamb2 * np.outer(d2, d2) + lamb3 * np.outer(l, l) + lamb4 * np.outer(t, t)

            else:

                I2 = trc
                I3 = J

                # Compute the rotation variant SVD of F
                u, s, vh = svd(F, full_matrices=True)
                vh = vh.T
                S = np.dot(u, np.dot(np.diag(s), vh.T))
                I1 = trace(S)

                s1 = s[0]
                s2 = s[1]
                s3 = s[2]

                if np.abs(s1*s2*s3-J) > 1e-6:
                    print(s1*s2*s3-J)
                # print(s1*s2*s3-J)

                [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(u, vh)

                e1 = d1
                e2 = d2
                e3 = d3

                # Original SD exactly as they appear in BSmith et. al. 2019
                # [this gives the exact same hessian H=H1 if used with Ts/Ls from main paper]
                lamb1 =  1. + 3./s1**4
                lamb2 =  1. + 3./s2**4
                lamb3 =  1. + 3./s3**4
                lamb4 =  1. + s1**2/J**2 - (I2 - s1**2)*s1**3/J**3
                lamb5 =  1. + s2**2/J**2 - (I2 - s2**2)*s2**3/J**3
                lamb6 =  1. + s3**2/J**2 - (I2 - s3**2)*s3**3/J**3
                lamb7 =  1. + s1**2/J**2 + (I2 - s1**2)*s1**3/J**3
                lamb8 =  1. + s2**2/J**2 + (I2 - s2**2)*s2**3/J**3
                lamb9 =  1. + s3**2/J**2 + (I2 - s3**2)*s3**3/J**3

                # Project to SPD if needed
                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)
                    lamb5 = max(lamb5, eps)
                    lamb6 = max(lamb6, eps)
                    lamb7 = max(lamb7, eps)
                    lamb8 = max(lamb8, eps)
                    lamb9 = max(lamb9, eps)

                H1 = lamb1 * outer(e1,e1) + lamb2 * outer(e2,e2) + lamb3 * outer(e3,e3) +\
                    lamb4 * outer(t1,t1) + lamb5 * outer(t2,t2) + lamb6 * outer(t3,t3) +\
                    lamb7 * outer(l1,l1) + lamb8 * outer(l2,l2) + lamb9 * outer(l3,l3)

                H = H1

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        delta = self.delta
        posJ = np.sqrt(J**2 + delta**2)
        Jr = 0.5 * (J + posJ)
        if np.isclose(Jr, 0):
            print("Small Jr", J, Jr)
            Jr = max(1e-8, Jr)

        dJrdF = 0.5 * (1. + J / np.sqrt(J**2 + delta**2)) * dJdF(F)

        if self.ndim == 2:
            stress = (1. + 1 / Jr**2) * F - trc / Jr**3 * dJrdF
        else:
            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            R = u.dot(vh.T)
            S = np.dot(vh, np.dot(np.diag(s), vh.T))
            I1 = trace(S)
            Is = (I1**2 - trc) / Jr
            dIsdF = 2 / Jr * (I1 * R - F) - Is / Jr * dJrdF

            stress = F + Is / 4. * dIsdF - 1/Jr * R + I1/Jr**2 * dJrdF

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        trc = trace(F.T.dot(F))

        delta = self.delta
        Jr = 0.5 * (J + np.sqrt(J**2 + delta**2))
        if np.isclose(Jr, 0):
            Jr = max(1e-8, Jr)

        if self.ndim == 2:
            energy  = 0.5 * (1. + 1 / Jr**2) * trc
        elif self.ndim == 3:
            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            R = u.dot(vh.T)
            S = np.dot(vh, np.dot(np.diag(s), vh.T))
            I1 = trace(S)
            Is = (I1**2 - trc) / Jr

            # energy  = 0.5 * trc + 1. / 8. * Is**2  / Jr - I1 / Jr
            energy  = 0.5 * trc + 1. / 8. * Is**2 - I1 / Jr

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
        d2 = d * d

        I = StrainTensors['I']
        F = StrainTensors['F'][gcounter]

        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T

        gR = dRdF(u, s, vh, True)
        # gR = dRdF(u, s, vh, False)

        H = np.eye(d2,d2) - gR
        H *= 2.

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

            gR = dRdF(u, s, vh, False)

            IS  = trace(S)
            IIS = trace(b)
            IIStarS = 0.5 * (IS*IS - IIS)

            newH = (1 + IS / J) * gR + (1 / J) * np.outer(r,r)
            newH = newH - (IS / J2) * (np.outer(g,r) + np.outer(r,g))
            newH = newH + (2.0 * IIStarS / J3) * np.outer(g,g)
            newH = newH - (IIStarS / J2) * HJ
            newH = newH + (1 / J2) * (np.outer(g,f) + np.outer(f,g))
            newH = newH - (1/J) * np.eye(9,9);
            H = H - 2.0 * newH
            H = H / 2.
            makezero(H)

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


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        u, s, vh = svd(F, full_matrices=True)
        vh = vh.T
        R = u.dot(vh.T)
        invF = np.linalg.inv(F)
        invR = np.linalg.inv(R)
        # not sure about 1/2, check
        energy  = einsum("ij,ij",F - R,F - R) + einsum("ij,ij",invF - invR, invF - invR)

        return energy





class OgdenNeoHookeanF(Material):
    """ Ogden neoHookean model

        W(F) = mu / 2 * (II_F - 3) - mu * log(J) + lambda / 2 * (J - 1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(OgdenNeoHookeanF, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        # the smaller minJ the more chance to untangle
        minJ = self.minJ
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = 0.
        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        I2 = trace(F.T.dot(F))

        if self.ndim == 3:
            I3 = J
            m_mu = self.mu
            m_lambda = self.lamb

            # Compute the rotation variant SVD of F
            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            S = np.dot(u, np.dot(np.diag(s), vh.T))
            I1 = trace(S)

            s1 = s[0]
            s2 = s[1]
            s3 = s[2]

            a11 =  J**2*(m_lambda + m_mu/J**2)/s1**2 + m_mu
            a12 =  J*s3*(m_lambda + m_mu/J**2) + s3*(m_lambda*(2*J - 2)/2 - m_mu/J)
            a13 =  J*s2*(m_lambda + m_mu/J**2) + s2*(m_lambda*(2*J - 2)/2 - m_mu/J)
            a22 =  J**2*(m_lambda + m_mu/J**2)/s2**2 + m_mu
            a23 =  J*s1*(m_lambda + m_mu/J**2) + s1*(m_lambda*(2*J - 2)/2 - m_mu/J)
            a33 =  J**2*(m_lambda + m_mu/J**2)/s3**2 + m_mu

            Hw = np.array([
                [a11,a12,a13],
                [a12,a22,a23],
                [a13,a23,a33],
                ])

            eigs, vecs = sp.linalg.eigh(Hw)
            vec1 = vecs[:,0]
            vec2 = vecs[:,1]
            vec3 = vecs[:,2]

            lamb1 = eigs[0]
            lamb2 = eigs[1]
            lamb3 = eigs[2]
            lamb4 =  m_mu + s1*(m_lambda*(2*J - 2)/2. - m_mu/J)
            lamb5 =  m_mu + s2*(m_lambda*(2*J - 2)/2. - m_mu/J)
            lamb6 =  m_mu + s3*(m_lambda*(2*J - 2)/2. - m_mu/J)
            lamb7 =  m_mu - s1*(m_lambda*(2*J - 2)/2. - m_mu/J)
            lamb8 =  m_mu - s2*(m_lambda*(2*J - 2)/2. - m_mu/J)
            lamb9 =  m_mu - s3*(m_lambda*(2*J - 2)/2. - m_mu/J)

            # Project to SPD if needed
            if self.stabilise_tangents:
                eps = self.tangent_stabiliser_value
                lamb1 = max(lamb1, eps)
                lamb2 = max(lamb2, eps)
                lamb3 = max(lamb3, eps)
                lamb4 = max(lamb4, eps)
                lamb5 = max(lamb5, eps)
                lamb6 = max(lamb6, eps)
                lamb7 = max(lamb7, eps)
                lamb8 = max(lamb8, eps)
                lamb9 = max(lamb9, eps)

            [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(u, vh)
            e1 = vec1[0] * d1 + vec1[1] * d2 + vec1[2] * d3
            e2 = vec2[0] * d1 + vec2[1] * d2 + vec2[2] * d3
            e3 = vec3[0] * d1 + vec3[1] * d2 + vec3[2] * d3

            H = lamb1 * outer(e1,e1) + lamb2 * outer(e2,e2) + lamb3 * outer(e3,e3) +\
                lamb4 * outer(t1,t1) + lamb5 * outer(t2,t2) + lamb6 * outer(t3,t3) +\
                lamb7 * outer(l1,l1) + lamb8 * outer(l2,l2) + lamb9 * outer(l3,l3)

        elif self.ndim == 2:
            sqrt = np.sqrt

            # Compute the rotation variant SVD of F
            u, s, vh = svd(F, full_matrices=True)
            vh = vh.T
            S = np.dot(u, np.dot(np.diag(s), vh.T))
            I1 = trace(S)

            s1 = s[0]
            s2 = s[1]

            [d1, d2, l, t] = GetEigenMatrices(u, vh)

            tau =  -(J**2*lamb*s1**2 - J**2*lamb*s2**2 + mu*s1**2 - mu*s2**2 + sqrt(14*J**6*lamb**2 - 16*J**5*lamb**2 + J**4*lamb**2*s1**4 + J**4*lamb**2*s2**4 + 4*J**4*lamb**2 - 4*J**4*lamb*mu + 2*J**2*lamb*mu*s1**4 + 2*J**2*lamb*mu*s2**4 - 2*J**2*mu**2 + mu**2*s1**4 + mu**2*s2**4))/(2*J**2*lamb*(2*J - 1))

            lamb1 =  (J**2*lamb*s1**2 + J**2*lamb*s2**2 + 2*J**2*mu + mu*s1**2 + mu*s2**2 - sqrt(14*J**6*lamb**2 - 16*J**5*lamb**2 + J**4*lamb**2*s1**4 + J**4*lamb**2*s2**4 + 4*J**4*lamb**2 - 4*J**4*lamb*mu + 2*J**2*lamb*mu*s1**4 + 2*J**2*lamb*mu*s2**4 - 2*J**2*mu**2 + mu**2*s1**4 + mu**2*s2**4))/(2*J**2)
            lamb2 =  (J**2*lamb*s1**2 + J**2*lamb*s2**2 + 2*J**2*mu + mu*s1**2 + mu*s2**2 + sqrt(14*J**6*lamb**2 - 16*J**5*lamb**2 + J**4*lamb**2*s1**4 + J**4*lamb**2*s2**4 + 4*J**4*lamb**2 - 4*J**4*lamb*mu + 2*J**2*lamb*mu*s1**4 + 2*J**2*lamb*mu*s2**4 - 2*J**2*mu**2 + mu**2*s1**4 + mu**2*s2**4))/(2*J**2)
            lamb3 =  -J*lamb + lamb + mu + mu/J
            lamb4 =  J*lamb - lamb + mu - mu/J

            # Project to SPD if needed
            if self.stabilise_tangents:
                eps = self.tangent_stabiliser_value
                lamb1 = max(lamb1, eps)
                lamb2 = max(lamb2, eps)
                lamb3 = max(lamb3, eps)
                lamb4 = max(lamb4, eps)

            # Scaling modes do not decouple
            gamma  = np.sqrt(1. + tau**2) # normaliser

            # Coupled scaling modes
            e1 = 1. / gamma * (tau * d1 + d2)
            e2 = 1. / gamma * (d1 - tau * d2)

            d1 = e1
            d2 = e2

            H = lamb1 * np.outer(d1, d1) + lamb2 * np.outer(d2, d2) + lamb3 * np.outer(l, l) + lamb4 * np.outer(t, t)

        self.H_VoigtSize = H.shape[0]

        return H


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        I2 = trace(F.T.dot(F))
        gJ = dJdF(F)

        stress = mu * F - mu / J * gJ + lamb * (J - 1) * gJ

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        I2 = trace(F.T.dot(F))
        energy  = 0.5 * mu * I2 - mu * np.log(J) + lamb / 2. * (J-1)**2

        return energy




class MIPS_F(Material):
    """
        W(F) = mu * II_F / d / J**(2/d) + lambda * (J + J**(-1) - 2)
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPS_F, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        minJ = self.minJ
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = 0.
        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        ndim = self.ndim
        d = self.ndim
        d2 = d * d

        trb = trace(F.T.dot(F))

        tmp1 = J if ndim == 2 else (J**2.)**(1./3.)

        if self.formulation_style == "classic":
            H = dJdF(F)
            f = vec(F)
            h = vec(H)

            tmp2 = 1. / (tmp1 * J)
            tmp3 = tmp2 / J

            hessian = mu * (2. / d / tmp1 * np.eye(ndim*ndim) - 4. / d2 * tmp2 * (np.outer(h,f) + np.outer(f, h)) +\
                2. * trb / d2 * (2. / d + 1.) * tmp3 * np.outer(h, h)) + lamb * (2. / J / J / J) * np.outer(h, h)

            # Ixf = d2JdFdF(F)
            # initial_stiffness = (lamb * (1. - 1. / J / J) - mu * (2. * trb / d2 * tmp2)) * Ixf

            sigmaH = np.zeros((ndim, ndim))
            sigmaJ = lamb * (1. - 1. / J / J) - mu * (2. * trb / d2 * tmp2)

            # Initail stiffness component
            initial_stiffness = GetInitialStiffnessPolyconvex(sigmaH, sigmaJ, F,
                stabilise=self.stabilise_tangents,
                eps=self.tangent_stabiliser_value
                )
            hessian += initial_stiffness

        elif self.formulation_style == "ps":
            I2 = trb
            if ndim == 2:
                # Get SVD of F
                [U, S, Vh] = svd(F, full_matrices=True); V = Vh.T
                s1 = S[0]
                s2 = S[1]

                [d1, d2, l, t] = GetEigenMatrices(U, V)

                a11 =  (I2*mu/s1**2 + 2*lamb/s1**2 - mu)/J
                a22 =  (I2*mu/s2**2 + 2*lamb/s2**2 - mu)/J
                a12 =  I2*mu/(2*J**2) + lamb*(1 + J**(-2)) - mu/s2**2 - mu/s1**2

                Hw = np.array([[a11,a12],[a12, a22]])

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 =  -lamb + lamb/(s1**2*s2**2) + mu/(2*s2**2) + mu/(s1*s2) + mu/(2*s1**2)
                lamb4 =   lamb - lamb/(s1**2*s2**2) - mu/(2*s2**2) + mu/(s1*s2) - mu/(2*s1**2)

                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)

                # Build Hessian
                ds = np.array([d1,d2]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
                hessian = ds.dot(HwSPD.dot(ds.T)) + lamb3 * np.outer(l,l) + lamb4 * np.outer(t,t)

            elif ndim == 3:

                m_mu = self.mu
                m_lambda = self.lamb
                I2 = trb

                # Get SVD of F
                [U, S, Vh] = svd(F, full_matrices=True); V = Vh.T
                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                a11 =  10*I2*m_mu/(27*tmp1*s1**2) + 2*m_lambda/(J*s1**2) - 2*m_mu/(9*tmp1)
                a22 =  10*I2*m_mu/(27*tmp1*s2**2) + 2*m_lambda/(J*s2**2) - 2*m_mu/(9*tmp1)
                a33 =  10*I2*m_mu/(27*tmp1*s3**2) + 2*m_lambda/(J*s3**2) - 2*m_mu/(9*tmp1)
                a12 =  4*I2*m_mu/(27*tmp1*s1*s2) + m_lambda*(s3 + 1/(J*s1*s2)) - 4*m_mu*s1/(9*tmp1*s2) - 4*m_mu*s2/(9*tmp1*s1)
                a13 =  4*I2*m_mu/(27*tmp1*s1*s3) + m_lambda*(s2 + 1/(J*s1*s3)) - 4*m_mu*s1/(9*tmp1*s3) - 4*m_mu*s3/(9*tmp1*s1)
                a23 =  4*I2*m_mu/(27*tmp1*s2*s3) + m_lambda*(s1 + 1/(J*s2*s3)) - 4*m_mu*s2/(9*tmp1*s3) - 4*m_mu*s3/(9*tmp1*s2)

                Hw = np.array([
                    [a11,a12,a13],
                    [a12,a22,a23],
                    [a13,a23,a33],
                    ])

                eigs, vecs = sp.linalg.eigh(Hw)
                vec1 = vecs[:,0]
                vec2 = vecs[:,1]
                vec3 = vecs[:,2]

                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 = eigs[2]
                lamb4 =  2*I2*m_mu/(9*tmp1*s2*s3) - m_lambda*s1 + m_lambda/(J*s2*s3) + 2*m_mu/(3*tmp1)
                lamb5 =  2*I2*m_mu/(9*tmp1*s1*s3) - m_lambda*s2 + m_lambda/(J*s1*s3) + 2*m_mu/(3*tmp1)
                lamb6 =  2*I2*m_mu/(9*tmp1*s1*s2) - m_lambda*s3 + m_lambda/(J*s1*s2) + 2*m_mu/(3*tmp1)
                lamb7 =  -2*I2*m_mu/(9*tmp1*s2*s3) + m_lambda*s1 - m_lambda/(J*s2*s3) + 2*m_mu/(3*tmp1)
                lamb8 =  -2*I2*m_mu/(9*tmp1*s1*s3) + m_lambda*s2 - m_lambda/(J*s1*s3) + 2*m_mu/(3*tmp1)
                lamb9 =  -2*I2*m_mu/(9*tmp1*s1*s2) + m_lambda*s3 - m_lambda/(J*s1*s2) + 2*m_mu/(3*tmp1)

                # Project to SPD if needed
                if self.stabilise_tangents:
                    eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, eps)
                    lamb2 = max(lamb2, eps)
                    lamb3 = max(lamb3, eps)
                    lamb4 = max(lamb4, eps)
                    lamb5 = max(lamb5, eps)
                    lamb6 = max(lamb6, eps)
                    lamb7 = max(lamb7, eps)
                    lamb8 = max(lamb8, eps)
                    lamb9 = max(lamb9, eps)

                [d1, d2, d3, l1, l2, l3, t1, t2, t3] = GetEigenMatrices(U, V)

                # Build Hessian
                ds = np.array([d1,d2,d3]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                    lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])
                hessian = ds.dot(HwSPD.dot(ds.T)) +\
                    lamb4 * outer(l1,l1) + lamb5 * outer(l2,l2) + lamb6 * outer(l3,l3) +\
                    lamb7 * outer(t1,t1) + lamb8 * outer(t2,t2) + lamb9 * outer(t3,t3)

        self.H_VoigtSize = hessian.shape[0]

        return hessian


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        ndim = self.ndim

        d = self.ndim
        d2 = d * d

        # tmp1 = J**(2. / d)
        tmp1 = J if ndim == 2 else (J**2.)**(1./3.)
        tmp2 = 1. / (tmp1 * J)

        trb = trace(F.T.dot(F))
        H = dJdF(F)

        sigmaF = 2. * mu / d / tmp1 * F
        sigmaJ = lamb * (1. - 1. / J / J) - 2. * mu * trb / d2 * tmp2
        P = sigmaF + sigmaJ * H

        return P


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        trb = trace(F.T.dot(F))

        energy = mu * trb / d / J**(2/d) + lamb * (J + 1./J - 2)

        return energy






class MooneyRivlinF(Material):
    """ Polyconvex Mooney Rivlin model: this implementation uses polyconvex formulation of Bonet et. al. 2014

        W(F) = mu1 * (II_F - N) + mu2 * (II_H - N) - (2 * mu1 + 4 * mu2) * log(J) + lambda / 2 * (J - 1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MooneyRivlinF, self).__init__(mtype, ndim, **kwargs)

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 9
        elif self.ndim==2:
            self.H_VoigtSize = 4

        minJ = self.minJ
        self.delta = np.sqrt(1e-8 + min(minJ, 0.)**2 * 0.04)
        # self.delta = 0.
        # LOW LEVEL DISPATCHER
        # self.has_low_level_dispatcher = True
        self.has_low_level_dispatcher = False

    def KineticMeasures(self,F,ElectricFieldx=0, elem=0):
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        ndim = self.ndim
        I = np.eye(ndim*ndim,ndim*ndim)

        H = dJdF(F)
        hessianJ = d2JdFdF(F)
        h = vec(H)

        sigmaH = 2. * mu2 * H
        sigmaJ = -(2. * mu1 + 4. * mu2) / J + lamb * (J - 1.)

        WJJ = (2. * mu1 + 4. * mu2) / J**2 + lamb

        fxIxf = Get_FxIxF(F)

        # Constitutive
        hessian = 2. * mu1 * I + 2. * mu2 * fxIxf + WJJ * np.outer(h, h)
        # Initail stiffness component
        # initial_stiffness = d2JdFdF(sigmaH + sigmaJ * F)
        initial_stiffness = GetInitialStiffnessPolyconvex(sigmaH, sigmaJ, F,
            stabilise=self.stabilise_tangents,
            # stabilise=False,
            eps=self.tangent_stabiliser_value
            )
        hessian += initial_stiffness


        self.H_VoigtSize = hessian.shape[0]

        return hessian


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        H = dJdF(F)

        sigmaF = 2. * mu1 * F
        sigmaH = 2. * mu2 * H
        sigmaJ = -(2. * mu1 + 4. * mu2) / J + lamb * (J - 1)

        P = sigmaF + cross(sigmaH, F) + sigmaJ * H

        return P


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu1 = self.mu1
        mu2 = self.mu2
        lamb = self.lamb

        II_F = trace(F.T.dot(F))
        H = dJdF(F)
        II_H = trace(H.T.dot(H))

        energy = mu1 * (II_F - N) + mu2 * (II_H - N) - (2. * mu1 + 4. * mu2) * np.log(J) + lamb / 2. * (J - 1)**2

        return energy



















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

        if fem_solver.use_ideal_element:
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

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = material.Hessian(StrainTensors,None,elem,counter)

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                CauchyStressTensor, H_Voigt, requires_geometry_update=fem_solver.requires_geometry_update)

            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        makezero(stiffness, 1e-12)
        # print(stiffness)
        # exit()
        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, CauchyStressTensor, H_Voigt,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveBF(B,SpatialGradient,self.ndim,self.nvar)

        BDB = B.dot(H_Voigt.dot(B.T))

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = vec(CauchyStressTensor)
            t = np.dot(B,TotalTraction)[:,None]

        return BDB, t


    def GetLocalTraction(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get traction vector of the system"""

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

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            t = self.TractionIntegrand(B, SpatialGradient[counter,:,:],
                CauchyStressTensor, requires_geometry_update=fem_solver.requires_geometry_update)

            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

        return tractionforce


    def TractionIntegrand(self, B, SpatialGradient, CauchyStressTensor,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveBF(B,SpatialGradient,self.ndim,self.nvar)

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = vec(CauchyStressTensor)
            t = np.dot(B,TotalTraction)[:,None]

        return t


    def GetEnergy(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get virtual energy of the system. For dynamic analysis this is handy for computing conservation of energy.
            The routine computes the global form of virtual internal energy i.e. integral of "W(C,G,C)"". This can be
            computed purely in a Lagrangian configuration.
        """

        if fem_solver.use_ideal_element:
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
