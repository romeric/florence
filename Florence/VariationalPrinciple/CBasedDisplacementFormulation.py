# -*- coding: utf-8 -*-
#
# EVERYTHING HERE IS EXPERMENTAL AND NOT FULLY PART OF FLORENCE - JUST ACADEMIC PROTOTYPES
#
import numpy as np
from numpy import einsum
import scipy as sp
from Florence.VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace
from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from .DisplacementApproachIndices import *
from Florence.MaterialLibrary.MaterialBase import Material
from Florence.Tensor import trace, Voigt, makezero, issymetric, ssvd
norm = np.linalg.norm
outer = np.outer
# np.set_printoptions(precision=16)

def vec(H):
    ndim = H.shape[0]
    if H.ndim == 4:
        x = H.flatten().reshape(ndim**2,ndim**2)
        return x
    else:
        return H.T.flatten()


def GetVoigtHessian(H):
    edim = H.shape[0]
    if H.ndim == 4:
        if edim == 2:
            H_Voigt = np.array([
                [H[0,0,0,0], H[0,0,1,1],H[0,0,0,1]],
                [H[0,0,1,1], H[1,1,1,1],H[1,1,0,1]],
                [H[0,0,0,1], H[1,1,0,1],H[0,1,0,1]]
                ])
        elif edim == 3:
            H_Voigt = np.array([
                [H[0,0,0,0], H[0,0,1,1],H[0,0,2,2],H[0,0,0,1],H[0,0,0,2],H[0,0,1,2]],
                [H[1,1,0,0], H[1,1,1,1],H[1,1,2,2],H[1,1,0,1],H[1,1,0,2],H[1,1,1,2]],
                [H[2,2,0,0], H[2,2,1,1],H[2,2,2,2],H[2,2,0,1],H[2,2,0,2],H[2,2,1,2]],
                [H[0,1,0,0], H[0,1,1,1],H[0,1,2,2],H[0,1,0,1],H[0,1,0,2],H[0,1,1,2]],
                [H[0,2,0,0], H[0,2,1,1],H[0,2,2,2],H[0,2,0,1],H[0,2,0,2],H[0,2,1,2]],
                [H[1,2,0,0], H[1,2,1,1],H[1,2,2,2],H[1,2,0,1],H[1,2,0,2],H[1,2,1,2]]
                ])
    elif H.ndim == 2:
        if edim == 4:
            H_Voigt = np.array([
                [H[0,0], H[0,3],H[0,1]],
                [H[3,0], H[3,3],H[1,3]],
                [H[0,1], H[1,3],H[1,1]]
                ])
        if edim == 9:
            H_Voigt = np.array([
                [H[0,0], H[0,4],H[0,8],H[0,1],H[0,2],H[0,5]],
                [H[4,0], H[4,4],H[4,8],H[1,4],H[4,6],H[4,5]],
                [H[8,0], H[8,4],H[8,8],H[8,1],H[2,8],H[7,8]],
                [H[0,1], H[1,4],H[8,1],H[1,1],H[1,2],H[1,5]],
                [H[0,2], H[4,6],H[2,8],H[1,2],H[2,2],H[2,7]],
                [H[0,5], H[4,5],H[7,8],H[1,5],H[2,7],H[7,7]]
                ])
    makezero(H_Voigt, 1e-10)
    return H_Voigt



def GetEigenMatrices(U, V):
    """ Eigenmatrices for C-based formulations
    """

    ndim = U.shape[0]

    if ndim == 3:
        D1 = np.array([[1.,0,0],[0,0,0],[0,0,0]])
        D1 = np.dot(V, np.dot(D1, V.T))
        d1 = vec(D1)
        D2 = np.array([[0.,0,0],[0,1.,0],[0.,0,0]])
        D2 = np.dot(V, np.dot(D2, V.T))
        d2 = vec(D2)
        D3 = np.array([[0.,0,0],[0,0.,0],[0,0,1.]])
        D3 = np.dot(V, np.dot(D3, V.T))
        d3 = vec(D3)

        L1 = np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        L1 = np.dot(V, np.dot(L1, V.T))
        l1 = vec(L1)
        L2 = np.array([[0.,0.,1.],[0.,0., 0],[1.,0.,0.]])
        L2 = np.dot(V, np.dot(L2, V.T))
        l2 = vec(L2)
        L3 = np.array([[0.,1.,0.],[1.,0.,0.],[0,0.,0.]])
        L3 = np.dot(V, np.dot(L3, V.T))
        l3 = vec(L3)

        return d1, d2, d3, l1, l2, l3

    elif ndim == 2:
        # Scale modes
        D1 = np.array([[1.,0],[0,0.]])
        D1 = np.dot(V, np.dot(D1, V.T))
        d1 = vec(D1)

        D2 = np.array([[0.,0],[0,1.]])
        D2 = np.dot(V, np.dot(D2, V.T))
        d2 = vec(D2)

        # Flip modes
        L = np.array([[0.,1],[1,0.]])
        L = np.dot(V, np.dot(L, V.T))
        l = vec(L)

        return d1, d2, l



def GetEigenMatrices_UL(U, V):
    """ Eigenmatrices for C-based formulations
    """

    ndim = U.shape[0]

    if ndim == 3:
        D1 = np.array([[1.,0,0],[0,0,0],[0,0,0]])
        D1 = np.dot(U, np.dot(D1, U.T))
        d1 = vec(D1)
        D2 = np.array([[0.,0,0],[0,1.,0],[0.,0,0]])
        D2 = np.dot(U, np.dot(D2, U.T))
        d2 = vec(D2)
        D3 = np.array([[0.,0,0],[0,0.,0],[0,0,1.]])
        D3 = np.dot(U, np.dot(D3, U.T))
        d3 = vec(D3)

        L1 = np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,0.]])
        L1 = np.dot(U, np.dot(L1, U.T))
        l1 = vec(L1)
        L2 = np.array([[0.,0.,1.],[0.,0., 0],[1.,0.,0.]])
        L2 = np.dot(U, np.dot(L2, U.T))
        l2 = vec(L2)
        L3 = np.array([[0.,1.,0.],[1.,0.,0.],[0,0.,0.]])
        L3 = np.dot(U, np.dot(L3, U.T))
        l3 = vec(L3)

        return d1, d2, d3, l1, l2, l3

    elif ndim == 2:
        # Scale modes
        D1 = np.array([[1.,0],[0,0.]])
        D1 = np.dot(U, np.dot(D1, U.T))
        d1 = vec(D1)

        D2 = np.array([[0.,0],[0,1.]])
        D2 = np.dot(U, np.dot(D2, U.T))
        d2 = vec(D2)

        # Flip modes
        L = np.array([[0.,1],[1,0.]])
        L = np.dot(U, np.dot(L, U.T))
        l = vec(L)

        return d1, d2, l




def FillConstitutiveBC(B,SpatialGradient,F,ndim,nvar):
    # See:
    # Introduction to Nonlinear Finite Element Analysis by Nam-Ho Kim, page 202 for the structure of [B_N]
    # Nonlinear Finite Element Methods by Peter Wriggers, page 124 for the structure of [B_L]

    # The derivation is as follows:
    # Consider the updated Lagrangian form (sigma : e) where e is symmetrised gradient
    # \int_v (sigma : \nabla u) dv = \int_V ( J^{-1} FSF^T : \nabla_0 u F^{-1}) J dV
    #                                = \int_V ( S : \nabla_0 F^T) dV
    # where  \nabla u =  \nabla_0 u F^{-1}; Bonet's book eqn 8.14 and
    # sigma = J^{-1} FSF^T and
    # dv = J dV

    if ndim == 2:
        B[0::ndim, 0] = F[0,0] * SpatialGradient[0,:]
        B[1::ndim, 0] = F[1,0] * SpatialGradient[0,:]

        B[0::ndim, 1] = F[0,1] * SpatialGradient[1,:]
        B[1::ndim, 1] = F[1,1] * SpatialGradient[1,:]

        B[0::ndim, 2] = F[0,0] * SpatialGradient[1,:] + F[0,1] * SpatialGradient[0,:]
        B[1::ndim, 2] = F[1,0] * SpatialGradient[1,:] + F[1,1] * SpatialGradient[0,:]
    else:
        B[0::ndim, 0] = F[0,0] * SpatialGradient[0,:]
        B[1::ndim, 0] = F[1,0] * SpatialGradient[0,:]
        B[2::ndim, 0] = F[2,0] * SpatialGradient[0,:]

        B[0::ndim, 1] = F[0,1] * SpatialGradient[1,:]
        B[1::ndim, 1] = F[1,1] * SpatialGradient[1,:]
        B[2::ndim, 1] = F[2,1] * SpatialGradient[1,:]

        B[0::ndim, 2] = F[0,2] * SpatialGradient[2,:]
        B[1::ndim, 2] = F[1,2] * SpatialGradient[2,:]
        B[2::ndim, 2] = F[2,2] * SpatialGradient[2,:]

        B[0::ndim, 3] = F[0,0] * SpatialGradient[1,:] + F[0,1] * SpatialGradient[0,:]
        B[1::ndim, 3] = F[1,0] * SpatialGradient[1,:] + F[1,1] * SpatialGradient[0,:]
        B[2::ndim, 3] = F[2,0] * SpatialGradient[1,:] + F[2,1] * SpatialGradient[0,:]

        B[0::ndim, 4] = F[0,0] * SpatialGradient[2,:] + F[0,2] * SpatialGradient[0,:]
        B[1::ndim, 4] = F[1,0] * SpatialGradient[2,:] + F[1,2] * SpatialGradient[0,:]
        B[2::ndim, 4] = F[2,0] * SpatialGradient[2,:] + F[2,2] * SpatialGradient[0,:]

        B[0::ndim, 5] = F[0,1] * SpatialGradient[2,:] + F[0,2] * SpatialGradient[1,:]
        B[1::ndim, 5] = F[1,1] * SpatialGradient[2,:] + F[1,2] * SpatialGradient[1,:]
        B[2::ndim, 5] = F[2,1] * SpatialGradient[2,:] + F[2,2] * SpatialGradient[1,:]






class StVenantKirchhoffC(Material):
    """The fundamental StVenantKirchhoff internal energy

        W(C) = mu/4*(C-I):(C-I) + lamb/4 tr(C-I)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(StVenantKirchhoffC, self).__init__(mtype, ndim, **kwargs)

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
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb
        # This is SPD
        C_Voigt = lamb * np.einsum("ij,kl",I,I) + mu * (np.einsum("ik,jl",I,I) + np.einsum("il,jk",I,I))
        C_Voigt = Voigt(C_Voigt,1)

        self.H_VoigtSize = C_Voigt.shape[0]

        return C_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb
        E = 0.5 * (C - I)
        stress = lamb * trace(E) * I + 2. * mu * E

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        C = np.dot(F.T,F)

        mu = self.mu
        lamb = self.lamb

        energy  = mu/4*np.einsum("ij,ij",C-I,C-I) + lamb/8 * trace(C-I)**2

        return energy







class OgdenNeoHookeanC(Material):
    """The fundamental Neo-Hookean internal energy, described in Ogden et. al.

        W(C) = mu/2*(C:I-3)- mu*lnJ + lamb/2*(J-1)**2

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(OgdenNeoHookeanC, self).__init__(mtype, ndim, **kwargs)

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
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        C = F.T.dot(F)
        invC = np.linalg.inv(C)
        c = J*J

        if self.formulation_style != "C" and self.formulation_style != "ps":
            raise ValueError("Material does not support this invariant formulation")

        # H  = 2. * (mu - lamb * (c - J)) * ( np.einsum("ik,jl", invC, invC) + np.einsum("il,jk", invC, invC) ) * 0.5
        # H += lamb * (2. * c - J) * np.einsum("ij,kl", invC, invC)
        # H_Voigt = GetVoigtHessian(H)
        # # makezero(H_Voigt)

        if self.formulation_style == "C":
            H  = 2. * (mu - lamb * (c - J)) * ( np.einsum("ik,jl", invC, invC) + np.einsum("il,jk", invC, invC) ) * 0.5
            H += lamb * (2. * c - J) * np.einsum("ij,kl", invC, invC)
            H_Voigt = GetVoigtHessian(H)


        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if self.ndim == 2:
                s1 = S[0]
                s2 = S[1]

                [d1, d2, l] = GetEigenMatrices(U, V)

                a11 =  (lamb*s2**2 + mu + mu/s1**2)/s1**2 + (-lamb*s2*(J - 1) - mu*s1 + mu/s1)/s1**3
                a22 =  (lamb*s1**2 + mu + mu/s2**2)/s2**2 + (-lamb*s1*(J - 1) - mu*s2 + mu/s2)/s2**3
                a12 =  (J*lamb + lamb*(J - 1))/J

                Hw = np.array([
                    [a11,a12],
                    [a12,a22],
                    ])

                # Stretch invariants way
                # eigs, vecs = sp.linalg.eigh(A)
                # vec1 = vecs[:,0]
                # vec2 = vecs[:,1]

                # lamb1 = eigs[0]
                # lamb2 = eigs[1]
                # lamb3 =  (-J*lamb*s1*s2 + lamb*s1*s2 + mu)/(s1**2*s2**2)

                # e1 = vec1[0] * d1 + vec1[1] * d2
                # e2 = vec2[0] * d1 + vec2[1] * d2

                # H = lamb1 * np.outer(e1,e1) + lamb2 * np.outer(e2,e2) + lamb3 * np.outer(l,l)


                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 =  (-J*lamb*s1*s2 + lamb*s1*s2 + mu)/J**2

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)

                ds = np.array([d1,d2]).T
                recA = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
                Hw = ds.dot(recA.dot(ds.T))

                # H = Hw + lamb3 * np.outer(l,l) + lamb4 * np.outer(t,t)
                H = Hw + lamb3 * np.outer(l,l)
                H_Voigt = GetVoigtHessian(H)

                # Classical way
                # N1 = V[0,:]
                # N2 = V[1,:]
                # S11 =  (lamb*s2*(J - 1) + mu*s1 - mu/s1)/s1
                # S22 =  (lamb*s1*(J - 1) + mu*s2 - mu/s2)/s2

                # Nabab = np.einsum("ij,kl",np.outer(N1,N2),np.outer(N1,N2))
                # Nabba = np.einsum("ij,kl",np.outer(N1,N2),np.outer(N2,N1))

                # Nbaba = np.einsum("ij,kl",np.outer(N2,N1),np.outer(N2,N1))
                # Nbaab = np.einsum("ij,kl",np.outer(N2,N1),np.outer(N1,N2))
                # # np.einsum("a,b,a,b", N1, N2, N1, N2) + np.einsum("a,b,b,a", N1, N2, N2, N1))
                # H2 = np.zeros((self.ndim, self.ndim, self.ndim, self.ndim))
                # if np.isclose(s1, s2):
                #     # This branch seems not right
                #     H2 += 2. * (a11 - a12) * (Nabab + Nabba)
                #     H2 += 2. * (a22 - a12) * (Nbaba + Nbaab)
                #     # print(vec(H2))
                #     # print(a11,a22)
                # else:
                #     # This branch gives correct results
                #     H2 += (S11 - S22) / (s1**2 - s2**2) * (Nabab + Nabba)
                #     H2 += (S22 - S11) / (s2**2 - s1**2) * (Nbaba + Nbaab)

                # # print(vec(H2))
                # # H1 = Hw + vec(H2)
                # # H_Voigt = GetVoigtHessian(H1)

                # # print(vec(Nabab))
                # # print(vec(Nabba))
                # # print(vec(Nbaba))
                # # print(vec(Nbaab))

                # diff = H1 - vec(H)
                # makezero(diff, 1e-11)
                # print(diff)

            elif self.ndim == 3:

                m_mu = self.mu
                m_lambda = self.lamb

                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                a11 =  (m_lambda*s2**2*s3**2 + m_mu + m_mu/s1**2)/s1**2 + (-m_lambda*s2*s3*(J - 1) - m_mu*s1 + m_mu/s1)/s1**3
                a22 =  (m_lambda*s1**2*s3**2 + m_mu + m_mu/s2**2)/s2**2 + (-m_lambda*s1*s3*(J - 1) - m_mu*s2 + m_mu/s2)/s2**3
                a33 =  (m_lambda*s1**2*s2**2 + m_mu + m_mu/s3**2)/s3**2 + (-m_lambda*s1*s2*(J - 1) - m_mu*s3 + m_mu/s3)/s3**3
                a12 =  (J*m_lambda*s3 + m_lambda*s3*(J - 1))/(s1*s2)
                a13 =  (J*m_lambda*s2 + m_lambda*s2*(J - 1))/(s1*s3)
                a23 =  (J*m_lambda*s1 + m_lambda*s1*(J - 1))/(s2*s3)

                Hw = np.array([
                    [a11,a12,a13],
                    [a12,a22,a23],
                    [a13,a23,a33]
                    ])

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 = eigs[2]
                lamb4 =  (-J*m_lambda*s1*s2*s3 + m_lambda*s1*s2*s3 + m_mu)/(s2**2*s3**2)
                lamb5 =  (-J*m_lambda*s1*s2*s3 + m_lambda*s1*s2*s3 + m_mu)/(s1**2*s3**2)
                lamb6 =  (-J*m_lambda*s1*s2*s3 + m_lambda*s1*s2*s3 + m_mu)/(s1**2*s2**2)

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)
                    lamb4 = max(lamb4, hessian_eps)
                    lamb5 = max(lamb5, hessian_eps)
                    lamb6 = max(lamb6, hessian_eps)

                [d1, d2, d3, l1, l2, l3] = GetEigenMatrices(U, V)
                ds = np.array([d1,d2,d3]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                    + lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])

                H = ds.dot(HwSPD.dot(ds.T)) + lamb4 * np.outer(l1,l1) + lamb5 * np.outer(l2,l2) + lamb6 * np.outer(l3,l3)
                H_Voigt = GetVoigtHessian(H)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        C = F.T.dot(F)
        invC = np.linalg.inv(C)
        c = J*J

        stress_stab = 0.

        if self.formulation_style == "C":
            stress = mu*I - mu * invC + lamb * (c - J) * invC
            stress_stab = stress

            # [U, S, V] = ssvd(F)
            # s1 = S[0]
            # s2 = S[1]
            # s3 = S[2]

            # sigmaS = np.zeros(3)
            # sigmaS[0] =  (lamb*s2*s3*(J - 1) + mu*s1 - mu/s1)/s1
            # sigmaS[1] =  (lamb*s1*s3*(J - 1) + mu*s2 - mu/s2)/s2
            # sigmaS[2] =  (lamb*s1*s2*(J - 1) + mu*s3 - mu/s3)/s3

            # stress2 = V.dot(np.diag(sigmaS).dot(V.T))

        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if self.ndim == 2:

                s1 = S[0]
                s2 = S[1]

                sigmaS = np.zeros(2)
                sigmaS[0] =  (lamb*s2*(J - 1) + mu*s1 - mu/s1)/s1
                sigmaS[1] =  (lamb*s1*(J - 1) + mu*s2 - mu/s2)/s2

                stress = V.dot(np.diag(sigmaS).dot(V.T))

            elif self.ndim == 3:

                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                sigmaS = np.zeros(3)
                # sigmaS[0] =  (lamb*s2*s3*(J - 1) + mu*s1 - mu/s1)/s1
                # sigmaS[1] =  (lamb*s1*s3*(J - 1) + mu*s2 - mu/s2)/s2
                # sigmaS[2] =  (lamb*s1*s2*(J - 1) + mu*s3 - mu/s3)/s3
                sigmaS[0] =  (-mu + s1*(lamb*s2*s3*(J - 1) + mu*s1))/s1**2
                sigmaS[1] =  (-mu + s2*(lamb*s1*s3*(J - 1) + mu*s2))/s2**2
                sigmaS[2] =  (-mu + s3*(lamb*s1*s2*(J - 1) + mu*s3))/s3**2

                stress = V.dot(np.diag(sigmaS).dot(V.T))

        return stress


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb

        I2 = trace(F.T.dot(F))
        energy  = 0.5 * mu * I2 - mu * np.log(J) + lamb / 2. * (J-1)**2

        return energy









class MIPS_C(Material):
    """
        W(F) = mu * II_F / d / J**(2/d) + lambda * (J + J**(-1) - 2)
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPS_C, self).__init__(mtype, ndim, **kwargs)

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
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        C = F.T.dot(F)
        c = J*J
        trC = trace(C)

        if self.formulation_style != "classic" and self.formulation_style != "ps":
            raise ValueError("Invalid invariant formulation for this material model")

        if self.formulation_style == "classic":

            d2 = d * d
            d3 = d2 * d
            tmp1 = (J**2)**(1. / d)
            invC = np.linalg.inv(C)

            H  = (lamb * (J + 1. / J) + 4. * mu * trC / d3 / tmp1) * np.einsum("ij,kl", invC, invC) +\
                (-2 * lamb * (J - 1. / J) + 4. * mu * trC / d2 / tmp1) * ( np.einsum("ik,jl", invC, invC) + np.einsum("il,jk", invC, invC) ) * 0.5 -\
                4. * mu / d2 / tmp1 * ( np.einsum("ij,kl", I, invC) + np.einsum("ij,kl", invC, I) )

            H_Voigt = GetVoigtHessian(H)


        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if self.ndim == 2:
                s1 = S[0]
                s2 = S[1]


                a11 = (3. * trC * mu) / 2. / J / s1**4 - 2. * mu / J / s1**2  - lamb / s1**4 * ( J - 3. / J)
                a22 = (3. * trC * mu) / 2. / J / s2**4 - 2. * mu / J / s2**2  - lamb / s2**4 * ( J - 3. / J)
                a12 =  lamb * (1. / J + 1. / J**3) - trC * mu / 2. / J**3

                Hw = np.array([
                    [a11,a12],
                    [a12,a22],
                    ])

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 =  trC * mu / 2. / J**3 - lamb * (J**2 - 1) / J**3

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)

                [d1, d2, l] = GetEigenMatrices(U, V)
                ds = np.array([d1,d2]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
                H = ds.dot(HwSPD.dot(ds.T)) + lamb3 * np.outer(l,l)

                H_Voigt = GetVoigtHessian(H)


            elif self.ndim == 3:

                m_mu = self.mu
                m_lambda = self.lamb

                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                I2 = trC
                tmp1 = (J**2.)**(1./3.)

                a11 =  (10*I2*m_mu/(27*tmp1*s1**2) + 2*m_lambda/(J*s1**2) - 2*m_mu/(9*J**(2/3)))/s1**2 + (2*I2*m_mu/(9*tmp1*s1) - m_lambda*(s2*s3 - 1/(J*s1)) - 2*m_mu*s1/(3*tmp1))/s1**3
                a22 =  (10*I2*m_mu/(27*tmp1*s2**2) + 2*m_lambda/(J*s2**2) - 2*m_mu/(9*J**(2/3)))/s2**2 + (2*I2*m_mu/(9*tmp1*s2) - m_lambda*(s1*s3 - 1/(J*s2)) - 2*m_mu*s2/(3*tmp1))/s2**3
                a33 =  (10*I2*m_mu/(27*tmp1*s3**2) + 2*m_lambda/(J*s3**2) - 2*m_mu/(9*J**(2/3)))/s3**2 + (2*I2*m_mu/(9*tmp1*s3) - m_lambda*(s1*s2 - 1/(J*s3)) - 2*m_mu*s3/(3*tmp1))/s3**3
                a12 =  (4*I2*m_mu/(27*tmp1*s1*s2) + m_lambda*(s3 + 1/(J*s1*s2)) - 4*m_mu*s1/(9*J**(2/3)*s2) - 4*m_mu*s2/(9*tmp1*s1))/(s1*s2)
                a13 =  (4*I2*m_mu/(27*tmp1*s1*s3) + m_lambda*(s2 + 1/(J*s1*s3)) - 4*m_mu*s1/(9*J**(2/3)*s3) - 4*m_mu*s3/(9*tmp1*s1))/(s1*s3)
                a23 =  (4*I2*m_mu/(27*tmp1*s2*s3) + m_lambda*(s1 + 1/(J*s2*s3)) - 4*m_mu*s2/(9*J**(2/3)*s3) - 4*m_mu*s3/(9*tmp1*s2))/(s2*s3)

                Hw = np.array([
                    [a11,a12,a13],
                    [a12,a22,a23],
                    [a13,a23,a33]
                    ])

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 = eigs[2]
                lamb4 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s2**2*s3**2)
                lamb5 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s1**2*s3**2)
                lamb6 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s1**2*s2**2)

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)
                    lamb4 = max(lamb4, hessian_eps)
                    lamb5 = max(lamb5, hessian_eps)
                    lamb6 = max(lamb6, hessian_eps)

                [d1, d2, d3, l1, l2, l3] = GetEigenMatrices(U, V)
                ds = np.array([d1,d2,d3]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                    + lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])

                H = ds.dot(HwSPD.dot(ds.T)) + lamb4 * np.outer(l1,l1) + lamb5 * np.outer(l2,l2) + lamb6 * np.outer(l3,l3)
                H_Voigt = GetVoigtHessian(H)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        I = StrainTensors['I']

        mu = self.mu
        lamb = self.lamb
        ndim = self.ndim

        C = F.T.dot(F)
        trC = trace(C)
        S_stab = []

        if self.formulation_style == "classic":

            d = self.ndim
            d2 = d * d
            invC = np.linalg.inv(C)

            tmp1 = (J**2)**(1. / d)
            tmp2 = 1. / (tmp1 * J)

            S = 2. * mu / d / tmp1 * I + ( lamb * (J - 1. / J) - 2. * mu * trC / d2 / tmp1) * invC

        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if ndim == 2:
                s1 = S[0]
                s2 = S[1]

                sigmaS = np.zeros(ndim)
                sigmaS[0] = mu / J - trC * mu / 2. / J / s1**2 + lamb / s1**2 * (J - 1. / J)
                sigmaS[1] = mu / J - trC * mu / 2. / J / s2**2 + lamb / s2**2 * (J - 1. / J)

                S = V.dot(np.diag(sigmaS).dot(V.T))

                if self.stabilise_tangents:
                    sigmaS[sigmaS<0] = self.tangent_stabiliser_value
                    S_stab = V.dot(np.diag(sigmaS).dot(V.T))

            elif ndim == 3:

                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                I2 = trC
                m_mu = self.mu
                m_lambda = self.lamb
                tmp1 = (J**2.)**(1./3.)

                sigmaS = np.zeros(ndim)
                sigmaS[0] =  (-2*I2*m_mu/(9*tmp1*s1) + m_lambda*(s2*s3 - 1/(J*s1)) + 2*m_mu*s1/(3*tmp1))/s1
                sigmaS[1] =  (-2*I2*m_mu/(9*tmp1*s2) + m_lambda*(s1*s3 - 1/(J*s2)) + 2*m_mu*s2/(3*tmp1))/s2
                sigmaS[2] =  (-2*I2*m_mu/(9*tmp1*s3) + m_lambda*(s1*s2 - 1/(J*s3)) + 2*m_mu*s3/(3*tmp1))/s3

                S = V.dot(np.diag(sigmaS).dot(V.T))

                if self.stabilise_tangents:
                    sigmaS[sigmaS<0] = self.tangent_stabiliser_value
                    S_stab = V.dot(np.diag(sigmaS).dot(V.T))

        return S, S_stab


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        trb = trace(F.T.dot(F))

        energy = mu * trb / d / (J**2)**(1. / d) + lamb * (J + 1./J - 2)

        return energy








class MIPS_b(Material):
    """
        W(F) = mu * II_F / d / J**(2/d) + lambda * (J + J**(-1) - 2)
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(MIPS_b, self).__init__(mtype, ndim, **kwargs)

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
        from Florence.MaterialLibrary.LLDispatch._NeoHookean_ import KineticMeasures
        return KineticMeasures(self,F)


    def Hessian(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        C = F.T.dot(F)
        c = J*J
        trC = trace(C)

        if self.formulation_style != "classic" and self.formulation_style != "ps":
            raise ValueError("Invalid invariant formulation for this material model")


        if self.formulation_style == "classic":

            d2 = d * d
            d3 = d2 * d
            tmp1 = (J**2)**(1. / d)
            invC = np.linalg.inv(C)

            H  = (lamb * (J + 1. / J) + 4. * mu * trC / d3 / tmp1) * np.einsum("ij,kl", invC, invC) +\
                (-2 * lamb * (J - 1. / J) + 4. * mu * trC / d2 / tmp1) * ( np.einsum("ik,jl", invC, invC) + np.einsum("il,jk", invC, invC) ) * 0.5 -\
                4. * mu / d2 / tmp1 * ( np.einsum("ij,kl", I, invC) + np.einsum("ij,kl", invC, I) )

            H = 1./J * np.einsum("iI,jJ,kK,lL,IJKL",F,F,F,F,H)
            H_Voigt = Voigt(H,1)


        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if self.ndim == 2:

                s1 = S[0]
                s2 = S[1]

                a11 = (3. * trC * mu) / 2. / J / s1**4 - 2. * mu / J / s1**2  - lamb / s1**4 * ( J - 3. / J)
                a22 = (3. * trC * mu) / 2. / J / s2**4 - 2. * mu / J / s2**2  - lamb / s2**4 * ( J - 3. / J)
                a12 =  lamb * (1. / J + 1. / J**3) - trC * mu / 2. / J**3

                Hw = np.array([
                    [a11,a12],
                    [a12,a22],
                    ])
                # Build spatial Hessian
                S2 = np.diag(S).dot(np.diag(S))
                Hw = S2.dot(Hw.dot(S2))

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 =  trC * mu / 2. / J**3 - lamb * (J**2 - 1) / J**3

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)

                # Push-forward flip eigenvalues
                lamb3 *= (s1*s2)**2

                [d1, d2, l] = GetEigenMatrices_UL(U, V)
                ds = np.array([d1,d2]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:])
                H = ds.dot(HwSPD.dot(ds.T)) + lamb3 * np.outer(l,l)
                H /= J

                H_Voigt = GetVoigtHessian(H)

            elif self.ndim == 3:

                m_mu = self.mu
                m_lambda = self.lamb

                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                I2 = trC
                tmp1 = (J**2.)**(1./3.)

                a11 =  (10*I2*m_mu/(27*tmp1*s1**2) + 2*m_lambda/(J*s1**2) - 2*m_mu/(9*J**(2/3)))/s1**2 + (2*I2*m_mu/(9*tmp1*s1) - m_lambda*(s2*s3 - 1./(J*s1)) - 2*m_mu*s1/(3*tmp1))/s1**3
                a22 =  (10*I2*m_mu/(27*tmp1*s2**2) + 2*m_lambda/(J*s2**2) - 2*m_mu/(9*J**(2/3)))/s2**2 + (2*I2*m_mu/(9*tmp1*s2) - m_lambda*(s1*s3 - 1./(J*s2)) - 2*m_mu*s2/(3*tmp1))/s2**3
                a33 =  (10*I2*m_mu/(27*tmp1*s3**2) + 2*m_lambda/(J*s3**2) - 2*m_mu/(9*J**(2/3)))/s3**2 + (2*I2*m_mu/(9*tmp1*s3) - m_lambda*(s1*s2 - 1./(J*s3)) - 2*m_mu*s3/(3*tmp1))/s3**3
                a12 =  (4*I2*m_mu/(27*tmp1*s1*s2) + m_lambda*(s3 + 1/(J*s1*s2)) - 4*m_mu*s1/(9*J**(2/3)*s2) - 4*m_mu*s2/(9*tmp1*s1))/(s1*s2)
                a13 =  (4*I2*m_mu/(27*tmp1*s1*s3) + m_lambda*(s2 + 1/(J*s1*s3)) - 4*m_mu*s1/(9*J**(2/3)*s3) - 4*m_mu*s3/(9*tmp1*s1))/(s1*s3)
                a23 =  (4*I2*m_mu/(27*tmp1*s2*s3) + m_lambda*(s1 + 1/(J*s2*s3)) - 4*m_mu*s2/(9*J**(2/3)*s3) - 4*m_mu*s3/(9*tmp1*s2))/(s2*s3)

                Hw = np.array([
                    [a11,a12,a13],
                    [a12,a22,a23],
                    [a13,a23,a33]
                    ])
                # Build spatial Hessian
                S2 = np.diag(S).dot(np.diag(S))
                Hw = S2.dot(Hw.dot(S2))

                eigs, vecs = sp.linalg.eigh(Hw)
                lamb1 = eigs[0]
                lamb2 = eigs[1]
                lamb3 = eigs[2]
                lamb4 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s2**2*s3**2)
                lamb5 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s1**2*s3**2)
                lamb6 =  (2*I2*J**(1./3)*m_mu/9. - J**2*m_lambda + m_lambda)/(J*s1**2*s2**2)

                if self.stabilise_tangents:
                    hessian_eps = self.tangent_stabiliser_value
                    lamb1 = max(lamb1, hessian_eps)
                    lamb2 = max(lamb2, hessian_eps)
                    lamb3 = max(lamb3, hessian_eps)
                    lamb4 = max(lamb4, hessian_eps)
                    lamb5 = max(lamb5, hessian_eps)
                    lamb6 = max(lamb6, hessian_eps)

                # Push-forward flip eigenvalues
                lamb4 *= (s2*s3)**2
                lamb5 *= (s1*s3)**2
                lamb6 *= (s1*s2)**2

                [d1, d2, d3, l1, l2, l3] = GetEigenMatrices_UL(U, V)
                ds = np.array([d1,d2,d3]).T
                HwSPD = lamb1 * vecs[:,0][None,:].T.dot(vecs[:,0][None,:]) + lamb2 * vecs[:,1][None,:].T.dot(vecs[:,1][None,:]) +\
                    + lamb3 * vecs[:,2][None,:].T.dot(vecs[:,2][None,:])

                H = ds.dot(HwSPD.dot(ds.T)) + lamb4 * np.outer(l1,l1) + lamb5 * np.outer(l2,l2) + lamb6 * np.outer(l3,l3)
                H_Voigt = GetVoigtHessian(H)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]
        I = StrainTensors['I']

        mu = self.mu
        lamb = self.lamb
        ndim = self.ndim

        C = F.T.dot(F)
        trC = trace(C)

        cauchy_stab = []

        if self.formulation_style == "classic":

            d = self.ndim
            d2 = d * d
            invC = np.linalg.inv(C)

            tmp1 = (J**2)**(1. / d) #J**(2. / d)
            tmp2 = 1. / (tmp1 * J)

            S = 2. * mu / d / tmp1 * I + ( lamb * (J - 1. / J) - 2. * mu * trC / d2 / tmp1) * invC

            cauchy = 1./J * np.dot(F,np.dot(S,F.T))


        elif self.formulation_style == "ps":

            [U, S, V] = ssvd(F)

            if ndim == 2:
                s1 = S[0]
                s2 = S[1]

                cauchyS = np.zeros(ndim)
                cauchyS[0] = mu / J - trC * mu / 2. / J / s1**2 + lamb / s1**2 * (J - 1. / J)
                cauchyS[1] = mu / J - trC * mu / 2. / J / s2**2 + lamb / s2**2 * (J - 1. / J)

                cauchyS[0] *= s1**2
                cauchyS[1] *= s2**2
                cauchyS /= J

                cauchy = U.dot(np.diag(cauchyS).dot(U.T))

                if self.stabilise_tangents:
                    cauchyS[cauchyS<0] = self.tangent_stabiliser_value
                    cauchy_stab = U.dot(np.diag(cauchyS).dot(U.T))


            elif ndim == 3:
                s1 = S[0]
                s2 = S[1]
                s3 = S[2]

                I2 = trC
                m_mu = self.mu
                m_lambda = self.lamb
                tmp1 = (J**2.)**(1./3.)

                cauchyS = np.zeros(ndim)
                cauchyS[0] =  (-2*I2*m_mu/(9*tmp1*s1) + m_lambda*(s2*s3 - 1/(J*s1)) + 2*m_mu*s1/(3*tmp1))/s1
                cauchyS[1] =  (-2*I2*m_mu/(9*tmp1*s2) + m_lambda*(s1*s3 - 1/(J*s2)) + 2*m_mu*s2/(3*tmp1))/s2
                cauchyS[2] =  (-2*I2*m_mu/(9*tmp1*s3) + m_lambda*(s1*s2 - 1/(J*s3)) + 2*m_mu*s3/(3*tmp1))/s3

                cauchyS[0] *= s1**2
                cauchyS[1] *= s2**2
                cauchyS[2] *= s3**2
                cauchyS /= J

                cauchy = U.dot(np.diag(cauchyS).dot(U.T))

                if self.stabilise_tangents:
                    cauchyS[cauchyS<0] = self.tangent_stabiliser_value
                    cauchy_stab = U.dot(np.diag(cauchyS).dot(U.T))


        return cauchy, cauchy_stab


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        mu = self.mu
        lamb = self.lamb
        d = self.ndim

        trb = trace(F.T.dot(F))

        energy = mu * trb / d / J**(2/d) + lamb * (J + 1./J - 2)

        return energy























__all__ = ["CBasedDisplacementFormulation"]

class CBasedDisplacementFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False, quadrature_degree=None):

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(CBasedDisplacementFormulation, self).__init__(mesh,variables_order=self.variables_order,
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

        # # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = material.Hessian(StrainTensors,None,elem,counter)

            CauchyStressTensor = []; stabilised_conjugate = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)
                if isinstance(CauchyStressTensor, tuple):
                    stabilised_conjugate = CauchyStressTensor[1]
                    CauchyStressTensor = CauchyStressTensor[0]

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter, :, :], F[counter, :, :],
                CauchyStressTensor, H_Voigt, requires_geometry_update=fem_solver.requires_geometry_update)

            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if material.nature != "linear":
                # DO NOT DO OVERRIDE ACTUAL STRESSES HERE AS TRACTIONS ARE MISCALCULATED
                if not material.stabilise_tangents:
                    stabilised_conjugate = CauchyStressTensor

                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],stabilised_conjugate)

            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        makezero(stiffness, 1e-12)
        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, F, CauchyStressTensor, H_Voigt,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        # makezero(SpatialGradient, 1e-12)
        FillConstitutiveBC(B,SpatialGradient,F,self.ndim,self.nvar)

        BDB = B.dot(H_Voigt.dot(B.T))

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = GetTotalTraction(CauchyStressTensor)
            t = np.dot(B,TotalTraction)

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

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor, _ = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            t = self.TractionIntegrand(B, SpatialGradient[counter,:,:], F[counter,:,:],
                CauchyStressTensor, requires_geometry_update=fem_solver.requires_geometry_update)

            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

        return tractionforce


    def TractionIntegrand(self, B, SpatialGradient, F, CauchyStressTensor,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveBC(B,SpatialGradient,F,self.ndim,self.nvar)

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = GetTotalTraction(CauchyStressTensor)
            t = np.dot(B,TotalTraction)

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


