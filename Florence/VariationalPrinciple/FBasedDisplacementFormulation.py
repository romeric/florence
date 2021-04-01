import numpy as np
from Florence.VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from Florence.Tensor import issymetric
from Florence.Tensor import makezero


def vec(H):
    ndim = H.shape[0]
    if H.ndim == 4:
        # print(H.shape)
        # H = np.einsum("ijlk",H)
        x = H.flatten().reshape(ndim**2,ndim**2)
        # H1 = np.random.rand(2,2,2,2)
        # print(x)
        # print()
        # HH = np.zeros((ndim**2,ndim**2))
        # for i in range(H.ndim):
            # HH[:,i] = H[i,:,:,:]
            # print(H[:,:,:,i])
        # print(H.flatten())
        # makezero(x)
        # x += x.T
        # x /= 2.
        # print(H)
        # s = np.linalg.svd(x)[1]
        # print(s)
        # exit()
        return x
        # return H.flatten().reshape(ndim**2,ndim**2)
    else:
        # return H.flatten()
        return H.T.flatten() # careful - ARAP needs this


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

import numpy as np
from numpy import einsum
from Florence.MaterialLibrary.MaterialBase import Material
from Florence.Tensor import trace, Voigt, makezero

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
            J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb

        invF = np.linalg.inv(F)
        invFt = invF.T.copy()

        # BECAREFUL IJKL OF F,F or invF,invF is not symmetric

        # H = mu * np.einsum("ik,jl", I, I) + lamb * np.einsum("ij,kl", invFt, invFt) +\
        #     (mu-lamb*np.log(J)) * np.einsum("ik,jl", invFt, invFt)

        # This symmetrisation is wrong perhaps
        # dum = 0.25 * (lamb * np.einsum("ij,kl", invFt, invFt) + lamb * np.einsum("ij,lk", invFt, invFt) +\
        #     lamb * np.einsum("ji,kl", invFt, invFt) + lamb * np.einsum("ji,lk", invFt, invFt) )

        dum = lamb * np.einsum("ij,kl", invFt, invFt)
        H = mu * 0.5 * (np.einsum("ik,jl", I, I) + np.einsum("il,jk", I, I)) + dum +\
            (mu-lamb*np.log(J)) * 0.5 * (np.einsum("ik,jl", invFt, invFt) + np.einsum("il,jk", invFt, invFt))
            # (mu-lamb*np.log(J)) * 0.5 * (np.einsum("ik,jl->ijkl", invFt, invFt) + np.einsum("il,jk->ijkl", invFt, invFt))

        # C_Voigt = lamb/J * np.einsum("ij,kl",I,I) + 1./J * (mu - lamb*np.log(J)) * (np.einsum("ik,jl",I,I) + np.einsum("il,jk",I,I))
        # stress = mu/J*(b-I) + lamb/J*np.log(J)*I
        # C_Voigt = (lamb * (2*J-1) - mu) * np.einsum("ij,kl",I,I) + (mu - lamb * (J-1))  * (np.einsum("ik,jl",I,I) + np.einsum("il,jk",I,I))
        # stress = 1.0*mu/J*b + (lamb*(J-1) - mu)*I
        # H = J * np.einsum("Jj,ijkl,Ll->iJkL",invF,(C_Voigt + np.einsum("ij,kl",stress,I)),invF)

        # H = mu * np.einsum("ij,kl", I, I) + lamb * np.einsum("ij,kl", invFt, invFt) +\
        #     (mu-lamb*np.log(J)) * np.einsum("ik,jl", invFt, invFt)
            # (mu-lamb*np.log(J)) * 1*(np.einsum("ik,jl", invFt, invFt) + np.einsum("il,jk", invFt, invFt)) # indefinite

        # For F based formulation we need to bring everything to reference domain, partial pull back

        H = np.einsum("klij",H) # this symmetry should be preserved anyway
        H = np.einsum("ijlk",H)

        H = vec(H)
        # H += H.T
        # H /= 2.

        self.H_VoigtSize = H.shape[0]

        return H

    def CauchyStress(self,StrainTensors,ElectricFieldx=None,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        mu = self.mu
        lamb = self.lamb
        invF = np.linalg.inv(F)
        invFt = invF.T.copy()

        stress = mu*F - (mu-lamb*np.log(J)) * invFt

        return stress.T # careful


    def InternalEnergy(self,StrainTensors,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        F = StrainTensors['F'][gcounter]

        if np.isclose(J, 0) or J < 0:
            delta = np.sqrt(0.04 * J * J + 1e-8);
            J = 0.5 * (J + np.sqrt(J**2 + 4 *delta**2))

        # energy  = mu/2.*(trace(C) - 3.) - mu*np.log(J) + lamb/2.*(J-1.)**2

        return energy


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
        # print(u,s,vh)
        # exit()
        if self.ndim == 2:
            s1 = s[0]
            s2 = s[1]
            T = np.array([[0.,-1],[1,0.]])
            T = 1./np.sqrt(2) * np.dot(u, np.dot(T, vh.T))
            t =  vec(T)
            H = np.eye(4,4)
            I_1 = s.sum()
            # filtered = 2.0 / I_1 if I_1 >= 2.0 else 1.0
            filtered = 1.0
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
            # s1s2 = s1 + s2
            # s1s3 = s1 + s3
            # s2s3 = s2 + s3
            # if (s1s2 < 2.0):
            #     s1s2 = 2.0
            # if (s1s3 < 2.0):
            #     s1s3 = 2.0
            # if (s2s3 < 2.0):
            #     s2s3 = 2.0
            # lamb1 = 2. / (s1s2)
            # lamb2 = 2. / (s1s3)
            # lamb3 = 2. / (s2s3)

            t1 = vec(T1)
            t2 = vec(T2)
            t3 = vec(T3)

            H = 2. * np.eye(9,9);
            H = H - (4. / (s0 + s1)) * np.outer(t1 , t1);
            H = H - (4. / (s1 + s2)) * np.outer(t2 , t2);
            H = H - (4. / (s0 + s2)) * np.outer(t3 , t3);

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

        sigma = 2. * (F - R)
        # print(sigma)
        # exit()

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
        SpatialGradient = np.einsum('ikj',MaterialGradient)
        # COMPUTE ONCE detJ
        # detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))
        detJ = np.einsum('i,i->i',AllGauss[:,0],det(ParentGradientX))
        # detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))

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
