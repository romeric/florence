from copy import deepcopy
import gc
from numpy.linalg import det, inv, norm, cond
from Florence import QuadratureRule, FunctionSpace
from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from ._ConstitutiveStiffnessDF_ import __ConstitutiveStiffnessIntegrandDF__
from Florence.Tensor import issymetric
from Florence.LegendreTransform import LegendreTransform
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from .VariationalPrinciple import *

from Florence.FiniteElements.Assembly.SparseAssemblyNative import SparseAssemblyNative
from Florence.FiniteElements.Assembly.RHSAssemblyNative import RHSAssemblyNative



__all__ = ["CoupleStressFormulation"]

class CoupleStressFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,0,0), subtype="lagrange_multiplier",
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=False,
        equally_spaced_bases=False, save_condensed_matrices=True):
        """

            Input:
                subtype:                    [str] either "lagrange_multiplier", "augmented_lagrange" or "penalty"
        """

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(CoupleStressFormulation, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces,
            compute_post_quadrature=compute_post_quadrature)

        self.fields = "couple_stress"
        self.nvar = self.ndim
        self.subtype = subtype
        self.save_condensed_matrices = save_condensed_matrices

        C = mesh.InferPolynomialDegree() - 1
        mesh.InferBoundaryElementType()

        if C < 1:
            raise ValueError("Incorrect initial mesh provided for the formulation. Mesh has to be at least order 2")

        # CHECK IF MESH IS APPROPRIATE
        # if C == 0:
            # warn('Mesh not appropriate for formulation')
            # raise ValueError('Mesh not appropriate for formulation. p>1 for primary variable (displacement)')
        # BUILD MESHES FOR ALL FIELDS
        p = C+1
        # DISPLACEMENTS
        mesh0 = deepcopy(mesh)
        # ROTATIONS
        mesh1 = deepcopy(mesh)
        mesh1 = mesh1.GetLinearMesh(remap=True)
        mesh1.GetHighOrderMesh(p=p-1)
        # LAGRANGE MULTIPLIER
        mesh2 = deepcopy(mesh)
        mesh2 = mesh2.GetLinearMesh(remap=True)
        mesh2.GetHighOrderMesh(p=p-1)
        # ALL MESHES
        self.meshes = (mesh0,mesh1,mesh2)


        # GET QUADRATURE RULES
        norder = C+2
        if mesh.element_type == "quad" or mesh.element_type == "hex":
            norder = C+1

        if quadrature_rules == None and self.quadrature_rules == None:
            # FOR DISPLACEMENT
            quadrature0 = QuadratureRule(optimal=3, norder=self.GetQuadratureOrder(norder,mesh.element_type)[0],
                mesh_type=mesh.element_type, is_flattened=False)
            # FOR ROTATIONS
            quadrature1 = QuadratureRule(optimal=3, norder=self.GetQuadratureOrder(norder,mesh.element_type)[0],
                mesh_type=mesh.element_type, is_flattened=False)
            # FOR LAGRANGE MULTIPLIER
            quadrature2 = QuadratureRule(optimal=3, norder=self.GetQuadratureOrder(norder,mesh.element_type)[0],
                mesh_type=mesh.element_type, is_flattened=False)
            # BOUNDARY
            bquadrature = QuadratureRule(optimal=3, norder=C+2, mesh_type=mesh.boundary_element_type, is_flattened=False)

            self.quadrature_rules = (quadrature0,quadrature1,quadrature2,bquadrature)
        else:
            self.quadrature_rules = quadrature_rules


        # GET FUNCTIONAL SPACES
        if function_spaces == None and self.function_spaces == None:
            # FOR DISPLACEMENT
            function_space0 = FunctionSpace(mesh0, self.quadrature_rules[0], p=mesh0.degree,
                equally_spaced=equally_spaced_bases, use_optimal_quadrature=False)
            # FOR ROTATIONS
            function_space1 = FunctionSpace(mesh1, self.quadrature_rules[1], p=mesh1.degree,
                equally_spaced=equally_spaced_bases, use_optimal_quadrature=False)
            # FOR LAGRANGE MULTIPLIER
            function_space2 = FunctionSpace(mesh2, self.quadrature_rules[2], p=mesh2.degree,
                equally_spaced=equally_spaced_bases, use_optimal_quadrature=False)
            # BOUNDARY
            bfunction_space = FunctionSpace(mesh0.CreateDummyLowerDimensionalMesh(), self.quadrature_rules[3], p=mesh0.degree,
                equally_spaced=equally_spaced_bases, use_optimal_quadrature=False)

            self.function_spaces = (function_space0, function_space1, function_space2, bfunction_space)
        else:
            self.function_spaces = function_spaces


        # local_size = function_space.Bases.shape[0]*self.nvar
        local_size = self.function_spaces[0].Bases.shape[0]*self.nvar
        self.local_rows = np.repeat(np.arange(0,local_size),local_size,axis=0)
        self.local_columns = np.tile(np.arange(0,local_size),local_size)
        self.local_size = local_size

        # FOR MASS
        local_size_m = self.function_spaces[0].Bases.shape[0]*self.nvar
        self.local_rows_mass = np.repeat(np.arange(0,local_size_m),local_size_m,axis=0)
        self.local_columns_mass = np.tile(np.arange(0,local_size_m),local_size_m)
        self.local_size_m = local_size_m


        if self.save_condensed_matrices:
            # elist = [0]*mesh.nelem # CANT USE ONE PRE-CREATED LIST AS IT GETS MODIFIED
            # KEEP VECTORS AND MATRICES SEPARATE BECAUSE OF THE SAME REASON
            if self.subtype == "lagrange_multiplier":
                self.condensed_matrices = {'k_uu':[0]*mesh.nelem,'k_us':[0]*mesh.nelem,
                'k_ww':[0]*mesh.nelem,'k_ws':[0]*mesh.nelem,'inv_k_ws':[0]*mesh.nelem}
                self.condensed_vectors = {'tu':[0]*mesh.nelem,'tw':[0]*mesh.nelem,'ts':[0]*mesh.nelem}
            elif self.subtype == "augmented_lagrange":
                self.condensed_matrices = {'k_uu':[0]*mesh.nelem,'k_us':[0]*mesh.nelem,
                'k_ww':[0]*mesh.nelem,'k_ws':[0]*mesh.nelem,'k_ss':[0]*mesh.nelem,'inv_k_ws':[0]*mesh.nelem}
                self.condensed_vectors = {'tu':[0]*mesh.nelem,'tw':[0]*mesh.nelem,'ts':[0]*mesh.nelem}
            elif self.subtype == "penalty":
                self.condensed_matrices = {'k_uu':[0]*mesh.nelem,'k_uw':[0]*mesh.nelem,'k_ww':[0]*mesh.nelem}
                self.condensed_vectors = {'tu':[0]*mesh.nelem,'tw':[0]*mesh.nelem}

        # COMPUTE THE COMMON/NEIGHBOUR NODES ONCE
        self.all_nodes = np.unique(self.meshes[1].elements)
        self.Elss, self.Poss = self.meshes[1].GetNodeCommonality()[:2]


    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerw, Eulers, Eulerp):

        massel=[]; f = []
        # COMPUTE THE STIFFNESS MATRIX
        if material.has_low_level_dispatcher:
            stiffnessel, t = self.__GetLocalStiffness__(material, fem_solver, Eulerx, Eulerw, Eulers, Eulerp, elem)
        else:
            stiffnessel, t = self.GetLocalStiffness(material, fem_solver, Eulerx, Eulerw, Eulers, Eulerp, elem)

        I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed is False:
            # GET THE FIELDS AT THE ELEMENT LEVEL
            LagrangeElemCoords = mesh[0].points[mesh[0].elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh[0].elements[elem,:],:]
            # COMPUTE THE MASS MATRIX
            if material.has_low_level_dispatcher:
                massel = self.__GetLocalMass__(material,fem_solver,elem)
            else:
                # massel = self.GetLocalMass(material,fem_solver,elem)
                massel = self.GetLocalMass(function_space[0], material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem)


        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = ApplyNeumannBoundaryConditions3D(MainData, mesh, elem, LagrangeElemCoords)

        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)
        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed is False:
            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem


    def GetMassMatrix(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerw, Eulerp):

        massel=[]
        # COMPUTE THE MASS MATRIX
        # if material.has_low_level_dispatcher:
        #     massel = self.__GetLocalMass__(material,fem_solver,elem)
        # else:
        #     massel = self.GetLocalMass(material,fem_solver,elem)
        massel = self.__GetLocalMass__(material,fem_solver,elem)

        I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_mass_elem, J_mass_elem, V_mass_elem


    def GetLocalStiffness(self, material, fem_solver, Eulerx, Eulerw, Eulers, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        # return self.K_uu(material, fem_solver, Eulerx, Eulerp, elem=0)
        if self.subtype=="lagrange_multiplier" or self.subtype=="augmented_lagrange":

            tractionforce = []
            k_uu, tu = self.K_uu(material, fem_solver, Eulerx, Eulerp, elem)
            k_uw = self.K_uw(material, fem_solver, Eulerx, Eulerp, elem)
            k_us = self.K_us(material, fem_solver, Eulerx, Eulerp, elem)


            # k_ww, tw = self.K_ww(material, fem_solver, Eulerw, Eulerp, elem)
            k_ww, tw = self.K_ww(material, fem_solver, Eulerx, Eulerp, elem)    # CHECK Eulerx vs Eulerw
            k_ws = self.K_ws(material, fem_solver, Eulerw, Eulerp, elem)

            k_ss, ts = self.K_ss(material, fem_solver, Eulerw, Eulerp, elem)

            if fem_solver.static_condensation is True:
                # IF STATIC CONDENSATION
                if self.subtype=="lagrange_multiplier":
                    inv_k_ws = inv(k_ws)
                    k0 = k_ww.dot(inv_k_ws)
                    k1 = k0.dot(k_us.T)
                    stiffness = k_uu + np.dot(np.dot(k_us,inv_k_ws),k1)

                    t0 = tw - np.dot(k0,ts)
                    tractionforce = tu - np.dot(np.dot(k_us,inv_k_ws),t0)

                    if self.save_condensed_matrices:
                        self.condensed_matrices['k_uu'][elem] = k_uu
                        self.condensed_matrices['k_us'][elem] = k_us
                        self.condensed_matrices['k_ww'][elem] = k_ww
                        self.condensed_matrices['k_ws'][elem] = k_ws
                        self.condensed_matrices['inv_k_ws'][elem] = inv_k_ws
                        self.condensed_vectors['tu'][elem] = tu
                        self.condensed_vectors['tw'][elem] = tw
                        self.condensed_vectors['ts'][elem] = ts

                elif self.subtype=="augmented_lagrange":
                    inv_k_ws = inv(k_ws)
                    k0 = k_ww.dot(inv_k_ws)
                    k1 = k0.dot(k_us.T)
                    k2 = inv(k_ws - k0.dot(k_ss))
                    stiffness = k_uu + np.dot(np.dot(k_us,k2),k1)

                    t0 = tw - np.dot(k0,ts)
                    tractionforce = tu - np.dot(np.dot(k_us,k2),t0)

                    if self.save_condensed_matrices:
                        self.condensed_matrices['k_uu'][elem] = k_uu
                        self.condensed_matrices['k_us'][elem] = k_us
                        self.condensed_matrices['k_ww'][elem] = k_ww
                        self.condensed_matrices['k_ws'][elem] = k_ws
                        self.condensed_matrices['k_ss'][elem] = k_ss
                        self.condensed_matrices['inv_k_ws'][elem] = inv_k_ws
                        self.condensed_vectors['tu'][elem] = tu
                        self.condensed_vectors['tw'][elem] = tw
                        self.condensed_vectors['ts'][elem] = ts

            else:
                # IF NO STATIC CONDENSATION
                raise NotImplementedError("Not implemented yet")
                k0 = np.concatenate((k_uu,k_uw, k_us),axis=1)
                k1 = np.concatenate((k_uw.T,k_ww, k_ws),axis=1)
                k2 = np.concatenate((k_us.T,k_ws.T, k_ss),axis=1)
                stiffness = np.concatenate((k0,k1, k2),axis=0)
                tractionforce = np.concatenate((tu,tw,ts))



        elif self.subtype=="penalty":

            tractionforce = []
            k_uu, tu = self.K_uu(material, fem_solver, Eulerx, Eulerp, elem)
            k_uu2, tu2 = self.K_uu_Penalty(material, fem_solver, Eulerx, Eulerp, elem)
            k_uw = material.kappa*self.K_us(material, fem_solver, Eulerx, Eulerp, elem)
            k_ww, tw = self.K_ww_Penalty(material, fem_solver, Eulerw, Eulerp, elem)

            if fem_solver.static_condensation is True:
                # IF STATIC CONDENSATION
                inv_k_ww = inv(k_ww)
                # stiffness = k_uu - np.dot(np.dot(k_uw,inv_k_ww),k_uw.T)
                stiffness = k_uu + k_uu2 - np.dot(np.dot(k_uw,inv_k_ww),k_uw.T)
                tractionforce = tu + tu2 - np.dot(np.dot(k_uw,inv_k_ww),tw)
            else:
                # IF NO STATIC CONDENSATION
                raise NotImplementedError("Not implemented yet")
                k0 = np.concatenate((k_uu+k_uu2,-k_uw),axis=1)
                stiffness = np.concatenate((-k_uw.T,-k_ww),axis=1)
                tractionforce = np.concatenate((tu,tw))


        else:
            raise ValueError("subtype of this variational formulation should be 'lagrange_multiplier' or 'penalty'")



        return stiffness, tractionforce



    def K_uu(self, material, fem_solver, Eulerx, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        mesh = self.meshes[0]
        function_spaces = self.function_spaces
        function_space = self.function_spaces[0]

        ndim = self.ndim
        nvar = self.nvar
        nodeperelem = meshes[0].elements.shape[1]
        # print nodeperelem

        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerELemCoords = Eulerx[mesh.elements[elem,:],:]

        Jm = function_spaces[0].Jm
        AllGauss = function_space.AllGauss


        # GET LOCAL KINEMATICS
        SpatialGradient, F, detJ = _KinematicMeasures_(Jm, AllGauss[:,0],
            LagrangeElemCoords, EulerELemCoords, fem_solver.requires_geometry_update)
        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        CauchyStressTensor, H_Voigt, _ = material.KineticMeasures(F,elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        stiffness, tractionforce = __ConstitutiveStiffnessIntegrandDF__(SpatialGradient,
            CauchyStressTensor,H_Voigt,detJ,self.nvar,fem_solver.requires_geometry_update)
        # # COMPUTE GEOMETRIC STIFFNESS
        # if fem_solver.requires_geometry_update:
        #     stiffness += self.__GeometricStiffnessIntegrand__(SpatialGradient,CauchyStressTensor,detJ)

        # SAVE AT THIS GAUSS POINT
        self.SpatialGradient = SpatialGradient
        self.detJ = detJ

        return stiffness, tractionforce

        # # ALLOCATE
        # stiffness = np.zeros((nodeperelem*nvar,nodeperelem*nvar),dtype=np.float64)
        # tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        # B = np.zeros((nodeperelem*nvar,material.elasticity_tensor_size),dtype=np.float64)

        # # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        # ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        # MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        # F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)

        # # COMPUTE REMAINING KINEMATIC MEASURES
        # StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # # UPDATE/NO-UPDATE GEOMETRY
        # if fem_solver.requires_geometry_update:
        #     # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        #     ParentGradientx = np.einsum('ijk,jl->kil',Jm,EulerELemCoords)
        #     # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
        #     SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
        #     # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        #     detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))
        # else:
        #     # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
        #     SpatialGradient = np.einsum('ikj',MaterialGradient)
        #     # COMPUTE ONCE detJ
        #     detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))


        # # LOOP OVER GAUSS POINTS
        # for counter in range(AllGauss.shape[0]):

        #     # COMPUTE THE HESSIAN AT THIS GAUSS POINT
        #     material.Hessian(StrainTensors,None, elem, counter)
        #     H_Voigt = material.elasticity_tensor

        #     # COMPUTE CAUCHY STRESS TENSOR
        #     CauchyStressTensor = []
        #     if fem_solver.requires_geometry_update:
        #         CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)


        #     # COMPUTE THE TANGENT STIFFNESS MATRIX
        #     BDB_1, t = self.K_uu_Integrand(B, SpatialGradient[counter,:,:],
        #         None, CauchyStressTensor, H_Voigt, analysis_nature=fem_solver.analysis_nature,
        #         has_prestress=fem_solver.has_prestress)

        #     # COMPUTE GEOMETRIC STIFFNESS MATRIX
        #     if fem_solver.requires_geometry_update:
        #         # BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
        #         # INTEGRATE TRACTION FORCE
        #         tractionforce += t*detJ[counter]

        #     # INTEGRATE STIFFNESS
        #     stiffness += BDB_1*detJ[counter]


        # # SAVE AT THIS GAUSS POINT
        # self.SpatialGradient = SpatialGradient
        # self.detJ = detJ

        # return stiffness, tractionforce


    def K_uw(self, material, fem_solver, Eulerx, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""
        return np.zeros((self.meshes[0].elements.shape[1]*self.ndim,self.meshes[1].elements.shape[1]*self.ndim),dtype=np.float64)


    def K_us(self, material, fem_solver, Eulerx, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        function_spaces = self.function_spaces

        Bases_s = function_spaces[2].Bases
        Ns = np.zeros((self.ndim,Bases_s.shape[0]*self.ndim),dtype=np.float64)
        Bu = np.zeros((self.meshes[0].elements.shape[1]*self.ndim,self.ndim),dtype=np.float64)
        stiffness = np.zeros((self.meshes[0].elements.shape[1]*self.ndim,self.meshes[2].elements.shape[1]*self.ndim))

        AllGauss = function_spaces[0].AllGauss
        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            Bu_Ns = self.K_us_Integrand(Bu, Ns, self.SpatialGradient[counter,:,:], Bases_s[:,counter])
            # INTEGRATE STIFFNESS
            stiffness += Bu_Ns*self.detJ[counter]


        return stiffness


    def K_ww(self, material, fem_solver, Eulerw, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        mesh = self.meshes[1]

        function_spaces = self.function_spaces
        function_space = self.function_spaces[1]

        ndim = self.ndim
        nvar = ndim
        nodeperelem = meshes[1].elements.shape[1]

        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerELemCoords = Eulerw[mesh.elements[elem,:],:]

        Jm = function_spaces[1].Jm
        AllGauss = function_space.AllGauss

        # # GET LOCAL KINEMATICS
        # SpatialGradient, F, detJ = _KinematicMeasures_(Jm, AllGauss[:,0],
        #     LagrangeElemCoords, EulerELemCoords, fem_solver.requires_geometry_update)
        # # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        # CauchyStressTensor, _, H_Voigt = material.KineticMeasures(F,elem=elem)


        # ALLOCATE
        stiffness = np.zeros((nodeperelem*nvar,nodeperelem*nvar),dtype=np.float64)
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        B = np.zeros((nodeperelem*nvar,material.gradient_elasticity_tensor_size),dtype=np.float64)

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # UPDATE/NO-UPDATE GEOMETRY
        if fem_solver.requires_geometry_update:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',Jm,EulerELemCoords)
            # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
            SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))
        else:
            # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
            SpatialGradient = np.einsum('ikj',MaterialGradient)
            # COMPUTE ONCE detJ
            detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))


        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            material.Hessian(StrainTensors,None, elem, counter)
            H_Voigt = material.gradient_elasticity_tensor

            # COMPUTE CAUCHY STRESS TENSOR
            CoupleStressVector = []
            if fem_solver.requires_geometry_update:
                CoupleStressVector = material.CoupleStress(StrainTensors,None,elem,counter).reshape(self.ndim,1)


            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.K_ww_Integrand(B, SpatialGradient[counter,:,:],
                None, CoupleStressVector, H_Voigt, analysis_nature=fem_solver.analysis_nature,
                has_prestress=fem_solver.has_prestress)

            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if fem_solver.requires_geometry_update:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]


        # # SAVE AT THIS GAUSS POINT
        # self.SpatialGradient = SpatialGradient
        # self.detJ = detJ

        return stiffness, tractionforce


    def K_ws(self, material, fem_solver, Eulerw, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        function_spaces = self.function_spaces

        Bases_w = function_spaces[1].Bases
        Bases_s = function_spaces[2].Bases
        Nw = np.zeros((Bases_w.shape[0]*self.ndim,self.ndim),dtype=np.float64)
        Ns = np.zeros((self.ndim,Bases_s.shape[0]*self.ndim),dtype=np.float64)
        stiffness = np.zeros((Bases_w.shape[0]*self.ndim,Bases_s.shape[0]*self.ndim))

        AllGauss = function_spaces[0].AllGauss
        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            Nw_Ns = self.K_ws_Integrand(Nw, Ns, Bases_w[:,counter], Bases_s[:,counter])
            # INTEGRATE STIFFNESS
            stiffness += Nw_Ns*self.detJ[counter]  ## CAREFUL ABOUT [CHECK] self.detJ[counter] ####################


        return -stiffness


    def K_ss(self, material, fem_solver, Eulers, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""
        stiffness = np.zeros((self.function_spaces[2].Bases.shape[0]*self.ndim,self.function_spaces[2].Bases.shape[0]*self.ndim),dtype=np.float64)
        tractionforce = np.zeros((self.function_spaces[2].Bases.shape[0]*self.ndim,1),dtype=np.float64)
        if self.subtype == "lagrange_multiplier":
            return stiffness, tractionforce

        EulerELemS = Eulers[self.meshes[2].elements[elem,:],:]
        Bases_s = self.function_spaces[2].Bases
        Ns = np.zeros((self.ndim,Bases_s.shape[0]*self.ndim),dtype=np.float64)
        AllGauss = self.function_spaces[2].AllGauss

        # FIND LAGRANGE MULTIPLIER AT ALL GAUSS POINTS
        EulerGaussS = np.dot(Bases_s.T,EulerELemS)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE STRESS
            LagrangeMultiplierStressVector = material.LagrangeMultiplierStress(EulerGaussS,elem=elem,gcounter=counter)
            # COMPUTE THE TANGENT STIFFNESS MATRIX
            NDN, t = self.K_ss_Integrand(Ns, Bases_s[:,counter], 0, LagrangeMultiplierStressVector, material.kappa,
                analysis_nature=fem_solver.analysis_nature, has_prestress=fem_solver.has_prestress)
            # INTEGRATE STIFFNESS
            stiffness += NDN*self.detJ[counter]  ## CAREFUL ABOUT [CHECK] self.detJ[counter] ####################
            # INTEGRAGE TRACTION
            if fem_solver.requires_geometry_update:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

        return stiffness, tractionforce



    def K_uu_Penalty(self, material, fem_solver, Eulerx, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        function_spaces = self.function_spaces

        Bu = np.zeros((self.meshes[0].elements.shape[1]*self.ndim,self.ndim),dtype=np.float64)
        stiffness = np.zeros((self.meshes[0].elements.shape[1]*self.ndim,self.meshes[0].elements.shape[1]*self.ndim))

        AllGauss = function_spaces[0].AllGauss
        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB = self.K_uu_Penalty_Integrand(Bu, self.SpatialGradient[counter,:,:])
            # INTEGRATE STIFFNESS
            stiffness += material.kappa*BDB*self.detJ[counter]


        # THIS CONTRIBUTES TO TRACTION AS WELL
        tractionforce = np.zeros((self.meshes[0].elements.shape[1]*self.ndim,1))
        return stiffness, tractionforce


    def K_ww_Penalty(self, material, fem_solver, Eulerw, Eulerp=None, elem=0):
        """Get stiffness matrix of the system"""

        meshes = self.meshes
        mesh = self.meshes[1]

        function_spaces = self.function_spaces
        function_space = self.function_spaces[1]

        ndim = self.ndim
        nvar = self.ndim
        nodeperelem = meshes[1].elements.shape[1]

        Jm = function_spaces[1].Jm
        AllGauss = function_space.AllGauss

        # ALLOCATE
        stiffness = np.zeros((nodeperelem*nvar,nodeperelem*nvar),dtype=np.float64)
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        Bases_w = self.function_spaces[1].Bases
        Nw = np.zeros((self.ndim,Bases_w.shape[0]*self.ndim),dtype=np.float64)
        # detJ = AllGauss[:,0]
        detJ = self.detJ

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE CAUCHY STRESS TENSOR
            CoupleStressVector = []
            if fem_solver.requires_geometry_update:
                CoupleStressVector = material.CoupleStress(StrainTensors,None,elem,counter).reshape(self.ndim,1)


            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.K_ww_Penalty_Integrand(Nw, Bases_w[:,counter],
                0, CoupleStressVector, material.kappa, analysis_nature=fem_solver.analysis_nature,
                has_prestress=fem_solver.has_prestress)

            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if fem_solver.requires_geometry_update:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += material.kappa*BDB_1*detJ[counter]


        return stiffness, tractionforce


    def GetLocalTraction(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""
        pass

    def K_uu_Integrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, H_Voigt, analysis_nature="nonlinear", has_prestress=True):

        ndim = self.ndim
        nvar = self.nvar

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T

        # THREE DIMENSIONS
        if SpatialGradient.shape[0]==3:

            B[0::nvar,0] = SpatialGradient[0,:]
            B[1::nvar,1] = SpatialGradient[1,:]
            B[2::nvar,2] = SpatialGradient[2,:]
            # Mechanical - Shear Terms
            B[1::nvar,5] = SpatialGradient[2,:]
            B[2::nvar,5] = SpatialGradient[1,:]

            B[0::nvar,4] = SpatialGradient[2,:]
            B[2::nvar,4] = SpatialGradient[0,:]

            B[0::nvar,3] = SpatialGradient[1,:]
            B[1::nvar,3] = SpatialGradient[0,:]

            if analysis_nature == 'nonlinear' or has_prestress:
                CauchyStressTensor_Voigt = np.array([
                    CauchyStressTensor[0,0],CauchyStressTensor[1,1],CauchyStressTensor[2,2],
                    CauchyStressTensor[0,1],CauchyStressTensor[0,2],CauchyStressTensor[1,2]
                    ]).reshape(6,1)

                TotalTraction = CauchyStressTensor_Voigt

        elif SpatialGradient.shape[0]==2:

            B[0::nvar,0] = SpatialGradient[0,:]
            B[1::nvar,1] = SpatialGradient[1,:]
            # Mechanical - Shear Terms
            B[0::nvar,2] = SpatialGradient[1,:]
            B[1::nvar,2] = SpatialGradient[0,:]

            if analysis_nature == 'nonlinear' or has_prestress:
                CauchyStressTensor_Voigt = np.array([
                    CauchyStressTensor[0,0],CauchyStressTensor[1,1],
                    CauchyStressTensor[0,1]]).reshape(3,1)

                TotalTraction = CauchyStressTensor

        BDB = np.dot(np.dot(B,H_Voigt),B.T)
        t=[]
        if analysis_nature == 'nonlinear' or has_prestress:
            t = np.dot(B,TotalTraction)

        return BDB, t


    def K_us_Integrand(self, Bu, Ns, SpatialGradient, Bases_s):

        ndim = self.ndim
        nvar = self.nvar

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T

        # THREE DIMENSIONS
        if SpatialGradient.shape[0]==3:

            # VORTICITY TERMS
            Bu[1::nvar,0] = -SpatialGradient[2,:]
            Bu[2::nvar,0] = SpatialGradient[1,:]

            Bu[0::nvar,1] = SpatialGradient[2,:]
            Bu[2::nvar,1] = -SpatialGradient[0,:]

            Bu[0::nvar,2] = -SpatialGradient[1,:]
            Bu[1::nvar,2] = SpatialGradient[0,:]

        elif SpatialGradient.shape[0]==2:
            # VORTICITY TERMS
            Bu[0::nvar,0] = -SpatialGradient[1,:]
            Bu[1::nvar,0] = SpatialGradient[0,:]

        for ivar in range(ndim):
            Ns[ivar,ivar::nvar] = Bases_s

        Bu_Ns = 0.5*np.dot(Bu,Ns)
        return Bu_Ns


    def K_ww_Integrand(self, B, SpatialGradient, ElectricDisplacementx,
        CoupleStressVector, H_Voigt, analysis_nature="nonlinear", has_prestress=True):

        ndim = self.ndim
        nvar = self.nvar

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T

        # THREE DIMENSIONS
        if SpatialGradient.shape[0]==3:

            # VORTICITY TERMS
            B[1::nvar,0] = -SpatialGradient[2,:]
            B[2::nvar,0] = SpatialGradient[1,:]

            B[0::nvar,1] = SpatialGradient[2,:]
            B[2::nvar,1] = -SpatialGradient[0,:]

            B[0::nvar,2] = -SpatialGradient[1,:]
            B[1::nvar,2] = SpatialGradient[0,:]

        elif SpatialGradient.shape[0]==2:
            # VORTICITY TERMS
            B[0::nvar,0] = -SpatialGradient[1,:]
            B[1::nvar,0] = SpatialGradient[0,:]


        BDB = np.dot(np.dot(B,H_Voigt),B.T)
        t=[]
        if analysis_nature == 'nonlinear' or has_prestress:
            t = np.dot(B,CoupleStressVector)
        return BDB, t


    def K_ws_Integrand(self, Nw, Ns, Bases_w, Bases_s):

        ndim = self.ndim
        nvar = self.nvar

        for ivar in range(ndim):
            Nw[ivar::nvar,ivar] = Bases_w

        for ivar in range(ndim):
            Ns[ivar,ivar::nvar] = Bases_s

        Nw_Ns = 0.5*np.dot(Nw,Ns)
        return Nw_Ns


    def K_ss_Integrand(self, Ns, Bases_s, ElectricDisplacementx,
        LagrangeMultiplierStressVector, kappa, analysis_nature="nonlinear", has_prestress=True):

        ndim = self.ndim
        nvar = self.nvar

        for ivar in range(ndim):
            Ns[ivar,ivar::nvar] = Bases_s

        if self.subtype == "augmented_lagrange":
            NDN = np.dot(Ns.T,Ns)/(1.0*kappa)
        else:
            NDN = np.zeros((self.function_spaces[2].Bases.shape[0]*self.ndim,self.function_spaces[2].Bases.shape[0]*self.ndim),dtype=np.float64)
        t=[]
        if analysis_nature == 'nonlinear' or has_prestress:
            t = np.dot(Ns,LagrangeMultiplierStressVector)
        return NDN, t


    def K_uu_Penalty_Integrand(self, Bu, SpatialGradient):

        ndim = self.ndim
        nvar = self.nvar

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T

        # THREE DIMENSIONS
        if SpatialGradient.shape[0]==3:

            # VORTICITY TERMS
            Bu[1::nvar,0] = -SpatialGradient[2,:]
            Bu[2::nvar,0] = SpatialGradient[1,:]

            Bu[0::nvar,1] = SpatialGradient[2,:]
            Bu[2::nvar,1] = -SpatialGradient[0,:]

            Bu[0::nvar,2] = -SpatialGradient[1,:]
            Bu[1::nvar,2] = SpatialGradient[0,:]

        elif SpatialGradient.shape[0]==2:
            # VORTICITY TERMS
            Bu[0::nvar,0] = -SpatialGradient[1,:]
            Bu[1::nvar,0] = SpatialGradient[0,:]


        BDB = 0.25*np.dot(Bu,Bu.T)
        return BDB


    def K_ww_Penalty_Integrand(self, Nw, Bases_w, ElectricDisplacementx,
        CoupleStressVector, kappa, analysis_nature="nonlinear", has_prestress=True):

        ndim = self.ndim
        nvar = self.nvar

        for ivar in range(ndim):
            Nw[ivar,ivar::nvar] = Bases_w

        NDN = kappa*np.dot(Nw.T,Nw)
        t=[]
        if analysis_nature == 'nonlinear' or has_prestress:
            t = np.dot(Nw,CoupleStressVector)
        return NDN, t



    def TractionIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, analysis_nature="nonlinear", has_prestress=True):
        """Applies to displacement potential based formulation"""
        pass



    def GetEnergy(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get virtual energy of the system. For dynamic analysis this is handy for computing conservation of energy.
            The routine computes the global form of virtual internal energy i.e. integral of "W(C,G,C)"". This can be
            computed purely in a Lagrangian configuration.
        """

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
        F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
        SpatialGradient = np.einsum('ikj',MaterialGradient)
        # COMPUTE ONCE detJ
        detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))

        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):
            if material.energy_type == "enthalpy":
                # COMPUTE THE INTERNAL ENERGY AT THIS GAUSS POINT
                energy = material.InternalEnergy(StrainTensors,ElectricFieldx[counter,:],0,elem,counter)
            elif material.energy_type == "internal_energy":
                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)
                # COMPUTE THE INTERNAL ENERGY AT THIS GAUSS POINT
                energy = material.InternalEnergy(StrainTensors,ElectricDisplacementx[counter,:],elem,counter)

            # INTEGRATE INTERNAL ENERGY
            internal_energy += energy*detJ[counter]

        return internal_energy



    def Assemble(self, fem_solver, material, Eulerx, Eulerw, Eulers, Eulerp):

        # GET MESH DETAILS
        # C = mesh.InferPolynomialDegree() - 1
        formulation = self
        meshes = formulation.meshes
        mesh = meshes[0]
        nvar = formulation.nvar
        ndim = formulation.ndim
        nelem = meshes[0].nelem
        nodeperelem = meshes[0].elements.shape[1]
        local_size = int(ndim*meshes[0].elements.shape[1] + ndim*meshes[1].elements.shape[1] + ndim*meshes[2].elements.shape[1])
        capacity = local_size**2

        # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF STIFFNESS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
        I_stiffness=np.zeros(int(capacity*nelem),dtype=np.int32)
        J_stiffness=np.zeros(int(capacity*nelem),dtype=np.int32)
        V_stiffness=np.zeros(int(capacity*nelem),dtype=np.float64)

        I_mass=[]; J_mass=[]; V_mass=[]
        if fem_solver.analysis_type !='static':
            # ALLOCATE VECTORS FOR SPARSE ASSEMBLY OF MASS MATRIX - CHANGE TYPES TO INT64 FOR DoF > 1e09
            I_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            J_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.int32)
            V_mass=np.zeros(int((nvar*nodeperelem)**2*nelem),dtype=np.float64)

        # T = np.zeros((local_size,1),np.float64)
        T = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)

        mass, F = [], []
        if fem_solver.has_moving_boundary:
            F = np.zeros((mesh.points.shape[0]*nvar,1),np.float64)


        if fem_solver.parallel:
            # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
            # ParallelTuple = parmap.map(formulation.GetElementalMatrices,np.arange(0,nelem,dtype=np.int32),
                # function_space, mesh, material, fem_solver, Eulerx, Eulerp)

            ParallelTuple = parmap.map(formulation,np.arange(0,nelem,dtype=np.int32),
                function_space, mesh, material, fem_solver, Eulerx, Eulerp, processes= int(multiprocessing.cpu_count()/2))


        for elem in range(nelem):

            if fem_solver.parallel:
                # UNPACK PARALLEL TUPLE VALUES
                I_stiff_elem = ParallelTuple[elem][0]; J_stiff_elem = ParallelTuple[elem][1]; V_stiff_elem = ParallelTuple[elem][2]
                t = ParallelTuple[elem][3]; f = ParallelTuple[elem][4]
                I_mass_elem = ParallelTuple[elem][5]; J_mass_elem = ParallelTuple[elem][6]; V_mass_elem = ParallelTuple[elem][6]

            else:
                # COMPUATE ALL LOCAL ELEMENTAL MATRICES (STIFFNESS, MASS, INTERNAL & EXTERNAL TRACTION FORCES )
                I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, \
                I_mass_elem, J_mass_elem, V_mass_elem = formulation.GetElementalMatrices(elem,
                    formulation.function_spaces, formulation.meshes, material, fem_solver, Eulerx, Eulerw, Eulers, Eulerp)

            # SPARSE ASSEMBLY - STIFFNESS MATRIX
            SparseAssemblyNative(I_stiff_elem,J_stiff_elem,V_stiff_elem,I_stiffness,J_stiffness,V_stiffness,
                elem,nvar,nodeperelem,mesh.elements)

            if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
                # SPARSE ASSEMBLY - MASS MATRIX
                SparseAssemblyNative(I_mass_elem,J_mass_elem,V_mass_elem,I_mass,J_mass,V_mass,
                    elem,nvar,nodeperelem,mesh.elements)

            if fem_solver.has_moving_boundary:
                # RHS ASSEMBLY
                RHSAssemblyNative(F,f,elem,nvar,nodeperelem,mesh.elements)

            # INTERNAL TRACTION FORCE ASSEMBLY
            RHSAssemblyNative(T,t,elem,nvar,nodeperelem,mesh.elements)

            if (elem % fem_solver.assembly_print_counter == 0 or elem==nelem-1) and elem != 0:
                nume = elem+1 if elem==nelem-1 else elem
                print(('Assembled {} element matrices').format(nume))


        if fem_solver.parallel:
            del ParallelTuple
            gc.collect()


        stiffness = coo_matrix((V_stiffness,(I_stiffness,J_stiffness)),
            shape=((nvar*mesh.points.shape[0],nvar*mesh.points.shape[0])),dtype=np.float64).tocsr()

        # GET STORAGE/MEMORY DETAILS
        fem_solver.spmat = stiffness.data.nbytes/1024./1024.
        fem_solver.ijv = (I_stiffness.nbytes + J_stiffness.nbytes + V_stiffness.nbytes)/1024./1024.

        del I_stiffness, J_stiffness, V_stiffness
        gc.collect()

        if fem_solver.analysis_type != 'static' and fem_solver.is_mass_computed==False:
            mass = csr_matrix((V_mass,(I_mass,J_mass)),shape=((nvar*mesh.points.shape[0],
                nvar*mesh.points.shape[0])),dtype=np.float64)

            fem_solver.is_mass_computed = True

        return stiffness, T, F, mass



    def GetAugmentedSolution(self, fem_solver, material, TotalDisp, Eulerx, Eulerw, Eulers, Eulerp):
        """Get condensed variables
        """

        if self.save_condensed_matrices is False:
            return 0., 0.

        mesh = self.meshes[0]
        elements = mesh.elements
        points = mesh.points
        nelem = mesh.nelem
        nodeperelem = mesh.elements.shape[1]

        C = mesh.InferPolynomialDegree() - 1
        ndim = mesh.InferSpatialDimension()

        function_space = FunctionSpace(mesh, p=C+1, evaluate_at_nodes=True)

        Jm = function_space.Jm
        AllGauss = function_space.AllGauss

        AllEulerW = np.zeros((nelem,self.meshes[1].elements.shape[1],ndim))
        AllEulerS = np.zeros((nelem,self.meshes[2].elements.shape[1],ndim))

        NodalEulerW = np.zeros((self.meshes[1].points.shape[0],self.ndim))
        NodalEulerS = np.zeros((self.meshes[2].points.shape[0],self.ndim))

        # LOOP OVER ELEMENTS
        for elem in range(nelem):
            # GET THE FIELDS AT THE ELEMENT LEVEL
            LagrangeElemCoords = points[elements[elem,:],:]
            EulerELemCoords = Eulerx[elements[elem,:],:]

            if self.subtype == "lagrange_multiplier":
                k_uu = self.condensed_matrices['k_uu'][elem]
                k_us = self.condensed_matrices['k_us'][elem]
                k_ww = self.condensed_matrices['k_ww'][elem]
                k_ws = self.condensed_matrices['k_ws'][elem]
                inv_k_ws = self.condensed_matrices['inv_k_ws'][elem]
                tu = self.condensed_vectors['tu'][elem]
                tw = self.condensed_vectors['tw'][elem]
                ts = self.condensed_vectors['ts'][elem]

                EulerElemW = np.dot(inv_k_ws,(ts - np.dot(k_us.T,EulerELemCoords.ravel())[:,None])).ravel()
                EulerElemS = np.dot(inv_k_ws,(tw - np.dot(k_ww,EulerElemW)[:,None])).ravel()

                # SAVE
                AllEulerW[elem,:,:] = EulerElemW.reshape(self.meshes[1].elements.shape[1],ndim)
                AllEulerS[elem,:,:] = EulerElemW.reshape(self.meshes[2].elements.shape[1],ndim)


        for inode in self.all_nodes:
            Els, Pos = self.Elss[inode], self.Poss[inode]
            ncommon_nodes = Els.shape[0]
            for uelem in range(ncommon_nodes):
                NodalEulerW += AllEulerW[Els[uelem],Pos[uelem],:]
                NodalEulerS += AllEulerS[Els[uelem],Pos[uelem],:]
            # AVERAGE OUT
            NodalEulerW[inode,:] /= ncommon_nodes
            NodalEulerS[inode,:] /= ncommon_nodes

        # NAKE SURE TO UPDATE THESE INSTEAD OF CREATING THEM IN WHICH CASE YOU HAVE TO RETURN THEM
        Eulerw[:,:] += NodalEulerW
        Eulers[:,:] += NodalEulerS


        return NodalEulerW, NodalEulerS

