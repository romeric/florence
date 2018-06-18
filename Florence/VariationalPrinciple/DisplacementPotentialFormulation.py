import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from .DisplacementPotentialApproachIndices import *
from ._ConstitutiveStiffnessDPF_ import __ConstitutiveStiffnessIntegrandDPF__
from ._TractionDPF_ import __TractionIntegrandDPF__
from Florence.Tensor import issymetric
from Florence.LegendreTransform import LegendreTransform


__all__ = ["DisplacementPotentialFormulation"]

class DisplacementPotentialFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False):

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(DisplacementPotentialFormulation, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces,
            compute_post_quadrature=compute_post_quadrature)

        self.fields = "electro_mechanics"
        self.nvar = self.ndim+1

        C = mesh.InferPolynomialDegree() - 1
        mesh.InferBoundaryElementType()

        if quadrature_rules == None and self.quadrature_rules == None:

            # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
            optimal_quadrature = 3
            # is_flattened = True
            is_flattened = False

            if mesh.element_type == "tri" or mesh.element_type == "tet":
                norder = 2*C
                # TAKE CARE OF C=0 CASE
                if norder == 0:
                    norder = 1

                norder_post = 2*(C+1)
            else:
                norder = C+2
                norder_post = 2*(C+2)

            # GET QUADRATURE
            quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder, mesh_type=mesh.element_type, is_flattened=is_flattened)
            if self.compute_post_quadrature:
                # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
                post_quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder_post, mesh_type=mesh.element_type)
            else:
                post_quadrature = None

            # BOUNDARY QUADRATURE
            bquadrature = QuadratureRule(optimal=optimal_quadrature, norder=C+2, mesh_type=mesh.boundary_element_type, is_flattened=is_flattened)

            self.quadrature_rules = (quadrature,post_quadrature,bquadrature)
        else:
            self.quadrature_rules = quadrature_rules

        if function_spaces == None and self.function_spaces == None:

            # CREATE FUNCTIONAL SPACES
            function_space = FunctionSpace(mesh, quadrature, p=C+1, equally_spaced=equally_spaced_bases, use_optimal_quadrature=is_flattened)
            if compute_post_quadrature:
                post_function_space = FunctionSpace(mesh, post_quadrature, p=C+1, equally_spaced=equally_spaced_bases)
            else:
                post_function_space = None

            # CREATE BOUNDARY FUNCTIONAL SPACES
            bfunction_space = FunctionSpace(mesh.CreateDummyLowerDimensionalMesh(),
                bquadrature, p=C+1, equally_spaced=equally_spaced_bases, use_optimal_quadrature=is_flattened)

            self.function_spaces = (function_space,post_function_space,bfunction_space)
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


    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerp):

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
        ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]

        # COMPUTE THE STIFFNESS MATRIX
        if material.has_low_level_dispatcher:
            stiffnessel, t = self.__GetLocalStiffness__(function_space, material, LagrangeElemCoords,
                EulerElemCoords, ElectricPotentialElem, fem_solver, elem)
        else:
            stiffnessel, t = self.GetLocalStiffness(function_space, material, LagrangeElemCoords,
                EulerElemCoords, ElectricPotentialElem, fem_solver, elem)

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


    def GetElementalMatricesInVectorForm(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerp):

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]
        ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]

        # COMPUTE THE TRACTION VECTOR
        if material.has_low_level_dispatcher:
            t = self.__GetLocalTraction__(function_space, material, LagrangeElemCoords,
                EulerElemCoords, ElectricPotentialElem, fem_solver, elem)
        else:
            t = self.GetLocalTraction(function_space, material, LagrangeElemCoords,
                EulerElemCoords, ElectricPotentialElem, fem_solver, elem)

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



    def GetLocalStiffness(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
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

        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            if material.energy_type == "enthalpy":
                # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                H_Voigt = material.Hessian(StrainTensors,ElectricFieldx[counter,:], elem, counter)

                # COMPUTE ELECTRIC DISPLACEMENT
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = []
                if fem_solver.requires_geometry_update:
                    CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricFieldx[counter,:],elem,counter)

            elif material.energy_type == "internal_energy":
                # THIS REQUIRES LEGENDRE TRANSFORM

                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

                # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                H_Voigt = material.Hessian(StrainTensors,ElectricDisplacementx, elem, counter)

                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = []
                if fem_solver.requires_geometry_update:
                    CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricDisplacementx,elem,counter)


            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                ElectricDisplacementx, CauchyStressTensor, H_Voigt, requires_geometry_update=fem_solver.requires_geometry_update)

            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if material.nature != "linear":
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]


        return stiffness, tractionforce



    def __GetLocalStiffness__(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get stiffness matrix of the system"""

        # GET LOCAL KINEMATICS
        SpatialGradient, F, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0], LagrangeElemCoords,
            EulerELemCoords, fem_solver.requires_geometry_update)
        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)
        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        ElectricDisplacementx, CauchyStressTensor, H_Voigt = material.KineticMeasures(F, ElectricFieldx, elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        stiffness, tractionforce = __ConstitutiveStiffnessIntegrandDPF__(SpatialGradient,ElectricDisplacementx,
            CauchyStressTensor,H_Voigt,detJ,self.nvar,fem_solver.requires_geometry_update)
        # COMPUTE LOCAL GEOMETRIC STIFFNESS
        if material.nature != "linear":
            stiffness += self.__GeometricStiffnessIntegrand__(SpatialGradient,CauchyStressTensor,detJ)

        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, H_Voigt, requires_geometry_update=True):
        """Overrides base for electric potential formulation"""

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T.copy()
        ElectricDisplacementx = ElectricDisplacementx.flatten().copy()

        FillConstitutiveB(B,SpatialGradient,self.ndim,self.nvar)
        BDB = B.dot(H_Voigt.dot(B.T))

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = GetTotalTraction(CauchyStressTensor,ElectricDisplacementx)
            t = np.dot(B,TotalTraction)

        return BDB, t


    def GetLocalTraction(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""

        nvar = self.nvar
        ndim = self.ndim
        nodeperelem = function_space.Bases.shape[0]

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = function_space.Jm
        AllGauss = function_space.AllGauss

        # ALLOCATE
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        B = np.zeros((nodeperelem*nvar,material.H_VoigtSize),dtype=np.float64)

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

        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            if material.energy_type == "enthalpy":
                # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                H_Voigt = material.Hessian(StrainTensors,ElectricFieldx[counter,:], elem, counter)

                # COMPUTE ELECTRIC DISPLACEMENT
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = []
                if fem_solver.requires_geometry_update:
                    CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricFieldx[counter,:],elem,counter)

            elif material.energy_type == "internal_energy":
                # THIS REQUIRES LEGENDRE TRANSFORM

                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

                # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                H_Voigt = material.Hessian(StrainTensors,ElectricDisplacementx, elem, counter)

                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = []
                if fem_solver.requires_geometry_update:
                    CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricDisplacementx,elem,counter)


            # COMPUTE THE TANGENT STIFFNESS MATRIX
            t = self.TractionIntegrand(B, SpatialGradient[counter,:,:],
                ElectricDisplacementx, CauchyStressTensor, requires_geometry_update=fem_solver.requires_geometry_update)

            if fem_solver.requires_geometry_update:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]


        return tractionforce


    def __GetLocalTraction__(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""

        # GET LOCAL KINEMATICS
        SpatialGradient, F, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0], LagrangeElemCoords,
            EulerELemCoords, fem_solver.requires_geometry_update)
        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)
        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        ElectricDisplacementx, CauchyStressTensor, _ = material.KineticMeasures(F, ElectricFieldx, elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        tractionforce = __TractionIntegrandDPF__(SpatialGradient,ElectricDisplacementx,
            CauchyStressTensor,detJ,material.H_VoigtSize,self.nvar,fem_solver.requires_geometry_update)

        return tractionforce


    def TractionIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, requires_geometry_update=True):
        """Applies to displacement potential based formulation"""

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T.copy()
        ElectricDisplacementx = ElectricDisplacementx.flatten().copy()

        FillConstitutiveB(B,SpatialGradient,self.ndim,self.nvar)

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = GetTotalTraction(CauchyStressTensor,ElectricDisplacementx)
            t = np.dot(B,TotalTraction)

        return t


    def GetLocalResidual(self):
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
                energy = material.InternalEnergy(StrainTensors,ElectricFieldx[counter,:],elem,counter)
            elif material.energy_type == "internal_energy":
                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)
                # COMPUTE THE INTERNAL ENERGY AT THIS GAUSS POINT
                energy = material.InternalEnergy(StrainTensors,ElectricDisplacementx[counter,:],elem,counter)

            # INTEGRATE INTERNAL ENERGY
            internal_energy += energy*detJ[counter]

        return internal_energy



    def GetLinearMomentum(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, VelocityElem, ElectricPotentialElem, fem_solver, elem=0):
        """Get linear momentum or virtual power of the system. For dynamic analysis this is handy for computing conservation of linear momentum.
            The routine computes the global form of virtual power i.e. integral of "P:Grad_0(V)"" where P is first Piola-Kirchhoff
            stress tensor and Grad_0(V) is the material gradient of velocity. Alternatively in update Lagrangian format this could be
            computed using "Sigma: Grad(V) J" where Sigma is the Cauchy stress tensor and Grad(V) is the spatial gradient of velocity.
            The latter approach is followed here
        """

        nvar = self.nvar
        ndim = self.ndim
        nodeperelem = function_space.Bases.shape[0]

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = function_space.Jm
        AllGauss = function_space.AllGauss

        internal_power = 0.

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)
        # TIME DERIVATIVE OF F
        Fdot = np.einsum('ij,kli->kjl', VelocityElem, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientx = np.einsum('ijk,jl->kil',Jm, EulerELemCoords)
        # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
        SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
        # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))

        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):
            GradV = np.dot(Fdot[counter,:,:],np.linalg.inv(F[counter,:,:]))

            if material.energy_type == "enthalpy":
                # COMPUTE ELECTRIC DISPLACEMENT
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)
                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricFieldx[counter,:],elem,counter)
            elif material.energy_type == "internal_energy":
                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)
                # COMPUTE CAUCHY STRESS TENSOR
                CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricDisplacementx,elem,counter)

            # INTEGRATE INTERNAL VIRTUAL POWER
            internal_power += np.einsum('ij,ij',CauchyStressTensor,GradV)*detJ[counter]

        return internal_power



    # ##############################################################################################
    # ##############################################################################################

    # def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
    #     CauchyStressTensor, H_Voigt, requires_geometry_update=True):

    #     ndim = self.ndim
    #     nvar = self.nvar

    #     # MATRIX FORM
    #     SpatialGradient = SpatialGradient.T


    #     # THREE DIMENSIONS
    #     if SpatialGradient.shape[0]==3:

    #         B[0::nvar,0] = SpatialGradient[0,:]
    #         B[1::nvar,1] = SpatialGradient[1,:]
    #         B[2::nvar,2] = SpatialGradient[2,:]
    #         # Mechanical - Shear Terms
    #         B[1::nvar,5] = SpatialGradient[2,:]
    #         B[2::nvar,5] = SpatialGradient[1,:]

    #         B[0::nvar,4] = SpatialGradient[2,:]
    #         B[2::nvar,4] = SpatialGradient[0,:]

    #         B[0::nvar,3] = SpatialGradient[1,:]
    #         B[1::nvar,3] = SpatialGradient[0,:]

    #         # Electrostatic
    #         B[3::nvar,6] = SpatialGradient[0,:]
    #         B[3::nvar,7] = SpatialGradient[1,:]
    #         B[3::nvar,8] = SpatialGradient[2,:]

    #         if requires_geometry_update:
    #             CauchyStressTensor_Voigt = np.array([
    #                 CauchyStressTensor[0,0],CauchyStressTensor[1,1],CauchyStressTensor[2,2],
    #                 CauchyStressTensor[0,1],CauchyStressTensor[0,2],CauchyStressTensor[1,2]
    #                 ]).reshape(6,1)

    #             # TotalTraction = np.concatenate((CauchyStressTensor_Voigt,ElectricDisplacementx[:,None]),axis=0)
    #             TotalTraction = np.concatenate((CauchyStressTensor_Voigt,ElectricDisplacementx),axis=0)

    #     elif SpatialGradient.shape[0]==2:

    #         B[0::nvar,0] = SpatialGradient[0,:]
    #         B[1::nvar,1] = SpatialGradient[1,:]
    #         # Mechanical - Shear Terms
    #         B[0::nvar,2] = SpatialGradient[1,:]
    #         B[1::nvar,2] = SpatialGradient[0,:]

    #         # Electrostatic
    #         B[2::nvar,3] = SpatialGradient[0,:]
    #         B[2::nvar,4] = SpatialGradient[1,:]

    #         if requires_geometry_update:
    #             CauchyStressTensor_Voigt = np.array([
    #                 CauchyStressTensor[0,0],CauchyStressTensor[1,1],
    #                 CauchyStressTensor[0,1]]).reshape(3,1)

    #             TotalTraction = np.concatenate((CauchyStressTensor_Voigt,ElectricDisplacementx[:,None]),axis=0)

    #     BDB = np.dot(np.dot(B,H_Voigt),B.T)
    #     t=[]
    #     if requires_geometry_update:
    #         t = np.dot(B,TotalTraction)

    #     return BDB, t


    # def GeometricStiffnessIntegrand(self,SpatialGradient,CauchyStressTensor):

    #     ndim = self.ndim
    #     nvar = self.nvar

    #     B = np.zeros((nvar*SpatialGradient.shape[0],ndim*ndim))
    #     SpatialGradient = SpatialGradient.T

    #     S = 0
    #     if SpatialGradient.shape[0]==3:

    #         B[0::nvar,0] = SpatialGradient[0,:]
    #         B[0::nvar,1] = SpatialGradient[1,:]
    #         B[0::nvar,2] = SpatialGradient[2,:]

    #         B[1::nvar,3] = SpatialGradient[0,:]
    #         B[1::nvar,4] = SpatialGradient[1,:]
    #         B[1::nvar,5] = SpatialGradient[2,:]

    #         B[2::nvar,6] = SpatialGradient[0,:]
    #         B[2::nvar,7] = SpatialGradient[1,:]
    #         B[2::nvar,8] = SpatialGradient[2,:]

    #         S = np.zeros((3*ndim,3*ndim))
    #         S[0:ndim,0:ndim] = CauchyStressTensor
    #         S[ndim:2*ndim,ndim:2*ndim] = CauchyStressTensor
    #         S[2*ndim:,2*ndim:] = CauchyStressTensor

    #     elif SpatialGradient.shape[0]==2:

    #         B[0::nvar,0] = SpatialGradient[0,:]
    #         B[0::nvar,1] = SpatialGradient[1,:]

    #         B[1::nvar,2] = SpatialGradient[0,:]
    #         B[1::nvar,3] = SpatialGradient[1,:]

    #         # S = np.zeros((3*ndim,3*ndim))
    #         S = np.zeros((ndim*ndim,ndim*ndim))
    #         S[0:ndim,0:ndim] = CauchyStressTensor
    #         S[ndim:2*ndim,ndim:2*ndim] = CauchyStressTensor
    #         # S[2*ndim:,2*ndim:] = CauchyStressTensor


    #     BDB = np.dot(np.dot(B,S),B.T)

    #     return BDB
