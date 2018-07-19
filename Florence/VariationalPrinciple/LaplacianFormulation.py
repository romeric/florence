import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from ._ConstitutiveStiffnessLaplacian_ import __ConstitutiveStiffnessIntegrandLaplacian__
from Florence.Tensor import issymetric
from Florence.LegendreTransform import LegendreTransform

__all__ = ["LaplacianFormulation"]

class LaplacianFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False, quadrature_degree=None):

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(LaplacianFormulation, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces,
            compute_post_quadrature=compute_post_quadrature)

        self.fields = "electrostatics"
        self.nvar = 1

        self.GetQuadraturesAndFunctionSpaces(mesh, variables_order=variables_order,
            quadrature_rules=quadrature_rules, quadrature_type=quadrature_type,
            function_spaces=function_spaces, compute_post_quadrature=compute_post_quadrature,
            equally_spaced_bases=equally_spaced_bases, quadrature_degree=quadrature_degree)


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


        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = ApplyNeumannBoundaryConditions3D(MainData, mesh, elem, LagrangeElemCoords)

        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)

        I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
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

        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = self.ApplyNeumannBoundaryConditions3D(fem_solver, mesh, elem, LagrangeElemCoords, ElectricPotentialElem)

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

            elif material.energy_type == "internal_energy":
                # THIS REQUIRES LEGENDRE TRANSFORM

                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

                # COMPUTE THE HESSIAN AT THIS GAUSS POINT
                H_Voigt = material.Hessian(StrainTensors,ElectricDisplacementx, elem, counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                ElectricDisplacementx, H_Voigt, analysis_nature=fem_solver.analysis_nature,
                has_prestress=fem_solver.has_prestress)

            # INTEGRATE TRACTION FORCE
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
        ElectricDisplacementx, _, H_Voigt = material.KineticMeasures(F, ElectricFieldx, elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        stiffness, tractionforce = __ConstitutiveStiffnessIntegrandLaplacian__(SpatialGradient,ElectricDisplacementx,
            H_Voigt,detJ,self.nvar,fem_solver.requires_geometry_update)

        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
            H_Voigt, analysis_nature="nonlinear", has_prestress=True):

            ndim = self.ndim
            nvar = self.nvar

            # MATRIX FORM
            SpatialGradient = SpatialGradient.T
            TotalTraction = ElectricDisplacementx.reshape(ElectricDisplacementx.shape[0],1)

            # THREE DIMENSIONS
            if SpatialGradient.shape[0]==3:
                # Electrostatic
                B[::nvar,0] = SpatialGradient[0,:]
                B[::nvar,1] = SpatialGradient[1,:]
                B[::nvar,2] = SpatialGradient[2,:]

            elif SpatialGradient.shape[0]==2:
                # Electrostatic
                B[::nvar,0] = SpatialGradient[0,:]
                B[::nvar,1] = SpatialGradient[1,:]

            BDB = np.dot(np.dot(B,H_Voigt),B.T)
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

                # COMPUTE ELECTRIC DISPLACEMENT
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

            elif material.energy_type == "internal_energy":
                # THIS REQUIRES LEGENDRE TRANSFORM

                # COMPUTE ELECTRIC DISPLACEMENT IMPLICITLY
                ElectricDisplacementx = material.ElectricDisplacementx(StrainTensors, ElectricFieldx[counter,:], elem, counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            t = self.TractionIntegrand(B, SpatialGradient[counter,:,:],
                ElectricDisplacementx, analysis_nature=fem_solver.analysis_nature,
                has_prestress=fem_solver.has_prestress)

            # INTEGRATE TRACTION FORCE
            tractionforce += t*detJ[counter]

        return tractionforce

    def __GetLocalTraction__(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""
        pass

    def TractionIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        analysis_nature="nonlinear", has_prestress=True):
        """Applies to displacement potential based formulation"""

        ndim = self.ndim
        nvar = self.nvar

        # MATRIX FORM
        SpatialGradient = SpatialGradient.T
        TotalTraction = ElectricDisplacementx.reshape(ElectricDisplacementx.shape[0],1)

        # THREE DIMENSIONS
        if SpatialGradient.shape[0]==3:
            # Electrostatic
            B[::nvar,0] = SpatialGradient[0,:]
            B[::nvar,1] = SpatialGradient[1,:]
            B[::nvar,2] = SpatialGradient[2,:]

        elif SpatialGradient.shape[0]==2:
            # Electrostatic
            B[::nvar,0] = SpatialGradient[0,:]
            B[::nvar,1] = SpatialGradient[1,:]

        t = np.dot(B,TotalTraction)

        return t


    def GetLocalResidual(self):
        pass