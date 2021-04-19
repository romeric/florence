import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from .DisplacementApproachIndices import *
from ._ConstitutiveStiffnessDF_ import __ConstitutiveStiffnessIntegrandDF__
from ._TractionDF_ import __TractionIntegrandDF__
from Florence.Tensor import issymetric, makezero

__all__ = ["DisplacementFormulation"]

class DisplacementFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False, quadrature_degree=None):

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(DisplacementFormulation, self).__init__(mesh,variables_order=self.variables_order,
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
        if material.has_low_level_dispatcher:
            stiffnessel, t = self.__GetLocalStiffness__(function_space,material,
                LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
        else:
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
        if material.has_low_level_dispatcher:
            t = self.__GetLocalTraction__(function_space,material,
            # t = self.GetLocalTraction(function_space,material,
                LagrangeElemCoords,EulerElemCoords,fem_solver,elem)
        else:
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
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # UPDATE/NO-UPDATE GEOMETRY
        if fem_solver.requires_geometry_update:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil', Jm, EulerElemCoords)
            # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
            SpatialGradient = np.einsum('ijk,kli->ilj', inv(ParentGradientx),Jm)
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            # detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))
            detJ = np.einsum('i,i,i->i',AllGauss[:,0], det(ParentGradientX), StrainTensors['J'])
        else:
            # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
            SpatialGradient = np.einsum('ikj',MaterialGradient)
            # COMPUTE ONCE detJ
            # detJ = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))
            detJ = np.einsum('i,i->i',AllGauss[:,0], det(ParentGradientX))


        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = material.Hessian(StrainTensors, None, elem, counter)

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                CauchyStressTensor, H_Voigt, requires_geometry_update=fem_solver.requires_geometry_update)

            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if material.nature != "linear":
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
            # INTEGRATE TRACTION FORCE
            if fem_solver.requires_geometry_update:
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        makezero(stiffness, 1e-12)
        return stiffness, tractionforce



    def __GetLocalStiffness__(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get stiffness matrix of the system"""

        # GET LOCAL KINEMATICS
        SpatialGradient, F, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0],
            LagrangeElemCoords, EulerElemCoords, fem_solver.requires_geometry_update)
        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        CauchyStressTensor, H_Voigt = material.KineticMeasures(F,elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        stiffness, tractionforce = __ConstitutiveStiffnessIntegrandDF__(SpatialGradient,
            CauchyStressTensor,H_Voigt,detJ,self.nvar,fem_solver.requires_geometry_update)
        # COMPUTE GEOMETRIC STIFFNESS
        if material.nature != "linear":
            stiffness += self.__GeometricStiffnessIntegrand__(SpatialGradient,CauchyStressTensor,detJ)

        return stiffness, tractionforce


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, CauchyStressTensor, H_Voigt,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveB(B,SpatialGradient,self.ndim,self.nvar)

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
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        B = np.zeros((nodeperelem*nvar,material.H_VoigtSize),dtype=np.float64)

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # UPDATE/NO-UPDATE GEOMETRY
        if fem_solver.requires_geometry_update:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',Jm, EulerElemCoords)
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

            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            t = self.TractionIntegrand(B, SpatialGradient[counter,:,:],
                CauchyStressTensor, requires_geometry_update=fem_solver.requires_geometry_update)

            if fem_solver.requires_geometry_update:
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]


        return tractionforce



    def __GetLocalTraction__(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
        """Get traction vector of the system"""

        # GET LOCAL KINEMATICS
        SpatialGradient, F, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0],
            LagrangeElemCoords, EulerElemCoords, fem_solver.requires_geometry_update)
        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        CauchyStressTensor, _ = material.KineticMeasures(F,elem=elem)
        # COMPUTE LOCAL CONSTITUTIVE STIFFNESS AND TRACTION
        tractionforce = __TractionIntegrandDF__(SpatialGradient,
            CauchyStressTensor,detJ,material.H_VoigtSize,self.nvar,fem_solver.requires_geometry_update)

        return tractionforce


    def TractionIntegrand(self, B, SpatialGradient, CauchyStressTensor,
        requires_geometry_update=True):
        """Applies to displacement based formulation"""

        SpatialGradient = SpatialGradient.T.copy()
        FillConstitutiveB(B,SpatialGradient,self.ndim,self.nvar)

        t=np.zeros((B.shape[0],1))
        if requires_geometry_update:
            TotalTraction = GetTotalTraction(CauchyStressTensor)
            t = np.dot(B,TotalTraction)

        return t



    def GetLocalResidual(self):
        pass



    def GetEnergy(self, function_space, material, LagrangeElemCoords, EulerElemCoords, fem_solver, elem=0):
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
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)

        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientx = np.einsum('ijk,jl->kil',Jm, EulerElemCoords)
        # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
        SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
        # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):
            # COMPUTE THE INTERNAL ENERGY AT THIS GAUSS POINT
            energy = material.InternalEnergy(StrainTensors,elem,counter)
            # INTEGRATE INTERNAL ENERGY
            internal_energy += energy*detJ[counter]

        return internal_energy




    def GetLinearMomentum(self, function_space, material, LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem=0):
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
        F = np.einsum('ij,kli->kjl', EulerElemCoords, MaterialGradient)
        # TIME DERIVATIVE OF F
        Fdot = np.einsum('ij,kli->kjl', VelocityElem, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientx = np.einsum('ijk,jl->kil',Jm, EulerElemCoords)
        # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
        SpatialGradient = np.einsum('ijk,kli->ilj',inv(ParentGradientx),Jm)
        # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
        detJ = np.einsum('i,i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)),np.abs(StrainTensors['J']))

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]):
            GradV = np.dot(Fdot[counter,:,:],np.linalg.inv(F[counter,:,:]))
            # COMPUTE CAUCHY
            CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)
            # INTEGRATE INTERNAL VIRTUAL POWER
            internal_power += np.einsum('ij,ij',CauchyStressTensor,GradV)*detJ[counter]

        return internal_power