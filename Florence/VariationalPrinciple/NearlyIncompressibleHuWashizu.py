import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence.Tensor import issymetric


class NearlyIncompressibleHuWashizu(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(2,0,0)):

        if mesh.element_type != "tet" and mesh.element_type != "tri":
            raise NotImplementedError( type(self.__name__), "has not been implemented for", mesh.element_type, "elements")

        self.variables_order = variables_order
        super(NearlyIncompressibleHuWashizu, self).__init__(mesh,variables_order=self.variables_order)



    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, TotalPot):

        # ALLOCATE
        Domain = function_space

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

        # COMPUTE THE STIFFNESS MATRIX
        stiffnessel, t = self.GetLocalStiffness(function_space,material,LagrangeElemCoords,EulerElemCoords,fem_solver,elem)

        I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
        if fem_solver.analysis_type != 'static':
            # COMPUTE THE MASS MATRIX
            massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)

        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = ApplyNeumannBoundaryConditions3D(MainData, mesh, elem, LagrangeElemCoords)

        # STATIC CONDENSATION
        # if C>0:
            # stiffnessel,f = StaticCondensation(stiffnessel,f,C,nvar)
            # massel,f = StaticCondensation(stiffnessel,f,C,nvar)

        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)
        if fem_solver.analysis_type != 'static':
            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem



    def GetLocalStiffness(self, function_space, material, LagrangeElemCoords, EulerELemCoords, fem_solver, elem=0):
        """Get stiffness matrix of the system"""

        nvar = self.nvar
        ndim = self.ndim
        Domain = function_space

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = Domain.Jm
        AllGauss = Domain.AllGauss

        # ALLOCATE
        stiffness = np.zeros((Domain.Bases.shape[0]*nvar,Domain.Bases.shape[0]*nvar),dtype=np.float64)
        tractionforce = np.zeros((Domain.Bases.shape[0]*nvar,1),dtype=np.float64)
        B = np.zeros((Domain.Bases.shape[0]*nvar,material.H_VoigtSize),dtype=np.float64)

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', inv(ParentGradientX), Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)


        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, self.analysis_nature)
        
        # UPDATE/NO-UPDATE GEOMETRY
        if fem_solver.requires_geometry_update:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',Domain.Jm,EulerELemCoords)
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
            H_Voigt = material.Hessian(StrainTensors,None,elem,counter)
            
            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_solver.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,None,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                CauchyStressTensor, H_Voigt, analysis_nature=fem_solver.analysis_nature, 
                has_prestress=fem_solver.has_prestress)
            
            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if fem_solver.requires_geometry_update:
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        return stiffness, tractionforce 



    def GetLocalMass(self):
        pass 


    def GetLocalResiduals(self):
        pass

    def GetLocalTractions(self):
        pass