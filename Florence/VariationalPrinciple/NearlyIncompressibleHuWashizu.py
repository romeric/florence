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
        # BUILD MESHES FOR THESE ORDRES
        p = mesh.InferPolynomialDegree()
        if p != self.variables_order[0] and self.variables_order[0]!=0:
            mesh.GetHighOrderMesh(C=self.variables_order[0]-1)
        if self.variables_order[2] == 0 or self.variables_order[1]==0 or self.variables_order[0]==0:
            # GET MEDIAN OF THE ELEMENTS FOR CONSTANT VARIABLES
            self.median, self.bases_at_median = mesh.Median

        # GET QUADRATURE RULES
        self.quadrature_rules = [None]*len(self.variables_order)
        QuadratureOpt = 3
        for counter, degree in enumerate(self.variables_order):
            if degree==0:
                degree=1
            # self.quadrature_rules[counter] = list(GetBasesAtInegrationPoints(degree-1, 2*degree,
                # QuadratureOpt,mesh.element_type))
            quadrature = QuadratureRule(optimal=QuadratureOpt, 
                norder=2*degree, mesh_type=mesh.element_type)
            self.quadrature_rules[counter] = quadrature
            # FunctionSpace(mesh, p=degree, 2*degree, QuadratureOpt,mesh.element_type)
            # self.quadrature_rules[counter] = list()

        # print dir(self.quadrature_rules[0])
        # print self.quadrature_rules[0][0].Jm
        # print self.median
        # exit()


    def GetLocalStiffness(self, bases, material, LagrangeElemCoords, EulerELemCoords, elem):

        nvar = self.nvar
        ndim = self.ndim
        Domain = bases

        # ALLOCATE
        stiffness = np.zeros((Domain.Bases.shape[0]*nvar,Domain.Bases.shape[0]*nvar),dtype=np.float64)
        tractionforce = np.zeros((MainData.Domain.Bases.shape[0]*nvar,1),dtype=np.float64)
        # B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize),dtype=np.float64)
        B = np.zeros((Domain.Bases.shape[0]*nvar,material.H_VoigtSize),dtype=np.float64)

        # COMPUTE KINEMATIC MEASURES AT ALL INTEGRATION POINTS USING EINSUM (AVOIDING THE FOR LOOP)
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Domain.Jm, LagrangeElemCoords)
        # MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
        MaterialGradient = np.einsum('ijk,kli->ijl', la.inv(ParentGradientX), Domain.Jm)
        # DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
        F = np.einsum('ij,kli->kjl', EulerELemCoords, MaterialGradient)

        # COMPUTE REMAINING KINEMATIC MEASURES
        StrainTensors = KinematicMeasures(F, AnalysisType)
        
        # UPDATE/NO-UPDATE GEOMETRY
        if fem_sovler.requires_geometry_update:
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientx = np.einsum('ijk,jl->kil',Domain.Jm,EulerELemCoords)
            # SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
            SpatialGradient = np.einsum('ijk,kli->ilj',la.inv(ParentGradientx),Domain.Jm)
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            detJ = np.einsum('i,i,i->i',Domain.AllGauss[:,0],np.abs(la.det(ParentGradientX)),np.abs(StrainTensors['J']))
        else:
            # SPATIAL GRADIENT AND MATERIAL GRADIENT TENSORS ARE EQUAL
            SpatialGradient = np.einsum('ikj',MaterialGradient)
            # COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
            detJ = np.einsum('i,i->i',Domain.AllGauss[:,0],np.abs(la.det(ParentGradientX)))

        

        # LOOP OVER GAUSS POINTS
        for counter in MainData.Range(Domain.AllGauss.shape[0]): 

            # COMPUTE THE HESSIAN AT THIS GAUSS POINT
            H_Voigt = material.Hessian(StrainTensors,elem,counter)
            
            # COMPUTE CAUCHY STRESS TENSOR
            CauchyStressTensor = []
            if fem_sovler.requires_geometry_update:
                CauchyStressTensor = material.CauchyStress(StrainTensors,ElectricFieldx,elem,counter)

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B,AnalysisType,
                MainData.Prestress,SpatialGradient[counter,:,:],CauchyStressTensor,H_Voigt)
            
            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if fem_sovler.requires_geometry_update:
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        return stiffness, tractionforce 




    def GetLocalResiduals(self):
        pass

    def GetLocalTractions(self):
        pass