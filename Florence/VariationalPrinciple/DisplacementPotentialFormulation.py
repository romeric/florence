import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence.FiniteElements.ElementalMatrices._KinematicMeasures_ import _KinematicMeasures_
from Florence.Tensor import issymetric
from Florence.LegendreTransform import LegendreTransform

import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from DisplacementPotentialApproachIndices import *

class DisplacementPotentialFormulation(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(1,), 
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=False):

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

        if quadrature_rules == None and self.quadrature_rules == None:

            # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
            optimal_quadrature = 3

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
            quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder, mesh_type=mesh.element_type)
            if self.compute_post_quadrature:
                # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
                post_quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder_post, mesh_type=mesh.element_type)
            else:
                post_quadrature = None

        if function_spaces == None and self.function_spaces == None:

            # CREATE FUNCTIONAL SPACES
            function_space = FunctionSpace(mesh, quadrature, p=C+1)
            if compute_post_quadrature:
                post_function_space = FunctionSpace(mesh, post_quadrature, p=C+1)
            else:
                post_function_space = None

        self.quadrature_rules = (quadrature,post_quadrature)
        self.function_spaces = (function_space,post_function_space)


        local_size = function_space.Bases.shape[0]*self.nvar
        self.local_rows = np.repeat(np.arange(0,local_size),local_size,axis=0)
        self.local_columns = np.tile(np.arange(0,local_size),local_size)
        self.local_size = local_size


    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerp):

        # ALLOCATE
        Domain = function_space

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
        if fem_solver.analysis_type != 'static':
            # COMPUTE THE MASS MATRIX
            massel = Mass(material,LagrangeElemCoords,EulerElemCoords,elem)

        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = ApplyNeumannBoundaryConditions3D(MainData, mesh, elem, LagrangeElemCoords)

        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)
        if fem_solver.analysis_type != 'static':
            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem



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
                ElectricDisplacementx, CauchyStressTensor, H_Voigt, analysis_nature=fem_solver.analysis_nature, 
                has_prestress=fem_solver.has_prestress)
            
            # COMPUTE GEOMETRIC STIFFNESS MATRIX
            if fem_solver.requires_geometry_update:
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor)
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        return stiffness, tractionforce 



    def GetLocalMass(self, function_space, formulation):

        ndim = self.ndim
        nvar = self.nvar
        Domain = function_space

        N = np.zeros((Domain.Bases.shape[0]*nvar,nvar))
        mass = np.zeros((Domain.Bases.shape[0]*nvar,Domain.Bases.shape[0]*nvar))

        # LOOP OVER GAUSS POINTS
        for counter in range(0,Domain.AllGauss.shape[0]):
            # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
            Jm = Domain.Jm[:,:,counter]
            Bases = Domain.Bases[:,counter]
            # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
            ParentGradientX=np.dot(Jm,LagrangeElemCoords)

            # UPDATE/NO-UPDATE GEOMETRY
            if MainData.GeometryUpdate:
                # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
                ParentGradientx = np.dot(Jm,EulerELemCoords)
            else:
                ParentGradientx = ParentGradientX

            # COMPUTE THE MASS INTEGRAND
            rhoNN = self.MassIntegrand(Bases,N,MainData.Minimal,MainData.MaterialArgs)

            if MainData.GeometryUpdate:
                # INTEGRATE MASS
                mass += rhoNN*MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
                # mass += rhoNN*w[g1]*w[g2]*w[g3]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
            else:
                # INTEGRATE MASS
                mass += rhoNN*MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))

        return mass 


    def GetLocalResiduals(self):
        pass

    def GetLocalTractions(self):
        pass


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, H_Voigt, analysis_nature="nonlinear", has_prestress=True):
        """Overrides base for electric potential formulation"""

        # MATRIX FORM
        # SpatialGradient = SpatialGradient.T
        # SpatialGradient = np.ascontiguousarray(SpatialGradient.T)
        SpatialGradient = SpatialGradient.T.copy()

        FillConstitutiveB(B,SpatialGradient,self.ndim,self.nvar)
        BDB = B.dot(H_Voigt.dot(B.T))
        
        t=[]
        if analysis_nature == 'nonlinear' or has_prestress:
            TotalTraction = GetTotalTraction(CauchyStressTensor,ElectricDisplacementx.ravel())
            t = np.dot(B,TotalTraction) 
                
        return BDB, t





    ###############################################################################################
    ###############################################################################################

    # def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
    #     CauchyStressTensor, H_Voigt, analysis_nature="nonlinear", has_prestress=True):

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

    #         if analysis_nature == 'nonlinear' or has_prestress:
    #             CauchyStressTensor_Voigt = np.array([
    #                 CauchyStressTensor[0,0],CauchyStressTensor[1,1],CauchyStressTensor[2,2],
    #                 CauchyStressTensor[0,1],CauchyStressTensor[0,2],CauchyStressTensor[1,2]
    #                 ]).reshape(6,1)

    #             TotalTraction = np.concatenate((CauchyStressTensor_Voigt,ElectricDisplacementx[:,None]),axis=0)

    #     elif SpatialGradient.shape[0]==2:

    #         B[0::nvar,0] = SpatialGradient[0,:]
    #         B[1::nvar,1] = SpatialGradient[1,:]
    #         # Mechanical - Shear Terms
    #         B[0::nvar,2] = SpatialGradient[1,:]
    #         B[1::nvar,2] = SpatialGradient[0,:]

    #         # Electrostatic 
    #         B[2::nvar,3] = SpatialGradient[0,:]
    #         B[2::nvar,4] = SpatialGradient[1,:]

    #         if analysis_nature == 'nonlinear' or has_prestress:
    #             CauchyStressTensor_Voigt = np.array([
    #                 CauchyStressTensor[0,0],CauchyStressTensor[1,1],
    #                 CauchyStressTensor[0,1]]).reshape(3,1)

    #             TotalTraction = np.concatenate((CauchyStressTensor_Voigt,ElectricDisplacementx[:,None]),axis=0)

    #     BDB = np.dot(np.dot(B,H_Voigt),B.T)
    #     t=[]
    #     if analysis_nature == 'nonlinear' or has_prestress:
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








    def __GetLocalStiffness__(self, function_space, material, LagrangeElemCoords, 
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get stiffness matrix of the system"""

        nvar = self.nvar
        nodeperelem = function_space.Bases.shape[0]
        AllGauss = function_space.AllGauss

        # ALLOCATE
        stiffness = np.zeros((nodeperelem*nvar,nodeperelem*nvar),dtype=np.float64)
        tractionforce = np.zeros((nodeperelem*nvar,1),dtype=np.float64)
        B = np.zeros((nodeperelem*nvar,material.H_VoigtSize),dtype=np.float64)

        SpatialGradient, F, detJ = _KinematicMeasures_(function_space.Jm, AllGauss[:,0], LagrangeElemCoords, 
            EulerELemCoords, fem_solver.requires_geometry_update)

        # GET ELECTRIC FIELD
        ElectricFieldx = - np.einsum('ijk,j',SpatialGradient,ElectricPotentialElem)

        # COMPUTE WORK-CONJUGATES AND HESSIAN AT THIS GAUSS POINT
        ElectricDisplacementx, CauchyStressTensor, H_Voigt = material.KineticMeasures(F, ElectricFieldx, elem=elem)

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]): 

            # COMPUTE THE TANGENT STIFFNESS MATRIX
            BDB_1, t = self.ConstitutiveStiffnessIntegrand(B, SpatialGradient[counter,:,:],
                ElectricDisplacementx[counter,:,:], CauchyStressTensor[counter,:,:], H_Voigt[counter,:,:], 
                analysis_nature=fem_solver.analysis_nature, has_prestress=fem_solver.has_prestress)

            
            if fem_solver.requires_geometry_update:
                # BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:],CauchyStressTensor[counter,:,:])
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

        # COMPUTE GEOMETRIC STIFFNESS MATRIX
        stiffness += self.__GeometricStiffnessIntegrand__(SpatialGradient,CauchyStressTensor,detJ)

        return stiffness, tractionforce




    # def foo(self,SpatialGradient, CauchyStressTensor):

    #     # print SpatialGradient.shape
    #     B = np.zeros((SpatialGradient.shape[0]*self.nvar,SpatialGradient.shape[0]*self.nvar))
    #     for a in range(SpatialGradient.shape[0]):
    #         for b in range(SpatialGradient.shape[0]):

    #             # I = np.eye(self.ndim,self.ndim)
    #             dum=0.
    #             for i in range(SpatialGradient.shape[1]):
    #                 for j in range(SpatialGradient.shape[1]):
    #                     dum += SpatialGradient[a,i]*CauchyStressTensor[i,j]*SpatialGradient[b,j]
    #             # B = np.zeros((self.ndim,self.ndim))
    #             # Bs = dum*I

    #             for i in range(self.ndim):
    #                 B[a*self.nvar+i,b*self.nvar+i] = dum


    #     return B


    # def foos(self, SpatialGradient, CauchyStressTensor, detJ):

    #     BDB = np.zeros((SpatialGradient.shape[1]*self.nvar,SpatialGradient.shape[1]*self.nvar))
    #     for g in range(detJ.shape[0]):
    #         # BDB_local = np.zeros((SpatialGradient.shape[1]*self.nvar,SpatialGradient.shape[1]*self.nvar))
    #         for a in range(SpatialGradient.shape[1]):
    #             for b in range(SpatialGradient.shape[1]):

    #                 dum=0.
    #                 for i in range(SpatialGradient.shape[2]):
    #                     for j in range(SpatialGradient.shape[2]):
    #                         dum += SpatialGradient[g,a,i]*CauchyStressTensor[g,i,j]*SpatialGradient[g,b,j]

    #                 for i in range(self.ndim):
    #                     # BDB_local[a*self.nvar+i,b*self.nvar+i] += dum
    #                     BDB[a*self.nvar+i,b*self.nvar+i] += dum*detJ[g]

    #     # BDB[a*self.nvar+i,b*self.nvar+i] += dum*detJ[g]

    #         # BDB += BDB_local#*detJ[g]


    #     return BDB