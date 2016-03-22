from warnings import warn
import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence.Tensor import issymetric


class NearlyIncompressibleHuWashizu(VariationalPrinciple):

    def __init__(self, mesh, variables_order=(2,0,0), 
        quadrature_rules=None, quadrature_type=None, function_spaces=None):

        if mesh.element_type != "tet" and mesh.element_type != "tri":
            raise NotImplementedError( type(self.__name__), "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(NearlyIncompressibleHuWashizu, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces)


        C = mesh.InferPolynomialDegree() - 1
        if C==0 and self.variables_order[0] > 1:
            warn("Incompatible mesh and for the interpolation degree chosen for function spaces")
            mesh.GetHighOrderMesh(p=C+1)             

        # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
        if mesh.element_type == "tri" or mesh.element_type == "tet":
            optimal_quadrature = 3

        norder = 2*C
        # TAKE CARE OF C=0 CASE
        if norder == 0:
            norder = 1
        # GET QUADRATURE
        quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder, mesh_type=mesh.element_type)
        function_space = FunctionSpace(mesh, quadrature, p=C+1)

        # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
        norder_post = 2*(C+1)
        post_quadrature = QuadratureRule(optimal=optimal_quadrature, norder=norder_post, mesh_type=mesh.element_type)

        # CREATE FUNCTIONAL SPACES
        post_function_space = FunctionSpace(mesh, post_quadrature, p=C+1)

        self.quadrature_rules = (quadrature,post_quadrature)
        self.function_spaces = (function_space,post_function_space)



    def GetElementalMatrices(self, elem, function_space, mesh, material, fem_solver, Eulerx, TotalPot):

        # ALLOCATE
        Domain = function_space

        massel=[]; f = []
        # GET THE FIELDS AT THE ELEMENT LEVEL
        LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
        EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

        if self.ndim==2:
            VolumesX = mesh.Areas()
            Volumesx = mesh.Areas(gpoints=Eulerx)
        elif ndim==3:
            VolumesX = mesh.Volumes()
            Volumesx = mesh.Volumes(gpoints=Eulerx)


        # COMPUTE THE STIFFNESS MATRIX
        stiffnessel, t = self.GetLocalStiffness(function_space, material, VolumesX, Volumesx,
            LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

        # STATIC CONDENSATION
        stiffnessel = stiffnessel[:12,:12] - stiffnessel[:12,12:].dot(np.linalg.inv(stiffnessel[12:,12:]).dot(stiffnessel[12:,:12]))

        I_mass_elem = []; J_mass_elem = []; V_mass_elem = []
        if fem_solver.analysis_type != 'static':
            # COMPUTE THE MASS MATRIX
            massel = Mass(MainData,LagrangeElemCoords,EulerElemCoords,elem)

        if fem_solver.has_moving_boundary:
            # COMPUTE FORCE VECTOR
            f = ApplyNeumannBoundaryConditions3D(formulation, mesh, elem, LagrangeElemCoords)

        I_stiff_elem, J_stiff_elem, V_stiff_elem = self.FindIndices(stiffnessel)
        if fem_solver.analysis_type != 'static':
            I_mass_elem, J_mass_elem, V_mass_elem = self.FindIndices(massel)

        return I_stiff_elem, J_stiff_elem, V_stiff_elem, t, f, I_mass_elem, J_mass_elem, V_mass_elem



    def GetLocalStiffness(self, function_space, material, VolumesX, Volumesx, LagrangeElemCoords, EulerELemCoords, fem_solver, elem=0):
        """Compute stiffness matrix of an element"""

        nvar = self.nvar
        ndim = self.ndim
        Domain = function_space

        det = np.linalg.det
        inv = np.linalg.inv
        Jm = Domain.Jm
        AllGauss = Domain.AllGauss


        material.kappa = material.lamb+2.0*material.mu/3.0
        material.pressure = (Volumesx - VolumesX)/VolumesX


        # ALLOCATE
        stiffness = np.zeros((Domain.Bases.shape[0]*nvar,Domain.Bases.shape[0]*nvar),dtype=np.float64)
        stiffness_XP = np.zeros(Domain.Bases.shape[0]*nvar,dtype=np.float64)
        stiffness_JJ = 0
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
        StrainTensors = KinematicMeasures(F, fem_solver.analysis_nature)
        # COFACTOR OF DEFORMATION GRADIENT TENSOR
        H = np.einsum('ijk,k->ijk',np.linalg.inv(F).T,StrainTensors['J'])
        # COMPUTE H:\nabla_{\delta\vec{v}} - TRANSPOSE IS NECESSARY TO FORCE F-CONTIGUOUS
        B_XP = np.einsum('ijk,kjl->il', H, MaterialGradient).T.ravel()
        
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

        # print(MaterialGradient.shape)
        # exit()
        stiffness_x = np.zeros((12,12))
        dN = np.zeros((6,2))

        # LOOP OVER GAUSS POINTS
        for counter in range(AllGauss.shape[0]): 

            # for a in range(6):
            #     for b in range(6):
            #         dNa = SpatialGradient[counter,a,:]
            #         dNb = SpatialGradient[counter,b,:]
                    # stiffness_x[2*a:2*(a+1),2*b:2*(b+1)] += dNa[:,None].dot(dNb[None,:])*detJ[counter]

            dN[:] += 1./Volumesx[elem]*SpatialGradient[counter,:,:]*detJ[counter]


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
                BDB_1 += self.GeometricStiffnessIntegrand(SpatialGradient[counter,:,:], CauchyStressTensor)
                # INTEGRATE TRACTION FORCE
                tractionforce += t*detJ[counter]

            # INTEGRATE STIFFNESS
            stiffness += BDB_1*detJ[counter]

            # K_JP, K_PJ
            # detJ_XP = np.einsum('i,i->i',AllGauss[:,0],np.abs(det(ParentGradientX)))
            # stiffness_XP[:] = B_XP*detJ_XP[counter]

            # K_JJ
            # stiffness_JJ += detJ_XP[counter]

        # stiffness_JP = np.copy(stiffness_JJ)
        # stiffness_JJ *= material.kappa

        # b1 = np.concatenate((stiffness,np.zeros((12,1)),stiffness_XP[:,None]),axis=1)
        # b2 = np.concatenate((np.zeros((1,12)),stiffness_XP[:,None].T),axis=0)
        # b3 = np.array([[stiffness_JJ, - stiffness_JP], [-stiffness_JP, 0]])
        # b4 = np.concatenate((b2,b3),axis=1)
        # bf = np.concatenate((b1,b4),axis=0)
        # stiffness = bf

        # stiffness = stiffness+stiffness_x*material.kappa/VolumesX[elem]

        # print dN[0,:,None].dot(dN[0,:,None].T)
        # exit()
        for a in range(6):
            for b in range(6):
                stiffness_x[2*a:2*(a+1),2*b:2*(b+1)] += dN[a,:,None].dot(dN[b,:,None].T)

        kappa_bar = material.kappa*Volumesx[elem]/VolumesX[elem]
        stiffness_x *= (kappa_bar*Volumesx[elem])
        stiffness = stiffness+stiffness_x

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