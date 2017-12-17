import numpy as np
from .VariationalPrinciple import VariationalPrinciple
from Florence import QuadratureRule, FunctionSpace

from Florence.FiniteElements.LocalAssembly.KinematicMeasures import *
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from Florence.Tensor import issymetric
from Florence.LegendreTransform import LegendreTransform

__all__ = ["ExplicitPenaltyContactFormulation"]

class ExplicitPenaltyContactFormulation(VariationalPrinciple):

    def __init__(self, mesh, plane_normal, distance, kappa, variables_order=(1,),
        quadrature_rules=None, quadrature_type=None, function_spaces=None, compute_post_quadrature=True,
        equally_spaced_bases=False, is_rigid_surface=True, contact_gap_tolerance=1e-6):
        """This is node-to-surface contact formulation based on penalty method"""

        self.plane_normal = np.array(plane_normal).flatten()
        self.distance = float(distance)
        self.kappa = float(kappa)
        self.contact_gap_tolerance = float(contact_gap_tolerance)
        self.is_rigid_surface = is_rigid_surface
        if self.is_rigid_surface:
            return

        if mesh.element_type != "tet" and mesh.element_type != "tri" and \
            mesh.element_type != "quad" and mesh.element_type != "hex":
            raise NotImplementedError( type(self).__name__, "has not been implemented for", mesh.element_type, "elements")

        if isinstance(variables_order,int):
            self.variables_order = (self.variables_order,)
        self.variables_order = variables_order

        super(ExplicitContactFormulation, self).__init__(mesh,variables_order=self.variables_order,
            quadrature_type=quadrature_type,quadrature_rules=quadrature_rules,function_spaces=function_spaces,
            compute_post_quadrature=compute_post_quadrature)

        self.fields = "mechanics"
        self.nvar = 0

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
        pass

    def GetElementalMatricesInVectorForm(self, elem, function_space, mesh, material, fem_solver, Eulerx, Eulerp):
        pass

    def GetLocalStiffness(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        pass

    def __GetLocalStiffness__(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        pass

    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
            H_Voigt, analysis_nature="nonlinear", has_prestress=True):
        pass

    def GetLocalTraction(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""
        pass

    def __GetLocalTraction__(self, function_space, material, LagrangeElemCoords,
        EulerELemCoords, ElectricPotentialElem, fem_solver, elem=0):
        """Get traction vector of the system"""
        pass

    def TractionIntegrand(self, B, SpatialGradient, ElectricDisplacementx,
        CauchyStressTensor, analysis_nature="nonlinear", has_prestress=True):
        pass

    def GetLocalResidual(self):
        pass



    def AssembleTractions(self, mesh, material, Eulerx):
        """Assembly of contact forces is provided within the class itself"""

        # RIGID WALL LOCATED AT
        L = self.distance
        normal = self.plane_normal
        k = self.kappa
        ndim = mesh.points.shape[1]

        if ndim == 2:
            boundary_surface = mesh.edges
        else:
            boundary_surface = mesh.faces
        surfNodes_no, surfNodes_idx, surfNodes_inv = np.unique(boundary_surface, return_index=True, return_inverse=True)
        surfNodes = Eulerx[surfNodes_no,:]
        gNx = surfNodes.dot(normal) + L

        T_contact = np.zeros((mesh.points.shape[0]*ndim,1))
        contactNodes_idx = gNx < self.contact_gap_tolerance
        contactNodes = surfNodes[contactNodes_idx,:]

        # IF NO CONTACT
        if contactNodes.shape[0] == 0:
            return T_contact

        contactNodes_global_idx = surfNodes_no[contactNodes_idx].astype(np.int64)
        normal_gap = gNx[contactNodes_idx]

        # LOOP VERSION
        # for node in range(contactNodes.shape[0]):
        #     t_local = k*normal_gap[node]*normal
        #     T_contact[contactNodes_global_idx[node]*ndim:(contactNodes_global_idx[node]+1)*ndim,0] += t_local

        # VECTORISED VERSION
        T_contact = T_contact.reshape(mesh.points.shape[0],ndim)
        t_local = k*np.outer(normal_gap,normal)
        T_contact[contactNodes_global_idx,:] = t_local
        T_contact = T_contact.ravel()

        return T_contact
