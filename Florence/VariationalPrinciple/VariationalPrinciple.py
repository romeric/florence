import numpy as np
from Florence import QuadratureRule, FunctionSpace, Mesh
from Florence.FiniteElements.LocalAssembly._KinematicMeasures_ import _KinematicMeasures_
from Florence.VariationalPrinciple._GeometricStiffness_ import GeometricStiffnessIntegrand as GetGeomStiffness
from .DisplacementApproachIndices import FillGeometricB
from ._MassIntegrand_ import __MassIntegrand__, __ConstantMassIntegrand__


__all__ = ["VariationalPrinciple"]


class VariationalPrinciple(object):

    energy_dissipation = []
    internal_energy = []
    kinetic_energy = []
    external_energy = []

    power_dissipation = []
    internal_power = []
    kinetic_power = []
    external_power = []


    def __init__(self, mesh, variables_order=(1,0),
        analysis_type='static', analysis_nature='nonlinear', fields='mechanics',
        quadrature_rules=None, median=None, quadrature_type=None,
        function_spaces=None, compute_post_quadrature=True):

        self.variables_order = variables_order
        self.nvar = None
        self.ndim = mesh.points.shape[1]

        if isinstance(self.variables_order,int):
            self.variables_order = tuple(self.variables_order)

        self.quadrature_rules = quadrature_rules
        self.quadrature_type = quadrature_type
        self.function_spaces = function_spaces
        self.median = median
        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature
        self.fields = fields

        self.compute_post_quadrature = compute_post_quadrature


        # GET NUMBER OF VARIABLES
        self.GetNumberOfVariables()

        # # BUILD MESHES FOR THESE ORDRES
        # p = mesh.InferPolynomialDegree()

        # if p != self.variables_order[0] and self.variables_order[0]!=0:
        #     mesh.GetHighOrderMesh(C=self.variables_order[0]-1)

        # if len(self.variables_order) == 2:
        #     if self.variables_order[2] == 0 or self.variables_order[1]==0 or self.variables_order[0]==0:
        #         # GET MEDIAN OF THE ELEMENTS FOR CONSTANT VARIABLES
        #         self.median, self.bases_at_median = mesh.Median

        #     # GET QUADRATURE RULES
        #     self.quadrature_rules = [None]*len(self.variables_order)
        #     QuadratureOpt = 3
        #     for counter, degree in enumerate(self.variables_order):
        #         if degree==0:
        #             degree=1
        #         # self.quadrature_rules[counter] = list(GetBasesAtInegrationPoints(degree-1, 2*degree,
        #             # QuadratureOpt,mesh.element_type))
        #         quadrature = QuadratureRule(optimal=QuadratureOpt,
        #             norder=2*degree, mesh_type=mesh.element_type)
        #         self.quadrature_rules[counter] = quadrature
        #         # FunctionSpace(mesh, p=degree, 2*degree, QuadratureOpt,mesh.element_type)
        #         # self.quadrature_rules[counter] = list()


    def GetQuadratureOrder(self,C,element_type):
        """Finds quadrature degree/strength for a given polynomial order C=p-1 [where is polynomial degree]"""
        if element_type == "tri" or element_type == "tet":
            norder = 2*C if C > 0 else 1
            norder_post = 2*(C+1)
        else:
            norder = C+2
            norder_post = 2*(C+2)

        return norder, norder_post


    def GetVolume(self, function_space, LagrangeElemCoords, EulerELemCoords, requires_geometry_update, elem=0):
        """ Find the volume (area in 2D) of element [could be curved or straight]
        """

        # GET LOCAL KINEMATICS
        detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0],
            LagrangeElemCoords, EulerELemCoords, requires_geometry_update)[2]

        # volume = 0.
        # LOOP OVER GAUSS POINTS
        # for counter in range(function_space.AllGauss.shape[0]):
            # volume += detJ[counter]

        # return volume

        return detJ.sum()


    def ConstitutiveStiffnessIntegrand(self, B, SpatialGradient, CauchyStressTensor, H_Voigt,
        analysis_nature="nonlinear", has_prestress=True):
        """Applies to displacement based formulation"""
        pass

    def __ConstitutiveStiffnessIntegrand__(self, SpatialGradient, H_Voigt, CauchyStressTensor, detJ,
        analysis_nature="nonlinear", has_prestress=False, requires_geometry_update=True):
        """Applies to displacement based formulation"""
        pass


    def GeometricStiffnessIntegrand(self, SpatialGradient, CauchyStressTensor):
        """Applies to displacement based, displacement potential based and all mixed
        formulations that involve static condensation"""

        ndim = self.ndim
        nvar = self.nvar

        B = np.zeros((nvar*SpatialGradient.shape[0],ndim*ndim))
        S = np.zeros((ndim*ndim,ndim*ndim))
        SpatialGradient = SpatialGradient.T.copy('c')

        FillGeometricB(B,SpatialGradient,S,CauchyStressTensor,ndim,nvar)

        BDB = np.dot(np.dot(B,S),B.T)

        return BDB


    def __GeometricStiffnessIntegrand__(self, SpatialGradient, CauchyStressTensor, detJ):
        """Applies to displacement based formulation"""
        return GetGeomStiffness(np.ascontiguousarray(SpatialGradient),CauchyStressTensor, detJ, self.nvar)


    # def TractionIntegrand(self, B, SpatialGradient, CauchyStressTensor,
    #     analysis_nature="nonlinear", has_prestress=True):
    #     """Applies to displacement based formulation"""

    #     SpatialGradient = SpatialGradient.T.copy()
    #     t=[]
    #     if analysis_nature == 'nonlinear' or has_prestress:
    #         TotalTraction = GetTotalTraction(CauchyStressTensor)
    #         t = np.dot(B,TotalTraction)

    #     return t


    def TractionIntegrand(self, B, SpatialGradient, CauchyStressTensor,
        analysis_nature="nonlinear", has_prestress=True):
        """Applies to displacement based formulation"""
        pass


    def __TractionIntegrand__(self, B, SpatialGradient, CauchyStressTensor,
        analysis_nature="nonlinear", has_prestress=True):
        """Applies to displacement based formulation"""
        pass


    def MassIntegrand(self, Bases, N, material):

        nvar = self.nvar
        ndim = self.ndim
        rho = material.rho

        for ivar in range(ndim):
            N[ivar::nvar,ivar] = Bases

        rhoNN = rho*np.dot(N,N.T)
        return rhoNN

    def GetConstantMassIntegrand(self, Domain, material):
        # COMPUTE THE CONSTANT PART OF MASS MATRIX 'rho*np.dot(N,N.T)'
        # THIS SHOULD BE CALLED ONCE AS IT IS THE SAME FOR EVERY ELEMENT
        # N = np.zeros((Domain.Bases.shape[0]*self.nvar,self.nvar))
        # rhoNN_all = np.zeros((Domain.Bases.shape[0]*self.nvar,Domain.Bases.shape[0]*self.nvar,Domain.AllGauss.shape[0]))
        # # LOOP OVER GAUSS POINTS
        # for counter in range(0,Domain.AllGauss.shape[0]):
        #     # COMPUTE THE MASS INTEGRAND
        #     rhoNN_all[:,:,counter] = self.MassIntegrand(Domain.Bases[:,counter],N,material)

        # self.constant_mass_integrand = rhoNN_all
        # return rhoNN_all

        N = np.zeros((Domain.Bases.shape[0]*self.nvar,self.nvar))
        rhoNN_all = np.zeros((Domain.AllGauss.shape[0],Domain.Bases.shape[0]*self.nvar,Domain.Bases.shape[0]*self.nvar))
        # LOOP OVER GAUSS POINTS
        for counter in range(0,Domain.AllGauss.shape[0]):
            # COMPUTE THE MASS INTEGRAND
            rhoNN_all[counter,:,:] = self.MassIntegrand(Domain.Bases[:,counter],N,material)

        self.constant_mass_integrand = rhoNN_all
        return rhoNN_all


    def GetLocalMass(self, function_space, material, LagrangeElemCoords, EulerELemCoords, fem_solver, elem):

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

            # COMPUTE THE MASS INTEGRAND
            rhoNN = self.MassIntegrand(Bases,N,material)
            # INTEGRATE MASS
            mass += rhoNN*Domain.AllGauss[counter,0]*np.abs(np.linalg.det(ParentGradientX))

        return mass

    def __GetLocalMass__(self, function_space, material, LagrangeElemCoords, EulerELemCoords, fem_solver, elem):
        # GET LOCAL KINEMATICS
        _, _, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0], LagrangeElemCoords,
            EulerELemCoords, False)
        return __MassIntegrand__(material.rho,function_space.Bases,detJ,material.ndim,material.nvar)


    def GetLocalMass_Efficient(self, function_space, material, LagrangeElemCoords, EulerELemCoords, fem_solver, elem):
        """Computes elemental mass matrix in a very efficient way by computing rhoNN only once"""

        Domain = function_space
        # MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
        ParentGradientX = np.einsum('ijk,jl->kil', Domain.Jm, LagrangeElemCoords)
        # COMPUTE ONCE detJ
        detJ = np.einsum('i,i->i',Domain.AllGauss[:,0],np.abs(np.linalg.det(ParentGradientX)))
        # INTEGRATE MASS
        # mass = np.einsum("ijk,k",self.constant_mass_integrand,detJ)
        return np.einsum("ijk,i",self.constant_mass_integrand,detJ)

        # Domain = function_space
        # rhoNN_all = self.constant_mass_integrand
        # mass = np.zeros((Domain.Bases.shape[0]*self.nvar,Domain.Bases.shape[0]*self.nvar))
        # LOOP OVER GAUSS POINTS
        # for counter in range(0,Domain.AllGauss.shape[0]):
        #     # INTEGRATE MASS
        #     mass += rhoNN_all[:,:,counter]*detJ[counter]
        # return mass

    def __GetLocalMass_Efficient__(self, function_space, material, LagrangeElemCoords, EulerELemCoords, fem_solver, elem):
        """Computes elemental mass matrix in a very efficient way by computing rhoNN only once"""
        # GET LOCAL KINEMATICS
        _, _, detJ = _KinematicMeasures_(function_space.Jm, function_space.AllGauss[:,0], LagrangeElemCoords,
            EulerELemCoords, False)
        return __ConstantMassIntegrand__(self.constant_mass_integrand,detJ)

    def GetLumpedMass(self,mass):
        return np.sum(mass,1)[:,None]

    def VerifyMass(self,density, ndim, mass, detJ):
        if np.isclose(mass.sum(),ndim*density*detJ.sum()):
            raise RuntimeError("Local mass matrix is not computed correctly")
        return


    def FindIndices(self,A):
        return self.local_rows, self.local_columns, A.ravel()
        # return self.local_rows, self.local_columns, A.flatten()



    def GetNumberOfVariables(self):
        """Returns (self.nvar) i.e. number of variables/unknowns per node, for the formulation.
            Note that self.nvar does not take into account the unknowns which get condensated
        """

        # nvar = 0
        # for i in self.variables_order:
        #     # DO NOT COUNT VARIABLES THAT GET CONDENSED OUT
        #     if i!=0:
        #         if mesh.element_type == "tri":
        #             nvar += (i+1)*(i+2) // 2
        #         elif mesh.element_type == "tet":
        #             nvar += (i+1)*(i+2)*(i+3) // 6
        #         elif mesh.element_type == "quad":
        #             nvar += (i+1)**2
        #         elif mesh.element_type == "hex":
        #             nvar += (i+1)**3

        # nvar = sum(self.variables_order)
        if self.nvar == None:
            self.nvar = self.ndim
        return self.nvar


    def __call__(self,*args,**kwargs):
        """This is purely to get around pickling while parallelising"""
        return self.GetElementalMatrices(*args,**kwargs)
        # return self.GetElementalMatricesInVectorForm(*args,**kwargs)

























class upFormulation():
    pass


class FiveFieldPenalty(VariationalPrinciple):

    def __init__(self,mesh):

        self.submeshes = []*4
        # PREPARE SUBMESHES
        if mesh.element_type == "tri":
            self.submeshes[0].elements = mesh.elements[:,:3]
            self.submeshes[0].nelem = mesh.elements.shape[0]
            self.submeshes[0].points = mesh.points[:self.submeshes[0].elements.max()+1,:]
            self.submeshes[0].element_type = "tri"




