import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 2
    MainData.Fields = 'Mechanics'
    # MainData.Fields = 'ElectroMechanics'
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.Analysis = 'Dynamic'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))

    Reader = 'ReadSeparate'
    ConnectivityFile = ProblemPath + '/elements_naca.dat'
    CoordinatesFile = ProblemPath +'/points_naca.dat'

    mesh = Mesh()
    mesh.Reader(reader_type="ReadSeparate",element_type="tri",
        connectivity_file=ConnectivityFile,coordinate_file=CoordinatesFile)


    def NURBSParameterisation():
        control = np.loadtxt(ProblemPath+'/controls_naca.dat',delimiter=',')
        knots = np.loadtxt(ProblemPath+'/knots_naca.dat',delimiter=',')
        return [({'U':(knots,),'Pw':control,'start':0,'end':2.039675505705710,'degree':3})]

    def NURBSCondition(x):
        return np.sqrt(x[:,:,0]**2 + x[:,:,1]**2) < 2

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(requires_cad=False,scale=1.0,condition=2.0)
    boundary_condition.SetNURBSParameterisation(NURBSParameterisation)
    boundary_condition.SetNURBSCondition(NURBSCondition,mesh.points[mesh.edges[:,:2],:])
        
    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    MainData.solver = solver

    # class BoundaryData(object):
    #     # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
    #     Type = 'nurbs'
    #     RequiresCAD = False
    #     # CurvilinearMeshNodalSpacing = 'fekete'

    #     IGES_File = ''
    #     scale = 1.0
    #     condition = 2


    #     def NURBSParameterisation(self):
    #         control = np.loadtxt(ProblemPath+'/controls_naca.dat',delimiter=',')
    #         knots = np.loadtxt(ProblemPath+'/knots_naca.dat',delimiter=',')
    #         return [({'U':(knots,),'Pw':control,'start':0,'end':2.039675505705710,'degree':3})]

    #     def NURBSCondition(self,x):
    #         return np.sqrt(x[:,0]**2 + x[:,1]**2) < 2


    #     def ProjectionCriteria(self,mesh):
    #         projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
    #         num = mesh.edges.shape[1]
    #         for iedge in range(mesh.edges.shape[0]):
    #             x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
    #             y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
    #             x *= self.scale
    #             y *= self.scale 
    #             if np.sqrt(x*x+y*y)< self.condition:
    #                 projection_edges[iedge]=1

    #         return projection_edges



    return mesh, material, boundary_condition