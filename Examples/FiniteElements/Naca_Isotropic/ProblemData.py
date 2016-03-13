import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *
from Florence.VariationalPrinciple import *


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
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear",parallel=True)

    return formulation, mesh, material, boundary_condition, solver, fem_solver
