import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *
from Florence.VariationalPrinciple import *

def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.AnalysisType = 'Linear'
    MainData.AnalysisType = 'Nonlinear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    mesh = Mesh()
    mesh.Reader(element_type="tet",reader_type="Sphere")


    cad_file = ProblemPath + '/Sphere.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=1000.0,condition=1e10,project_on_curves=False,solve_for_planar_faces=False)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    # MainData.solver = solver 

    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear")

    return formulation, mesh, material, boundary_condition, solver, fem_solver