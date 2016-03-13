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
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

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

    # FileName = ProblemPath + '/Mesh_Cube_Tet_181.dat'
    # FileName = ProblemPath + '/Sphere_1483.dat'
    # FileName = ProblemPath + '/Torus_612.dat' # Torus
    # FileName = ProblemPath + '/Torus_check.dat'
    # FileName = ProblemPath + '/TPipe_4006.dat'
    # FileName = ProblemPath + '/TPipe_2262.dat'
    filename = ProblemPath + '/Hollow_Cylinder.dat'
    # FileName = ProblemPath + "/TPipe_2_1302.dat"
    # FileName = ProblemPath + "/TPipe_2_1247.dat"
    # FileName = ProblemPath + "/FullTPipe.dat"
    # FileName = ProblemPath + '/Cylinder.dat'
    # FileName = ProblemPath + '/Revolution_1.dat'
    # FileName = ProblemPath + '/Extrusion_116.dat'
    # FileName = ProblemPath + '/Extrusion_2_416.dat'
    # FileName = ProblemPath + '/ufc_206.dat'
    # FileName = ProblemPath + '/ucp_206.dat'
    # FileName = ProblemPath + '/penc.dat'
    # FileName = ProblemPath + '/gopro.dat' #
    # FileName = ProblemPath + '/bracketH0.dat' #

    # FileName = ProblemPath + '/form1.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")


    # IGES_File = ProblemPath + '/Sphere.igs'
    # IGES_File = ProblemPath + '/Torus.igs'
    # IGES_File = ProblemPath + '/PipeTShape.igs'
    # IGES_File = ProblemPath + '/TPipe_2.igs'
    # IGES_File = ProblemPath + '/FullTPipe.igs'
    cad_file = ProblemPath + '/Hollow_Cylinder.igs'
    # IGES_File = ProblemPath + '/Cylinder.igs'
    # IGES_File = ProblemPath + '/Revolution_1.igs'
    # IGES_File = ProblemPath + '/Extrusion.igs'
    # IGES_File = ProblemPath + '/Extrusion_2.igs'
    # IGES_File = ProblemPath + '/ufc_206.igs'
    # IGES_File = ProblemPath + '/ucp_206.igs'
    # IGES_File = ProblemPath + '/Porta_Canetas.igs'
    # IGES_File = ProblemPath + '/gopro.igs' #
    # IGES_File = ProblemPath + '/bracket.igs' #

    # IGES_File = ProblemPath + '/form1.igs'

    # sphere
    # scale = 1000.
    # condition = 1000.

    # torus
    # scale = 1000.
    # condition = 1.0e20

    # pipe t-shape
    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    # MainData.solver = solver

    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear")

    return formulation, mesh, material, boundary_condition, solver, fem_solver

