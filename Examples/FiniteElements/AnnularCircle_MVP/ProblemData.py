import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData):

    MainData.ndim = 2   

    # material = Material("LinearModel",MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
    #     E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)

    # material = NearlyIncompressibleNeoHookean(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))

    # FileName = ProblemPath + '/Mesh_Annular_Circle_502.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_312.dat'
    filename = ProblemPath + '/Mesh_Annular_Circle_75.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Read")


    cad_file = ProblemPath + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=1000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(analysis_type="static",analysis_nature="nonlinear")

    return formulation, mesh, material, boundary_condition, solver, fem_solver