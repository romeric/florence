import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 3
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'

    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/Hollow_Cylinder.dat'


    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")

    cad_file = ProblemPath + '/Hollow_Cylinder.igs'

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver

    return mesh, material, boundary_condition

