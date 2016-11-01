import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(*args, **kwargs):


    ndim = 3
    p = 2

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
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
    mesh.GetHighOrderMesh(p=p)

    cad_file = ProblemPath + '/Sphere.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=1000.0,condition=1e10,project_on_curves=False,solve_for_planar_faces=False)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear")

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition, solver=solver)

    solution.CurvilinearPlot()


if __name__ == "__main__":
    ProblemData()