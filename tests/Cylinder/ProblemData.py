from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    assert len(args) == 1
    MainData = args[0]
    ndim = 3

    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)

    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/Hollow_Cylinder.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")
    mesh.GetHighOrderMesh(p=MainData.C+1)

    cad_file = ProblemPath + '/Hollow_Cylinder.igs'

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=5,analysis_nature="linear")

    return formulation, mesh, material, boundary_condition, solver, fem_solver

