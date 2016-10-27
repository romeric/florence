import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
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
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_nature="linear")

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.CurvilinearPlot()
    # print solution.sol[:,:,0].shape
    # mesh.WriteHDF5("/home/roman/ZZ_P"+str(MainData.C+1)+".mat",{"TotalDisp":solution.sol})

if __name__ == "__main__":
    class MainData():
        C = 1
    ProblemData(MainData)