import os, sys
from Florence import *
from Florence.VariationalPrinciple import *


def high_order_curved_mesh_generation(p=2):
    """An example of high order curved mesh generation on a hollow cylinder
        with unstructured tetrahedral elements
    """

    ProblemPath = PWD(__file__)
    mesh_file = ProblemPath + '/Hollow_Cylinder.dat'
    cad_file = ProblemPath + '/Hollow_Cylinder.igs'

    mesh = Mesh()
    mesh.Reader(filename=mesh_file, element_type="tet")
    mesh.GetHighOrderMesh(p=p, Decimals=7)
    ndim = mesh.InferSpatialDimension()

    material = NeoHookean_2(ndim, youngs_modulus=1e5, poissons_ratio=0.48)

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition, project_on_curves=True, solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2, analysis_nature="linear", has_low_level_dispatcher=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # In-built fancy curvilinear mesh plotter
    # solution.CurvilinearPlot(plot_points=True, point_radius=0.2, color="#E3A933")
    # Write the results to VTK
    mesh.points += solution.sol[:,:,-1]
    mesh.WriteVTK("cylinder_mesh")



if __name__ == "__main__":
    high_order_curved_mesh_generation(p=2)


