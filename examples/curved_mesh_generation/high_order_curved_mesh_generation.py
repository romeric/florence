import os, sys
from Florence import *
from Florence.VariationalPrinciple import *


def high_order_curved_mesh_generation(p=2, analysis_nature="linear",
    optimise=True, recompute_sparsity_pattern=True, squeeze_sparsity_pattern=False):
    """An example of high order curved mesh generation on a hollow cylinder
        with unstructured tetrahedral elements
    """

    ProblemPath = PWD(__file__)
    mesh_file = ProblemPath + '/Hollow_Cylinder.dat'
    cad_file = ProblemPath + '/Hollow_Cylinder.igs'

    mesh = Mesh()
    mesh.Read(filename=mesh_file, reader_type="salome", element_type="tet")
    mesh.GetHighOrderMesh(p=p, Decimals=7)
    ndim = mesh.InferSpatialDimension()

    material = NeoHookean(ndim, youngs_modulus=1e5, poissons_ratio=0.48)

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition, project_on_curves=True, solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,
        analysis_nature=analysis_nature,
        optimise=optimise,
        recompute_sparsity_pattern=recompute_sparsity_pattern,
        squeeze_sparsity_pattern=squeeze_sparsity_pattern)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # check mesh quality
    assert solution.ScaledJacobian.min() > 0.2
    assert solution.ScaledJacobian.min() < 0.3
    assert solution.ScaledHH.min() > 0.35
    assert solution.ScaledHH.min() < 0.55
    assert solution.ScaledFF.min() > 0.45
    assert solution.ScaledFF.min() < 0.65

    # In-built fancy curvilinear mesh plotter
    # solution.CurvilinearPlot(plot_points=True, point_radius=0.2, color="#E3A933")
    # Write the results to VTK
    # mesh.points += solution.sol[:,:,-1]
    # mesh.WriteVTK("cylinder_mesh")



if __name__ == "__main__":
    # With optimisation ON
    high_order_curved_mesh_generation(p=2, analysis_nature="linear")
    high_order_curved_mesh_generation(p=2, analysis_nature="nonlinear")

    # With optimisation OFF
    high_order_curved_mesh_generation(p=2, analysis_nature="linear", optimise=False)
    high_order_curved_mesh_generation(p=2, analysis_nature="nonlinear", optimise=False)

    high_order_curved_mesh_generation(p=2, analysis_nature="linear", optimise=False,
        recompute_sparsity_pattern=False)
    high_order_curved_mesh_generation(p=2, analysis_nature="nonlinear", optimise=False,
        recompute_sparsity_pattern=False)

    high_order_curved_mesh_generation(p=2, analysis_nature="linear", optimise=False,
        recompute_sparsity_pattern=False, squeeze_sparsity_pattern=True)
    high_order_curved_mesh_generation(p=2, analysis_nature="nonlinear", optimise=False,
        recompute_sparsity_pattern=False, squeeze_sparsity_pattern=True)


