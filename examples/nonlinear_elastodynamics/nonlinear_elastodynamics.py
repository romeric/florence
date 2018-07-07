import numpy as np
from Florence import *


def nonlinear_elastodynamics(optimise=True):

    n=6
    mesh = Mesh()
    mesh.Parallelepiped(upper_right_front_point=(1,1,1),nx=n,ny=n,nz=n,element_type="hex") # p=4

    material = MooneyRivlin(mesh.InferSpatialDimension(), mu1=1e5, mu2=1e5, lamb=1e6, rho=1200)

    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,:] = 0.
        X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        boundary_data[X_0,2] = 0.3

        return boundary_data

    time_step = 10
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementFormulation(mesh)


    implicit_solver = FEMSolver(total_time=.1,
        number_of_load_increments=time_step,
        analysis_nature="nonlinear",
        analysis_type="dynamic",
        analysis_subtype="implicit",
        newton_raphson_tolerance=1e-10,
        optimise=optimise,
        compute_energy_dissipation=True,
        compute_linear_momentum_dissipation=True
        )

    results_implicit = implicit_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    boundary_condition.__reset_state__()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    explicit_solver_consistent_mass = FEMSolver(total_time=.1,
        number_of_load_increments=time_step*8,
        analysis_nature="nonlinear",
        analysis_type="dynamic",
        analysis_subtype="explicit",
        mass_type="consistent",
        optimise=optimise,
        compute_energy_dissipation=True,
        compute_linear_momentum_dissipation=True
        )

    results_explicit_consistent_mass = explicit_solver_consistent_mass.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    boundary_condition.__reset_state__()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    explicit_solver_lumped_mass = FEMSolver(total_time=.1,
        number_of_load_increments=time_step*8,
        analysis_nature="nonlinear",
        analysis_type="dynamic",
        analysis_subtype="explicit",
        mass_type="lumped",
        optimise=optimise
        )

    results_explicit_lumped_mass = explicit_solver_lumped_mass.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    s1 = results_implicit.GetSolutionVectors()
    s2 = results_explicit_consistent_mass.GetSolutionVectors()
    s3 = results_explicit_lumped_mass.GetSolutionVectors()

    norm = lambda s: np.linalg.norm(s)

    assert norm(s1[:,:,-1]) > 3.
    assert norm(s2[:,:,-1]) > 3.
    assert norm(s3[:,:,-1]) > 3.
    assert norm(s1[:,:,-1]) < 3.3
    assert norm(s2[:,:,-1]) < 3.3
    assert norm(s3[:,:,-1]) < 3.3

    # results_implicit.Plot(configuration="deformed")
    # results_explicit_lumped_mass.Plot(configuration="deformed")


if __name__ == "__main__":
    nonlinear_elastodynamics(optimise=False)
    nonlinear_elastodynamics(optimise=True)