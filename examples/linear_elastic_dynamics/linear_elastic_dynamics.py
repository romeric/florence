import numpy as np
from Florence import *


def linear_elastic_dynamics():
    """An example of finite element simulation of linear elastodynamics with
        a 2D column of linear quad elements
    """

    mesh = Mesh()
    mesh.Rectangle(upper_right_point=(1,10), element_type="quad", nx=10, ny=100)
    ndim = mesh.InferSpatialDimension()

    v = 0.49
    mu = 1e5
    material = LinearElastic(ndim, mu=mu, lamb=2.*mu*v/(1-2.*v), density=1100)
    # Or use this material model alternatively
    # material = IncrementalLinearElastic(ndim, mu=mu, lamb=2.*mu*v/(1-2.*v), density=1100)


    def DirichletFuncDynamic(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],ndim, time_step))+np.NAN
        # FIX BASE OF COLUMN
        Y_0 = np.isclose(mesh.points[:,1],0.0)
        boundary_data[Y_0,:,:] = 0.
        # APLLY DIRICHLET DRIVEN LOAD TO TOP OF THE COLUMN X-DIRECTION
        Y_0 = np.isclose(mesh.points[:,1],mesh.points[:,1].max())
        boundary_data[Y_0,0,:] = np.linspace(0,2,time_step)

        return boundary_data

    time_step = 300
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFuncDynamic, mesh, time_step)

    formulation = DisplacementFormulation(mesh)

    # Solve with performance optimisation off
    implicit_fem_solver_1 = FEMSolver(total_time=60.,
        number_of_load_increments=time_step,
        analysis_nature="linear",
        analysis_type="dynamic",
        optimise=False,
        print_incremental_log=True)

    implicit_fem_solver_results_1 = implicit_fem_solver_1.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # Solve with performance optimisation on
    implicit_fem_solver_2 = FEMSolver(total_time=60.,
        number_of_load_increments=time_step,
        analysis_nature="linear",
        analysis_type="dynamic",
        optimise=False,
        print_incremental_log=True)

    implicit_fem_solver_results_2 = implicit_fem_solver_2.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # Get underlying solution vectors
    solution_1 = implicit_fem_solver_results_1.GetSolutionVectors()
    solution_2 = implicit_fem_solver_results_1.GetSolutionVectors()

    assert np.isclose(np.linalg.norm(solution_1),np.linalg.norm(solution_2), rtol=1e-5, atol=1e-5)

    # Write results to plot in paraview
    # implicit_fem_solver_results_1.WriteVTK("linear_dynamic_results_1", quantity=1)
    # implicit_fem_solver_results_2.WriteVTK("linear_dynamic_results_2", quantity=1)


if __name__ == "__main__":
    linear_elastic_dynamics()
