from Florence import *
import numpy as np


def nonlinear_electroelastodynamics():
    """This example checks all variants of nonlinear electromechanics formulation i.e.
        (linearised/nonlinear) static/implicit dynamics/explicit dynamics
    """

    mesh = Mesh()
    mesh.Parallelepiped(upper_right_front_point=(1,1,0.001),nx=10,ny=10,nz=1, element_type="hex")

    mu = 5.0e4
    mu1 = mu
    mu2 = mu
    eps_2 = 4.0*8.8541e-12
    v = 0.4
    lamb = 2.*mu*v/(1-2.*v)
    material = IsotropicElectroMechanics_108(3, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1200.)

    formulation = DisplacementPotentialFormulation(mesh)


    def dirichlet_function(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],4))+np.NAN

        Z_0 = np.logical_and(np.isclose(mesh.points[:,0],0.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,1],0.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,0],1),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,1],1),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.

        Z_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Z_0,3] = 0.

        Z_0 = np.isclose(mesh.points[:,2],.001)
        boundary_data[Z_0,3] = 9e3

        return boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(dirichlet_function, mesh)

    nonlinear_static_solver = FEMSolver(total_time=60.,
        number_of_load_increments=25,
        analysis_nature="nonlinear",
        analysis_type="static",
        newton_raphson_tolerance=1e-5,
        newton_raphson_solution_tolerance=1e-11,
        optimise=True,
        print_incremental_log=True,
        )

    nonlinear_static_results = nonlinear_static_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    nonlinear_dynamic_solver = FEMSolver(total_time=60.,
        number_of_load_increments=250,
        analysis_nature="nonlinear",
        analysis_type="dynamic",
        newton_raphson_tolerance=1e-5,
        newton_raphson_solution_tolerance=1e-11,
        optimise=True,
        print_incremental_log=True,
        )

    nonlinear_dynamic_results = nonlinear_dynamic_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    # boundary_condition.__reset_state__()
    # boundary_condition.SetDirichletCriteria(dirichlet_function, mesh)

    # nonlinear_dynamic_solver_exp = FEMSolver(total_time=6.,
    #     number_of_load_increments=200000,
    #     save_frequency=200000,
    #     analysis_nature="nonlinear",
    #     analysis_type="dynamic",
    #     analysis_subtype="explicit",
    #     newton_raphson_tolerance=1e-5,
    #     newton_raphson_solution_tolerance=1e-11,
    #     optimise=True,
    #     print_incremental_log=True,
    #     )

    # nonlinear_dynamic_results_exp = nonlinear_dynamic_solver_exp.Solve(formulation=formulation, mesh=mesh,
    #         material=material, boundary_condition=boundary_condition)


    boundary_condition.__reset_state__()
    boundary_condition.SetDirichletCriteria(dirichlet_function, mesh)

    linear_static_solver = FEMSolver(total_time=60.,
        number_of_load_increments=250,
        analysis_nature="linear",
        analysis_type="static",
        newton_raphson_tolerance=1e-5,
        newton_raphson_solution_tolerance=1e-11,
        optimise=True,
        print_incremental_log=True,
        )

    linear_static_results = linear_static_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    boundary_condition.__reset_state__()
    boundary_condition.SetDirichletCriteria(dirichlet_function, mesh)

    linear_dynamic_solver = FEMSolver(total_time=60.,
        number_of_load_increments=1000,
        analysis_nature="linear",
        analysis_type="dynamic",
        newton_raphson_tolerance=1e-5,
        newton_raphson_solution_tolerance=1e-11,
        optimise=True,
        print_incremental_log=True,
        break_at_increment=100,
        )

    linear_dynamic_results = linear_dynamic_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)


    s1 = nonlinear_static_results.GetSolutionVectors()
    s2 = nonlinear_dynamic_results.GetSolutionVectors()
    # s3 = nonlinear_dynamic_results_exp.GetSolutionVectors()
    s4 = linear_static_results.GetSolutionVectors()
    s5 = linear_dynamic_results.GetSolutionVectors()

    norm = lambda x: np.linalg.norm(x[:,2,-1])
    assert norm(s1) > 0.13 and norm(s1) < 0.15
    assert norm(s2) > 0.13 and norm(s2) < 0.15
    assert norm(s4) > 0.13 and norm(s4) < 0.15



if __name__ == "__main__":
    nonlinear_electroelastodynamics()