import numpy as np
from Florence import *


def staggered_multiphysics_solver():
    """In this example we test the performance of staggered solver
        against the monolithic for the class of coupled coupled multiphysics
        problem namely nonlinear electromechanics. In staggered scheme, the
        solid geometry/mesh is updated only at increments whereas the electrical
        problem (which is an enhahnced version of the anisotropic Poisson's equation)
        is solved iteratively and fed back to the solid. Two approaches exists in florence
        for staggered schemes namely a traction based staggered approach and a
        displacement/potential based staggered scheme. In the traction based scheme,
        the mechanical/electrical problems interchange forces whereas in the
        displacement/potential based scheme they interchange updated solutions
        to each other. On the other hand, the monolithic approach treats the coupled
        problem as a single system and solves the entire problem iteratively.
        Both quasi-static and dynamic cases are presented in this example. Note that,
        in the spirit of the staggered scheme, the dynamic staggered scheme is
        semi-explicit in that the solid problem is solved explicitly while the
        electrical problem is solved in an implicit fashion. As a result the time step
        has to be significantly smaller. Also as a result of explicitness, the problem is
        substantially decoupled and only solution parameters are exhanged between
        mechanical and electrical problems however due to the nature of constitutive law
        both problems in turn generate internal forces acting on the other. In other words,
        the staggerd dynamic problem includes both traction and displacement/potential based
        schemes in a single approach
    """

    mesh = Mesh()
    # Create a 2D column
    mesh.Rectangle(upper_right_point=(1,10),nx=4,ny=40, element_type="quad")
    ndim = mesh.InferSpatialDimension()

    # Material constants
    e_vacuum = 8.8541e-12
    nu = 0.45

    # We use a hyperelastic MooneyRivlin model with ideal dielectric behaviour
    material = IsotropicElectroMechanics_108(ndim, mu1=1.0e5, mu2=0.,
        lamb=2.*1.0e5*nu/(1-2.*nu), eps_2=4.0*e_vacuum, rho=1200.)

    # Coupled electromechanics variational formulation
    formulation = DisplacementPotentialFormulation(mesh)


    def DirichletFunc(mesh):
        """We use a Dirichlet driven boundary condition for all cases
            that is for (staggered and monolithic) (static and dynamic).
            The boundary conditions here define an actuation type problem
            where electrical loads generate mechanical deformation
        """

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        # Closed circuit condition - electrical boundary condition
        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[X_0,2] = 0.
        X_1 = np.isclose(mesh.points[:,0],0.5)
        boundary_data[X_1,2] = 1e7

        # Fix columns base - no mechanical loading is applied
        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,:2] = 0.

        return boundary_data



    time_step = 50
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)


    # Quasi-static nonlinear monolithic solver
    static_monolithic_solver = FEMSolver(
        number_of_load_increments=time_step,
        analysis_nature="nonlinear",
        analysis_type="static",
        newton_raphson_tolerance=1e-5,
        optimise=True,
        )

    static_monolithic_solver_results = static_monolithic_solver.Solve(formulation=formulation,
        mesh=mesh, material=material, boundary_condition=boundary_condition)


    # Dynamic nonlinear monolithic solver with Newmark beta
    dynamic_monolithic_solver = FEMSolver(
        total_time=10.,
        number_of_load_increments=time_step,
        analysis_nature="nonlinear",
        analysis_type="dynamic",
        newton_raphson_tolerance=1e-5,
        optimise=True,
        )

    dynamic_monolithic_solver_results = dynamic_monolithic_solver.Solve(formulation=formulation,
        mesh=mesh, material=material, boundary_condition=boundary_condition)


    # Note that, whenever analysis_nature is prescribed as "linear" to the solver,
    # florence automatically generates a linearised constitutive law for the nonlinear material
    # model to facilitate incremental load stepping and avoid iterations. This models in essence,
    # can be termed as consistent incrementally linearised version of their nonlinear counterpart


    # Quasi-static nonlinear staggered solver - traction based
    static_staggered_solver_1 = FEMSolver(
        number_of_load_increments=time_step,
        analysis_nature="linear",
        analysis_type="static",
        linearised_electromechanics_solver="traction_based",
        newton_raphson_tolerance=1e-5,
        optimise=True,
        )

    static_staggered_solver_1_results = static_staggered_solver_1.Solve(formulation=formulation,
        mesh=mesh, material=material, boundary_condition=boundary_condition)


    # Quasi-static nonlinear staggered solver - potential based
    static_staggered_solver_2 = FEMSolver(
        number_of_load_increments=time_step,
        analysis_nature="linear",
        analysis_type="static",
        linearised_electromechanics_solver="potential_based",
        newton_raphson_tolerance=1e-5,
        optimise=True,
        )

    static_staggered_solver_2_results = static_staggered_solver_2.Solve(formulation=formulation,
        mesh=mesh, material=material, boundary_condition=boundary_condition)


    # Dynamic nonlinear staggered solver with leap frog explicit scheme
    dynamic_staggered_solver = FEMSolver(
        total_time=10.,
        number_of_load_increments=time_step*30,
        analysis_nature="linear",
        analysis_type="dynamic",
        mass_type="lumped",
        newton_raphson_tolerance=1e-5,
        optimise=True,
        )

    # Reset boundary condition before running second dynamic example
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    dynamic_staggered_solver_results = dynamic_staggered_solver.Solve(formulation=formulation,
        mesh=mesh, material=material, boundary_condition=boundary_condition)


    # Plot all solutions
    # static_monolithic_solver_results.Plot(configuration="deformed",quantity=0)
    # dynamic_staggered_solver_results.Plot(configuration="deformed",quantity=0)
    # static_staggered_solver_1_results.Plot(configuration="deformed",quantity=0)
    # static_staggered_solver_2_results.Plot(configuration="deformed",quantity=0)
    # dynamic_monolithic_solver.Plot(configuration="deformed",quantity=0)




if __name__ == "__main__":
    staggered_multiphysics_solver()