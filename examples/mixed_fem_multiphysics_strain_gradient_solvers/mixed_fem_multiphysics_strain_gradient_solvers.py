import numpy as np
from Florence import *


def strain_gradient_elastodynamics():
    """An example of strain gradient elasticity under explicit dynamics with penalty
        contact. The strain gradient model is based the couple stress (constrained Cosserat) theory
        for solids. The couple stress model in florence is implemented using standard C0 continuous
        elements using both penalty or Lagrange multiplier approaches
    """

    mesh = Mesh()
    mesh.HollowCircle(inner_radius=30, outer_radius=50,nrad=6,ncirc=120, element_type="quad")
    mesh.GetHighOrderMesh(p=2)

    mu = 1.0e5
    v = 0.4
    material = CoupleStressModel(2, mu=mu, lamb=2.*mu*v/(1-2.*v), eta=1000., kappa=1e-6, rho=1100.)


    def DirichletFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN
        return boundary_data

    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN
        mag=3.5e4
        d1 = np.ones(150)*mag
        d2 = np.zeros(time_step-150)
        d = np.concatenate((d1,d2))
        boundary_data[:,0,:] = d
        return boundary_data


    time_step = 2000
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # Contact formulation
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 80, 5e6)

    # Lagrange multiplier strain gradient formulation
    lagrange_multiplier_strain_gradient = CoupleStressFormulation(mesh,
        save_condensed_matrices=False, subtype="lagrange_multiplier")

    # Penalty strain gradient formulation
    penalty_strain_gradient = CoupleStressFormulation(mesh,
        save_condensed_matrices=False, subtype="penalty")

    fem_solver = FEMSolver(total_time=60.,
        number_of_load_increments=time_step,
        analysis_type="dynamic",
        analysis_nature="linear",
        print_incremental_log=True,
        include_physical_damping=True,
        damping_factor=2.,
        break_at_increment=400)


    penalty_results = fem_solver.Solve(formulation=penalty_strain_gradient, mesh=mesh,
            material=material, boundary_condition=boundary_condition,
            contact_formulation=contact_formulation)

    lagrange_multiplier_results = fem_solver.Solve(formulation=lagrange_multiplier_strain_gradient, mesh=mesh,
            material=material, boundary_condition=boundary_condition,
            contact_formulation=contact_formulation)


    # Uncomment to plot both results superimposed on top of each other
    # import matplotlib.pyplot as plt
    # figure = plt.figure()
    # penalty_results.Plot(configuration="deformed", quantity=0,
    #     plot_edges=False, figure=figure, show_plot=False)
    # lagrange_multiplier_results.Plot(configuration="deformed",
    #     quantity=0, plot_edges=True, colorbar=False, figure=figure, show_plot=False)
    # plt.show()


if __name__ == "__main__":
    strain_gradient_elastodynamics()

