import numpy as np
from Florence import *


def electro_hyperelastic_explicit_dynamics(recompute_sparsity_pattern=True, squeeze_sparsity_pattern=False):
    """A an example of multiphysics nonlinear explicit dynamics with nonlinear
        electro-hyperelastic material
    """


    mesh = Mesh()
    mesh.Parallelepiped(upper_right_front_point=(1,1,0.1),nx=12,ny=12,nz=2,element_type="hex")


    e0 = 8.8541e-12 # vacuum permittivity
    v = 0.35        # poisson's ratio
    # material = IsotropicElectroMechanics_101(3, mu=1.0e5, mu2=0., lamb=2.*1.0e5*v/(1-2.*v), eps_1=4.0*e0, rho=1100.)
    material = IsotropicElectroMechanics_105(3, mu1=1.0e5, mu2=0., lamb=2.*1.0e5*v/(1-2.*v), eps_1=4.5*e0, eps_2=4.8*e0, rho=1100.)


    def DirichletFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],4, time_step))+np.NAN

        # Fix the two base sides of the box
        X_0 = np.logical_and(np.isclose(mesh.points[:,0],0.),np.isclose(mesh.points[:,2],0.))
        boundary_data[X_0,:3,:] = 0.
        X_1 = np.logical_and(np.isclose(mesh.points[:,0],1.),np.isclose(mesh.points[:,2],0.))
        boundary_data[X_1,:3,:] = 0.

        # Closed circuit condition
        Z_bottom = np.isclose(mesh.points[:,2],0)
        boundary_data[Z_bottom,3,:] = 0.
        Z_top = np.isclose(mesh.points[:,2],0.05)
        boundary_data[Z_top,3,:] = 8e6*np.linspace(0,1,time_step)

        return boundary_data


    time_step = 500
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)

    formulation = DisplacementPotentialFormulation(mesh)
    fem_solver = FEMSolver(total_time=.5,
        number_of_load_increments=time_step,
        analysis_type="dynamic",
        analysis_subtype="explicit",
        mass_type="consistent",
        optimise=True,
        activate_explicit_multigrid=True,
        recompute_sparsity_pattern=recompute_sparsity_pattern,
        squeeze_sparsity_pattern=squeeze_sparsity_pattern,
        save_frequency=5)

    results = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition)

    # check results
    assert np.linalg.norm(results.GetSolutionVectors()[:,2,-1]) > 0.9
    assert np.linalg.norm(results.GetSolutionVectors()[:,2,-1]) < 1.2

    # Write results to paraview
    # results.WriteVTK("explicit_electro_elastodynamics", quantity=2)


if __name__ == "__main__":
    electro_hyperelastic_explicit_dynamics()
    electro_hyperelastic_explicit_dynamics(recompute_sparsity_pattern=False, squeeze_sparsity_pattern=False)
    electro_hyperelastic_explicit_dynamics(recompute_sparsity_pattern=False, squeeze_sparsity_pattern=True)
