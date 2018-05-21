from Florence import *
from Florence.VariationalPrinciple import *


def crash_analysis():
    """ Car crash analysis in a simplified 2D car geometry with hyperelastic
        explicit dynamics solver using explicit penalty contact formulation
    """

    mesh = Mesh()
    mesh.ReadGmsh("Car2D.msh",element_type="quad")
    mesh.points /=1000.

    mu = 1.0e6
    mu1 = mu
    mu2 = 0.
    v = 0.495
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(2, mu1=mu1, mu2=mu2, lamb=lamb, rho=8000.)


    def DirichletFunc(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN
        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[:,1,:] = 0.
        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN

        mag = 5e5
        n = 3000
        stage0 = np.ones(n)*mag
        stage1 = np.zeros(time_step-n)
        full_stage = np.concatenate((stage0,stage1))
        boundary_data[:,0,:] = full_stage

        return boundary_data


    time_step = 6000
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    formulation = DisplacementFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh,
        np.array([-1.,0.]), # unit vector specifying the direction of normal contact
        180.,               # normal gap [distance of rigid body from the deformable object]
        1e7                 # penalty parameter kappa
        )
    fem_solver = FEMSolver(total_time=9,
        number_of_load_increments=time_step,
        analysis_type="dynamic",
        analysis_subtype="explicit",
        mass_type="lumped",
        analysis_nature="nonlinear",
        optimise=True,
        save_frequency=50)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, contact_formulation=contact_formulation)

    # Write results to vtk file
    solution.WriteVTK("crash_analysis_results", quantity=0)


if __name__ == "__main__":
    crash_analysis()