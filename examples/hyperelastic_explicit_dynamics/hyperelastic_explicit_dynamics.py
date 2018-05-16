from __future__ import division
import os, sys
from Florence import *
from Florence.VariationalPrinciple import *


def explicit_dynamics_mechanics():
    """A hyperelastic explicit dynamics example using Mooney Rivlin model
        of a column under compression with cubic (p=3) hexahedral elements
    """

    mesh = Mesh()
    mesh.Parallelepiped(upper_right_front_point=(1,1,6),nx=3,ny=3,nz=18,element_type="hex")
    mesh.GetHighOrderMesh(p=3)
    ndim = mesh.InferSpatialDimension()

    material = ExplicitMooneyRivlin_0(ndim, mu1=1e5, mu2=1e5, lamb=200e5, rho=1000)

    def DirichletFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,:,:] = 0.

        return boundary_data

    def NeumannFuncDyn(mesh, time_step):

        boundary_flags = np.zeros((mesh.faces.shape[0]),dtype=np.uint8)
        boundary_data = np.zeros((mesh.faces.shape[0],3))
        mag = -8e3

        for i in range(mesh.faces.shape[0]):
            coord = mesh.points[mesh.faces[i,:],:]
            avg = np.sum(coord,axis=0)/mesh.faces.shape[1]
            if np.isclose(avg[2],mesh.points[:,2].max()):
                boundary_data[i,2] = mag
                boundary_flags[i] = True

        return boundary_flags, boundary_data


    time_step = 2000
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)


    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver( total_time=.5,
                            number_of_load_increments=time_step,analysis_type="dynamic",
                            analysis_subtype="explicit",
                            mass_type="lumped",
                            analysis_nature="nonlinear",
                            newton_raphson_tolerance=1e-5,
                            has_low_level_dispatcher=True,
                            print_incremental_log=True,
                            save_frequency=10)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # In-built plotter
    # solution.Plot(quantity=2,configuration='deformed')
    # Write to paraview
    # solution.WriteVTK("explicit_dynamics_mechanics",quantity=2)
    # Write to HDF5/MATLAB(.mat)
    # solution.WriteHDF5("explicit_dynamics_mechanics",compute_recovered_fields=False)



if __name__ == "__main__":
    explicit_dynamics_mechanics()






