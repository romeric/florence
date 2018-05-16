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



def ElectroExpDyn():

    # n=24
    # n=48
    # n=100
    n=8
    mesh = Mesh()
    mesh.Parallelepiped(upper_right_front_point=(1,1,0.1),nx=n,ny=n,nz=2,element_type="hex")
    # mesh.Parallelepiped(upper_right_front_point=(1,1,0.1),nx=n,ny=n,nz=4,element_type="hex")
    # mesh.Parallelepiped(upper_right_front_point=(1,1,0.01),nx=n,ny=n,nz=1,element_type="hex")
    # mesh.GetHighOrderMesh(p=2)
    # mesh.SimplePlot()
    # exit()

    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.35 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = IsotropicElectroMechanics_108(3, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1000.)
    # material = IsotropicElectroMechanics_108(3, mu1=1., mu2=0.5, lamb=2.5, eps_2=1.2e-3, rho=2.)
    print(mesh.Bounds)


    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],4))+np.NAN

        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[X_0,:3] = 0.
        X_0 = np.isclose(mesh.points[:,0],1.)
        boundary_data[X_0,:3] = 0.

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,3] = 0.
        # X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[X_0,3] = 3e6
        # X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max()/2.)
        # boundary_data[X_0,3] = 2e6
        X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max()) #
        boundary_data[X_0,3] = 4e5

        # X_0 = np.isclose(mesh.points[:,2],0)
        # boundary_data[X_0,:] = 0.
        # X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[X_0,2] = 0.2

        return boundary_data



    def DirichletFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],4, time_step))+np.NAN

        X_0 = np.isclose(mesh.points[:,0],0)
        boundary_data[X_0,:3,:] = 0.
        X_0 = np.isclose(mesh.points[:,0],1.)
        boundary_data[X_0,:3,:] = 0.

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,3,:] = 0.
        # X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[X_0,3,:] = 3e6*np.ones(time_step)
        X_0 = np.isclose(mesh.points[:,2],0.05)
        boundary_data[X_0,3,:] = 2e6*np.ones(time_step)

        # X_0 = np.isclose(mesh.points[:,2],0)
        # boundary_data[X_0,:] = 0.
        # X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[X_0,2] = 0.2

        return boundary_data

    # time_step = 500
    time_step = 3000
    # time_step = 250
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None

    formulation = DisplacementPotentialFormulation(mesh)
    fem_solver = FEMSolver(total_time=.5,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3,
        has_low_level_dispatcher=False,
        # has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=5,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver)

    # solution.WriteVTK("ElectroExpDyn3", quantity=2, interpolation_degree=5)
    # solution.WriteVTK("ElectroImpDyn", quantity=2, interpolation_degree=5)
    # solution.WriteVTK("ElectroStatic", quantity=2, interpolation_degree=5)



def Pipe_Exp():



    nalong = 6

    mesh = Mesh()
    mesh.HollowArc(inner_radius=10, outer_radius=30, element_type="quad", nrad=1, ncirc=6)

    mesh2 = Mesh()
    mesh2.Rectangle(lower_left_point=(20,-90), upper_right_point=(30,0), element_type="quad", nx=1, ny=nalong)
    mesh += mesh2



    mesh2 = Mesh()
    mesh2.Rectangle(lower_left_point=(-90,20), upper_right_point=(0,30), element_type="quad", nx=nalong, ny=1)
    mesh += mesh2

    mesh.points[:,0] += 90
    mesh.points[:,1] += 90

    mesh2 = deepcopy(mesh)
    mesh2.points[:,0] *=-1.
    mesh += mesh2

    mesh2 = deepcopy(mesh)
    mesh2.points[:,1] *=-1.
    mesh += mesh2

    mesh.points[:,0] += 120
    mesh.points[:,1] += 120

    mesh.Extrude(nlong=40,length=2400)
    mesh.points /=240.
    print mesh.Bounds

    # mesh.SimplePlot()
    # exit()

    # material = MooneyRivlin_0(mesh.InferSpatialDimension(), mu1=1e5, mu2=1e5, lamb=5e5, rho=1000)
    material = ExplicitMooneyRivlin_0(mesh.InferSpatialDimension(), mu1=1e5, mu2=1e5, lamb=200e5, rho=1000)
    # material.mu = 2e5
    # material.GetYoungsPoissonsFromLameParameters()
    # print material.nu
    # exit()


    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,:] = 0.
        X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        boundary_data[X_0,2] = -3.2

        return boundary_data

    def DirichletFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN

        X_0 = np.isclose(mesh.points[:,2],0)
        boundary_data[X_0,:,:] = 0.
        X_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        boundary_data[X_0,2,:] = -np.linspace(0,3.2,time_step)

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):

        # boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN
        # mag = 7e3
        # X_0 = np.isclose(mesh.points[:,0],1)
        # boundary_data[X_0,0,:] = np.linspace(0,mag,time_step)
        # return boundary_data

        boundary_flags = np.zeros((mesh.faces.shape[0]),dtype=np.uint8)
        boundary_data = np.zeros((mesh.faces.shape[0],3))
        # mag = 6e-3
        mag = 3e3
        # mag = 9e3

        for i in range(mesh.faces.shape[0]):
            coord = mesh.points[mesh.faces[i,:],:]
            avg = np.sum(coord,axis=0)/mesh.faces.shape[1]
            # print avg
            # if np.isclose(avg[0],1):
            if np.isclose(avg[2],mesh.points[:,2].max()):
            # if np.isclose(avg[2],100):
                boundary_data[i,0] = -mag
                boundary_flags[i] = True

        return boundary_flags, boundary_data


    time_step = 1000
    # time_step = 58000
    # time_step = 30
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    # boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)


    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(total_time=.1,number_of_load_increments=time_step,analysis_type="dynamic",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        # mass_type="lumped",
        mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-5, has_low_level_dispatcher=False,
        print_incremental_log=True, parallelise=False, break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition, solver=solver)

    # solution.CurvilinearPlot()
    # solution.WriteVTK("MechCheck", quantity=0)

    solution.sol = solution.sol[:,:,::10]
    # solution.sol = solution.sol[:,:,::2]
    # solution.WriteVTK("MechCheck_Exp", quantity=2, interpolation_degree=5)
    solution.WriteVTK("Pipe_Exp", quantity=2, interpolation_degree=5)



def WPlate():

    mesh = Mesh()
    mesh.Parallelepiped(element_type="hex",nx=60,ny=60,nz=1, upper_right_front_point=(100,100,1)) ##
    # mesh.Parallelepiped(element_type="hex",nx=80,ny=80,nz=1, upper_right_front_point=(100,100,1)) ##
    # mesh.Parallelepiped(element_type="hex",nx=40,ny=40,nz=1, upper_right_front_point=(100,100,1)) ##
    # mesh.Parallelepiped(element_type="hex",nx=20,ny=20,nz=1, upper_right_front_point=(100,100,1)) ##
    # mesh.Parallelepiped(element_type="hex",nx=6,ny=6,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=16,ny=16,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=10,ny=10,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=16,ny=16,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=12,ny=12,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=12,ny=12,nz=1, upper_right_front_point=(100,100,1))
    # mesh.Parallelepiped(element_type="hex",nx=2,ny=2,nz=1, upper_right_front_point=(100,100,1))
    # print mesh.Bounds
    # mesh.GetHighOrderMesh(p=2)
    # mesh = mesh.ConvertToLinearMesh()
    # mesh.ChangeType()
    # print mesh.points.shape[0]*4, mesh.points.shape[0]
    # exit()
    # mesh.SimplePlot()

    # pp = PostProcess(3,3)
    # pp.CurvilinearPlot(mesh,plot_edges=True)
    # exit()


    # exit()

    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.4 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)

    # print(mesh.Bounds)
    # print(mesh.points)
    # exit()


    # material = IsotropicElectroMechanics_101(ndim, mu=mu1, lamb=lamb, eps_1=eps_2, mus=mus, lambs=lambs)
    material = IsotropicElectroMechanics_108(3, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1000.)

    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],4))+np.NAN

        Z_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Z_0,3] = 0.
        Z_0 = np.isclose(mesh.points[:,2],1.)
        # boundary_data[Z_0,3] = 6.e7
        boundary_data[Z_0,3] = 1.e8

        # Killer Deformation symmetric
        Z_0 = np.logical_and(np.isclose(mesh.points[:,0],0.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,1],0.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.

        # X_0 = np.isclose(mesh.points[:,0],50.)
        # boundary_data[X_0,0] = 0.
        # X_0 = np.isclose(mesh.points[:,1],50.)
        # boundary_data[X_0,1] = 0.

        Z_0 = np.logical_and(np.isclose(mesh.points[:,0],100.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,1],100.),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.

        return boundary_data

    time_step = 4000
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFunc, mesh, time_step)

    formulation = DisplacementPotentialFormulation(mesh)
    # solver = LinearSolver(dont_switch_solver=True)
    # solver = LinearSolver(linear_solver = 'direct',linear_solver_type='mumps',dont_switch_solver=True)
    solver = LinearSolver(linear_solver = 'direct',linear_solver_type='umfpack',dont_switch_solver=True)
    # solver = LinearSolver(linear_solver = 'direct',dont_switch_solver=True)

    fem_solver = FEMSolver(total_time=.1, number_of_load_increments=time_step,
        analysis_nature="nonlinear", analysis_type = "dynamic", analysis_subtype="explicit",mass_type="lumped",
        save_frequency=10,
        newton_raphson_tolerance=1e-4, maximum_iteration_for_newton_raphson=200,
        has_low_level_dispatcher=False, print_incremental_log=True)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition, solver=solver)


    solution.WriteVTK(PWD(__file__)+"/WPlate",quantity=2, interpolation_degree=2)



def DiscExp(p=2):

    mesh = Mesh()
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=12, ncirc=12)
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=30, ncirc=30)
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=30, ncirc=60)
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=45, ncirc=90)
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=120, ncirc=120)
    # mesh.Cylinder(radius=20,length=1,nlong=1, nrad=90, ncirc=150)
    # exit()

    radius = 60.
    # mesh.Arc(radius=radius, element_type="quad", nrad=8, ncirc=8)
    # mesh.Arc(radius=radius, element_type="quad", nrad=30, ncirc=30)
    # mesh.Arc(radius=radius, element_type="quad", nrad=50, ncirc=75)
    mesh.Arc(radius=radius, element_type="quad", nrad=60, ncirc=90)
    mesh.Extrude(nlong=1,length=1.)
    mesh.GetHighOrderMesh(p=2)
    # mesh.SimplePlot()
    # print mesh.points.shape[0]*4
    # exit()


    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.3 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)

    print(mesh.Bounds)
    # print(mesh.points)
    # exit()
    print mesh.elements.shape


    # material = IsotropicElectroMechanics_101(ndim, mu=mu1, lamb=lamb, eps_1=eps_2, mus=mus, lambs=lambs)
    material = IsotropicElectroMechanics_108(3, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1100.)

    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],4))+np.NAN

        r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
        Z_0 = np.logical_and(np.isclose(r,radius),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.

        Z_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Z_0,3] = 0.
        Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[Z_0,3] = 6.5e7
        boundary_data[Z_0,3] = 4.e7 #
        # boundary_data[Z_0,3] = 1.e7

        Z_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Z_0,0] = 0.
        Z_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Z_0,1] = 0.
        Z_0 = np.logical_and(np.isclose(mesh.points[:,0],0.),np.isclose(mesh.points[:,1],0.))
        boundary_data[Z_0,1] = 0.


        Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max()/2.)
        boundary_data[Z_0,3] = 2.e7

        return boundary_data


    # def DirichletFunc(mesh, time_step):

    #     boundary_data = np.zeros((mesh.points.shape[0],4, time_step))+np.NAN

    #     r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
    #     Z_0 = np.logical_and(np.isclose(r,100.),np.isclose(mesh.points[:,2],0.))
    #     # Z_0 = np.logical_and(np.isclose(r,20.),np.isclose(mesh.points[:,2],0.))
    #     boundary_data[Z_0,:3,:] = 0.

    #     Z_0 = np.isclose(mesh.points[:,2],0.)
    #     boundary_data[Z_0,3,:] = 0.
    #     Z_0 = np.isclose(mesh.points[:,2],1.)

    #     # mag = 5.5e7
    #     mag = 4.9e7
    #     print mag
    #     d1 = mag/4.*np.linspace(0,1,100)
    #     d2 = mag/4.*np.ones(100)
    #     d3 = np.linspace(mag/4.,mag/2.,100)
    #     d4 = mag/2.*np.ones(100)
    #     d5 = np.linspace(mag/2.,3.*mag/4.,100)
    #     d6 = 3.*mag/4.*np.ones(200)
    #     d7 = np.linspace(3.*mag/4.,0.9*mag,100)
    #     d8 = 0.9*mag*np.ones(100)
    #     d9 = np.linspace(0.9*mag,mag,100)
    #     d = np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8,d9))
    #     # d = np.concatenate((d1,d2,d3,d4,d5,d6,d7,d8))
    #     # import matplotlib.pyplot as plt
    #     # plt.plot(np.arange(time_step),d)
    #     # plt.show()
    #     # exit()
    #     # print(d8)
    #     boundary_data[Z_0,3,:] = d

    #     Z_0 = np.isclose(mesh.points[:,0],0.)
    #     boundary_data[Z_0,0,:] = 0.
    #     Z_0 = np.isclose(mesh.points[:,1],0.)
    #     boundary_data[Z_0,1,:] = 0.

    #     return boundary_data

    # time_step = 15000
    # time_step = 2000
    time_step = 10000
    boundary_condition = BoundaryCondition()
    # boundary_condition.SetDirichletCriteria(DirichletFunc, mesh, time_step)
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementPotentialFormulation(mesh)


    fem_solver = FEMSolver(total_time=10., number_of_load_increments=time_step, analysis_nature="nonlinear", analysis_type="dynamic",
        analysis_subtype="explicit",
        save_frequency=50,
        newton_raphson_tolerance=1e-4, maximum_iteration_for_newton_raphson=200, include_physical_damping=False, damping_factor=1.,
        has_low_level_dispatcher=True, print_incremental_log=True, parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # print solution.sol.shape
    # solution.WriteVTK(PWD(__file__)+"/DiscExp4",quantity=2, interpolation_degree=3)
    # solution.WriteHDF5(PWD(__file__)+"/DiscExp4",compute_recovered_fields=False)

    # import matplotlib.pyplot as plt
    # plt.plot(solution.sol[:,2,:].T,solution.sol[:,3,:].T)
    # plt.show()



def HollowDiscExp(p=2):

    # radius = 60.
    radius = 120.
    mesh = Mesh()
    # mesh.HollowCylinder(inner_radius=5, outer_radius=20,length=1,nlong=1, nrad=12, ncirc=24)


    # mesh.Arc(radius=radius, element_type="quad", nrad=8, ncirc=8)
    # mesh.HollowArc(inner_radius=10, outer_radius=radius, nrad=12, ncirc=24, element_type="hex", end_angle=np.pi/2)
    # mesh.HollowArc(inner_radius=10, outer_radius=radius, nrad=24, ncirc=48, element_type="hex", end_angle=np.pi/2)
    # mesh.HollowArc(inner_radius=10, outer_radius=radius, nrad=60, ncirc=90, element_type="hex", end_angle=np.pi/2)
    mesh.HollowArc(inner_radius=10, outer_radius=radius, nrad=180, ncirc=120, element_type="hex", end_angle=np.pi/2)
    mesh.Extrude(nlong=1,length=1.)
    # mesh.GetHighOrderMesh(p=4)
    # mesh.SimplePlot()
    # print(mesh.Bounds)
    # print mesh.points.shape[0]*4
    # exit()


    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.3 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)

    # print(mesh.Bounds)
    # print(mesh.points)
    # exit()
    print mesh.elements.shape


    # material = IsotropicElectroMechanics_101(ndim, mu=mu1, lamb=lamb, eps_1=eps_2, mus=mus, lambs=lambs)
    material = IsotropicElectroMechanics_108(3, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1100.)

    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],4))+np.NAN

        r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
        Z_0 = np.logical_and(np.isclose(r,r.min()),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.
        Z_0 = np.logical_and(np.isclose(r,radius),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3] = 0.

        Z_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Z_0,3] = 0.
        Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[Z_0,3] = 6.5e7
        boundary_data[Z_0,3] = 4.e7 #
        # boundary_data[Z_0,3] = 1.e7

        Z_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Z_0,0] = 0.
        Z_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Z_0,1] = 0.


        # Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max()/2.)
        # boundary_data[Z_0,3] = 2.e7

        return boundary_data


    def DirichletFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],4, time_step))+np.NAN

        r = np.sqrt(mesh.points[:,0]**2 + mesh.points[:,1]**2)
        Z_0 = np.logical_and(np.isclose(r,r.min()),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3,:] = 0.
        Z_0 = np.logical_and(np.isclose(r,radius),np.isclose(mesh.points[:,2],0.))
        boundary_data[Z_0,:3,:] = 0.

        Z_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Z_0,3,:] = 0.
        Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max())
        # boundary_data[Z_0,3] = 6.5e7
        boundary_data[Z_0,3,:] = 4.e7 #
        # boundary_data[Z_0,3] = 1.e7

        Z_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Z_0,0,:] = 0.
        Z_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Z_0,1,:] = 0.


        # Z_0 = np.isclose(mesh.points[:,2],mesh.points[:,2].max()/2.)
        # boundary_data[Z_0,3] = 2.e7

        return boundary_data

    # import pyamg
    # exit()
    time_step = 2000
    # time_step = 4000
    # time_step = 10000
    print time_step
    boundary_condition = BoundaryCondition()
    # boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)

    formulation = DisplacementPotentialFormulation(mesh)


    fem_solver = FEMSolver(total_time=10., number_of_load_increments=time_step, analysis_nature="nonlinear", analysis_type="dynamic",
        # analysis_subtype="explicit",
        analysis_subtype="implicit",
        save_frequency=10,
        newton_raphson_tolerance=1e-4, maximum_iteration_for_newton_raphson=200, include_physical_damping=False, damping_factor=1.,
        has_low_level_dispatcher=True, print_incremental_log=True, parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition)

    # solution.WriteVTK(PWD(__file__)+"/HollowDiscThin",quantity=2, interpolation_degree=3)
    # solution.WriteVTK(PWD(__file__)+"/HollowDisc",quantity=2, interpolation_degree=3)
    # solution.WriteHDF5(PWD(__file__)+"/HollowDisc",compute_recovered_fields=False)


def Contact():


    mesh = Mesh()
    mesh.Rectangle(element_type="quad",upper_right_point=(1,10),nx=2,ny=10)
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # print mesh.Bounds

    # wall situtated at x=5
    L = 1+1e-6
    normal = np.array([-1.,0.])
    # normal = np.array([1.,0.])
    k = 1e5
    ndim = mesh.points.shape[1]


    boundary_surface = mesh.edges
    surfNodes_no, surfNodes_idx, surfNodes_inv = np.unique(boundary_surface, return_index=True, return_inverse=True)
    surfNodes = mesh.points[surfNodes_no,:]
    gNx = surfNodes.dot(normal) + L

    # contactNodes = np.where(gNx<=0.)
    # contactNodes_idx = gNx <= 0
    contactNodes_idx = gNx < 1e-6
    # contactNodes = mesh.points[contactNodes_idx,:]
    contactNodes = surfNodes[contactNodes_idx,:]
    # print surfNodes
    # print gNx
    # print surfNodes_inv
    # print surfNodes_inv[surfNodes_idx[contactNodes_idx]]
    # print np.where(contactNodes_idx==True)[0]
    # print surfNodes_inv[np.where(contactNodes_idx==True)[0]]
    # print surfNodes_idx
    # print surfNodes_idx[contactNodes_idx]
    contactNodes_global_idx = surfNodes_no[contactNodes_idx].astype(np.int64)
    print contactNodes_global_idx
    # print gNx[gNx<=0]
    # print contactNodes_idx
    # print contactNodes_idx.shape, mesh.points.shape, gNx.shape, surfNodes.shape

    T_contact = np.zeros((mesh.points.shape[0]*ndim,1))
    # t_local = k*gNx*normal
    # print t_local
    # print contactNodes.shape, gNx.shape
    normal_gap = gNx[contactNodes_idx]
    for node in range(contactNodes.shape[0]):
        t_local = k*normal_gap[node]*normal
        T_contact[contactNodes_global_idx[node]*ndim:(contactNodes_global_idx[node]+1)*ndim,0] += t_local
        # for iterator in range(ndim):
            # T_contact[contactNodes_global_idx[node]*ndim+iterator,0] += t_local[iterator::ndim]
            # T_contact[contactNodes_global_idx[node]*ndim+iterator,0] += t_local
    print T_contact


def MechExpDyn2D():


    mesh = Mesh()
    # mesh.Rectangle(upper_right_point=(1,10),nx=2,ny=20,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=4,ny=40,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=8,ny=80,element_type="quad")
    mesh.Rectangle(upper_right_point=(1,10),nx=16,ny=160,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=8*16,ny=8*160,element_type="quad")

    # mesh.points[:,0] += 4
    # mesh.WriteVTK("Wall3")
    # exit()
    # mesh.Parallelepiped(upper_right_front_point=(1,10,25),nx=2,ny=3,nz=4,element_type="hex")
    # mesh.Parallelepiped(upper_right_front_point=(1,10,25),nx=30,ny=30,nz=30,element_type="hex")
    # mesh.Parallelepiped(upper_right_front_point=(1,10,25),nx=40,ny=50,nz=60,element_type="tet")
    # mesh.ReadGmsh("car_notyres2.msh",element_type="tet")
    # mesh.GetHighOrderMesh(p=3)


    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.4995 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(2, mu1=mu1, mu2=mu2, lamb=lamb, rho=1000.)
    # material = ExplicitMooneyRivlin_0(mesh.InferSpatialDimension(), mu1=1., mu2=0, lamb=2., rho=1.)
    print(mesh.Bounds)


    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN

        X_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[X_0,:] = 0.

        # X_0 = np.isclose(mesh.points[:,1],10)
        # boundary_data[X_0,0] = 5.

        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN

        X_0 = np.isclose(mesh.points[:,1],10)
        boundary_data[X_0,0] = 9.e3

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN

        mag = 8.e3
        d1 = np.ones(500)*mag
        d2 = np.zeros(time_step-500)
        d = np.concatenate((d1,d2))
        X_0 = np.isclose(mesh.points[:,1],10.0)
        boundary_data[X_0,0,:] = d

        return boundary_data


    # time_step = 5000
    time_step = 40000
    # time_step = 250
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    # boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None
    # exit()

    formulation = DisplacementFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 3., 2e7)
    # contact_formulation = None
    fem_solver = FEMSolver(total_time=5,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3,
        # has_low_level_dispatcher=False,
        has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=200,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver, contact_formulation=contact_formulation)

    solution.WriteVTK("MechExpDyn2D", quantity=21, interpolation_degree=5)



def ElectroExpDyn2D():


    mesh = Mesh()
    # mesh.Rectangle(upper_right_point=(1,10),nx=2,ny=20,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=4,ny=40,element_type="quad")
    mesh.Rectangle(upper_right_point=(1,10),nx=8,ny=80,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=16,ny=160,element_type="quad")


    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.49 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = IsotropicElectroMechanics_108(2, mu1=mu1, mu2=mu2, lamb=lamb, rho=1000.)
    # material = ExplicitMooneyRivlin_0(mesh.InferSpatialDimension(), mu1=1., mu2=0, lamb=2., rho=1.)
    print(mesh.Bounds)


    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        X_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[X_0,:2] = 0.

        X_0 = np.isclose(mesh.points[:,0],.5)
        boundary_data[X_0,2] = 0.

        # X_0 = np.isclose(mesh.points[:,1],10)
        # boundary_data[X_0,0] = 5.

        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN

        X_0 = np.isclose(mesh.points[:,0],1.)
        boundary_data[X_0,0] = 9.e-3

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN

        mag = 8.e3
        d1 = np.ones(500)*mag
        d2 = np.zeros(time_step-500)
        d = np.concatenate((d1,d2))
        X_0 = np.isclose(mesh.points[:,1],10.0)
        boundary_data[X_0,0,:] = d

        return boundary_data


    time_step = 10000
    # time_step = 40000
    # time_step = 250
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    # boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None
    # exit()

    formulation = DisplacementPotentialFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 3., 2e7)
    # contact_formulation = None
    fem_solver = FEMSolver(total_time=5,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3,
        # has_low_level_dispatcher=False,
        has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=200,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver, contact_formulation=contact_formulation)

    solution.WriteVTK("ElectroExpDyn2D", quantity=21, interpolation_degree=5)




def SphereTranslation():


    mesh = Mesh()
    # mesh.Circle(radius=1.,ncirc=30,nrad=7,element_type="quad")
    # mesh.HollowCircle(inner_radius=0.4, outer_radius=1.,ncirc=20,nrad=3,element_type="quad")
    # mesh.HollowCircle(inner_radius=0.4, outer_radius=1.,ncirc=60,nrad=6,element_type="quad")
    # mesh.HollowCircle(inner_radius=0.4, outer_radius=1.,ncirc=100,nrad=10,element_type="quad")
    # mesh.HollowCircle(inner_radius=0.4, outer_radius=1.,ncirc=120,nrad=12,element_type="quad")
    mesh.HollowCircle(inner_radius=0.4, outer_radius=1.,ncirc=480,nrad=48,element_type="quad")

    mesh.GetHighOrderMesh(p=5)

    # mesh.SimplePlot()

    # mesh.Rectangle(lower_left_point=(0,-10), upper_right_point=(1,10),nx=16,ny=160,element_type="quad")
    # mesh.points[:,0] += 2
    # mesh.WriteVTK("Wall")
    # exit()

    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.495 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(2, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1000.)
    print(mesh.Bounds)
    # exit()


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN
        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN
        return boundary_data

        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN

        mag = 2.1e4
        print(mag)
        # exit()
        d1 = np.ones(5000)*mag
        d2 = np.zeros(time_step-5000)
        d = np.concatenate((d1,d2))
        X_0 = np.isclose(mesh.points[:,0],-1.0)
        boundary_data[X_0,0,:] = d

        return boundary_data


    # time_step = 12000
    time_step = 30000
    # time_step = 50000
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None
    print 55

    formulation = DisplacementFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 2., 2e7)
    # contact_formulation = None
    fem_solver = FEMSolver(total_time=4,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3,
        # has_low_level_dispatcher=False,
        has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=50,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=10000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver, contact_formulation=contact_formulation)

    # solution.WriteVTK("SphereTranslation", quantity=0, interpolation_degree=5)
    # solution.WriteVTK("SphereTranslation3", quantity=21, interpolation_degree=5)
    # solution.WriteVTK("SphereTranslation_", quantity=21, interpolation_degree=0)
    # solution.WriteHDF5("SphereTranslation2")
    # mesh.WriteHDF5("MeshSphereTranslation")


def SphereMerge():

    # mesh = Mesh()
    # mesh.Parallelepiped(lower_left_rear_point=(2,-3,-3),upper_right_front_point=(2.5,3,3))
    # mesh.WriteVTK("Wall3D")
    # exit()

    nlong=10
    length=1
    steps = 150
    mesh = Mesh()
    mesh.ReadHDF5("MeshSphereTranslation")
    mesh2D = deepcopy(mesh)
    mesh.Extrude(length=length,nlong=nlong)
    # mesh.SimplePlot()
    # exit()
    from scipy.io import loadmat
    sol = loadmat(PWD(__file__)+"/SphereTranslation2")['Solution']
    sol = sol[:,:,:steps]
    print sol.shape


    print "Starting extrusion"
    # mesh3D = deepcopy(mesh)
    sol3D = np.zeros((mesh.points.shape[0],3,sol.shape[2]))
    for i in range(sol.shape[2]):
        dum = deepcopy(mesh2D)
        dum.points += sol[:,:2,i]
        dum.Extrude(length=length,nlong=nlong)
        sol3D[:,:3,i] = dum.points - mesh.points
        print i
    # exit()

    pp = PostProcess(3,3)
    pp.SetMesh(mesh)
    pp.SetSolution(sol3D)
    pp.SetFormulation(DisplacementFormulation(mesh))

    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    v = 0.495 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(3, mu1=mu1, mu2=mu2, lamb=lamb, rho=1000.)
    pp.SetMaterial(material)
    pp.SetFEMSolver(FEMSolver(number_of_load_increments=steps, analysis_nature="nonlinear"))
    pp.WriteVTK(PWD(__file__)+"/SphereTranslation3D",quantity=41)


def ElectroExpDyn2D2():


    mesh = Mesh()
    # mesh.Rectangle(upper_right_point=(1,10),nx=2,ny=1,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=2,ny=20,element_type="quad")
    mesh.Rectangle(upper_right_point=(1,10),nx=4,ny=40,element_type="quad")
    # mesh.Rectangle(upper_right_point=(1,10),nx=8,ny=80,element_type="quad")

    # mesh.points[:,0] += 2
    # mesh.WriteVTK("Wall")
    # exit()
    # mesh.PlotMeshNumbering()

    e0 = 8.8541e-12
    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    eps_2 = 4.0*e0
    v = 0.495 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = IsotropicElectroMechanics_108(2, mu1=mu1, mu2=mu2, lamb=lamb, eps_2=eps_2, rho=1000.)
    print(mesh.Bounds)


    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        X_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[X_0,:2] = 0.

        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[X_0,2] = 0.
        # X_0 = np.isclose(mesh.points[:,0],0.5)
        # # X_0 = np.isclose(mesh.points[:,0],1.)
        # boundary_data[X_0,2] = 1e7

        return boundary_data


    def DirichletFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN

        X_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[X_0,:2,:] = 0.

        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[X_0,2,:] = 0.

        return boundary_data


    def NeumannFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        # X_0 = np.isclose(mesh.points[:,0],0.5)
        # boundary_data[X_0,2] = 9.e4
        # X_0 = np.isclose(mesh.points[:,0],1.)
        X_0 = np.isclose(mesh.points[:,0],0.5)
        # boundary_data[X_0,2] = 1.e10
        boundary_data[X_0,2] = 9.e-4

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):

        boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN

        mag = 9e-4
        d1 = np.ones(5000)*mag
        d2 = np.zeros(time_step-5000)
        # mag = 3e-4
        # d1 = np.ones(2000)*mag
        # d2 = np.zeros(time_step-2000)
        d = np.concatenate((d1,d2))
        X_0 = np.isclose(mesh.points[:,0],0.5)
        boundary_data[X_0,2,:] = d

        return boundary_data


    # time_step = 20000
    time_step = 40000
    # time_step = 1800
    boundary_condition = BoundaryCondition()
    # boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None

    formulation = DisplacementPotentialFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 1.5, 1e4)
    contact_formulation = None
    fem_solver = FEMSolver(total_time=5,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3, has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=80,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver, contact_formulation=contact_formulation)

    solution.WriteVTK("ElectroExpDyn2D", quantity=0, interpolation_degree=5)


def CarGeom(p=2):

    ndim = 3


    mesh = Mesh()
    mesh.ReadGmsh("car2.msh",element_type="tet", read_surface_info=True)
    mesh.GetHighOrderMesh(p=2, Decimals=7)

    material = MooneyRivlin_0(ndim, mu1=1., mu2=1.,lamb=400)
    cad_file = 'car.step'

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition, project_on_curves=True, solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)
    # exit()

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", iterative_solver_tolerance=5.0e-07, dont_switch_solver=True)
    # solver = None
    formulation = DisplacementFormulation(mesh)
    # fem_solver = FEMSolver(number_of_load_increments=2,analysis_nature="linear")
    fem_solver = FEMSolver(number_of_load_increments=2, analysis_nature="linear", has_low_level_dispatcher=True)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh,
            material=material, boundary_condition=boundary_condition, solver=solver)

    solution.CurvilinearPlot(plot_points=True, point_radius=0.2, color="#E3A933")




def CarCrash():

    mesh = Mesh()
    # mesh.ReadGmsh("car0.msh",element_type="tet", read_surface_info=True)
    # mesh.ReadGmsh("car2.msh",element_type="tet")
    # mesh.ReadGmsh("car3.msh",element_type="tet")
    # mesh.ReadHDF5("car4")


    # mesh.ReadGmsh("car_notyres0.msh",element_type="tet", read_surface_info=True)
    mesh.ReadGmsh("car_notyres2.msh",element_type="tet", read_surface_info=True)
    mesh.points[:,0] -= mesh.points[:,0].min()
    mesh.points[:,1] -= mesh.points[:,1].min()

    v = mesh.Volumes()
    print v.min(), v.max()
    # mesh = mesh.GetLocalisedMesh(v>.5)
    mesh = mesh.GetLocalisedMesh(v>.05)

    # print mesh.nnode*3, mesh.nelem
    # mesh.SimplePlot()
    # exit()
    # print mesh.Bounds
    # mesh.RemoveElements((0.,           -1.89905898,  -43.59201906, 140.51271929,   39.24887553,   43.59208611))
    # mesh.RemoveElements((5.,           -1.89905898,  -43.59201906, 140.51271929,   39.24887553,   43.59208611))
    # mesh.SimplePlot()
    # mesh.GetHighOrderMesh(p=4)

    # from scipy.io import loadmat
    # sol = loadmat("CarCrash")['Solution']
    # pp = PostProcess(3,3)
    # pp.SetMesh(mesh)
    # pp.SetFormulation(DisplacementFormulation(mesh))
    # pp.SetSolution(sol)
    # print sol.shape
    # pp.WriteVTK("CarCrash", quantity=41)
    # exit()

    # print mesh.face_to_surface
    # flags = np.ones_like(mesh.face_to_surface)
    # fs = [1,3,6,7,25,28,30,31,35,37,46,55,57,62,69,84,86,90,91,93,104,106,119]
    # for i in fs:
    #     flags[mesh.face_to_surface==i] = 0
    # pp = PostProcess(3,3)
    # pp.CurvilinearPlot(mesh,np.zeros_like(mesh.points),ProjectionFlags=flags)
    # exit()


    mu = 1.0e5
    mu1 = mu
    mu2 = 0.
    v = 0.425 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(3, mu1=mu1, mu2=mu2, lamb=lamb, rho=1000.)
    print(mesh.Bounds)
    # exit()


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN
        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],3))+np.NAN

        # mag = 7e4
        mag = 2e5
        boundary_data[:,0] = -mag

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],3, time_step))+np.NAN

        mag = 2e5
        print(mag)
        # exit()
        d1 = np.ones(500)*mag
        d2 = np.zeros(time_step-500)
        d = np.concatenate((d1,d2))
        X_0 = np.greater(mesh.points[:,0],135.0)
        # print mesh.points[X_0,:]
        # exit()
        boundary_data[X_0,0,:] = -d

        return boundary_data


    # time_step = 8000
    # time_step = 20000
    time_step = 100000
    # time_step = 1
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    print 355
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    # boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    formulation = DisplacementFormulation(mesh)
    # contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([1.,0.,0.]), 3., 2e7)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([1.,0.,0.]), 20., 2e7)
    # contact_formulation = None
    fem_solver = FEMSolver(total_time=10,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3, has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=100,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=100000,
        save_incremental_solution=False, incremental_solution_filename=PWD(__file__)+"/results/CarNoTyres")

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, contact_formulation=contact_formulation)

    # solution.WriteVTK("CarCrash", quantity=0, interpolation_degree=5)
    # solution.WriteHDF5("CarCrash",compute_recovered_fields=False)

    # solution.WriteVTK("CarCrashMedium", quantity=0, interpolation_degree=5)
    # solution.WriteHDF5("CarCrashMedium",compute_recovered_fields=False)

    # solution.WriteVTK("CarCrashNoTyres", quantity=0, interpolation_degree=5)
    # solution.WriteHDF5("CarCrashNoTyres",compute_recovered_fields=False)




def Car2D():

    mesh = Mesh()
    # mesh.ReadGmsh("Car2D.msh",element_type="tri")
    mesh.ReadGmsh("Car2D.msh",element_type="quad")
    # mesh.ReadDCM("Circle",element_type="quad")
    # mesh.SimplePlot()
    mesh.points /=1000.

    # mu = 1.0e5
    # mu1 = mu
    # mu2 = 0.
    # v = 0.495 # poissons ratio
    # lamb = 2.*mu*v/(1-2.*v)
    # material = ExplicitMooneyRivlin_0(2, mu1=mu1, mu2=mu2, lamb=lamb, rho=1000.)
    # print(mesh.Bounds)
    # # exit()



    mu = 1.0e6
    mu1 = mu
    mu2 = 0.
    v = 0.495 # poissons ratio
    lamb = 2.*mu*v/(1-2.*v)
    material = ExplicitMooneyRivlin_0(2, mu1=mu1, mu2=mu2, lamb=lamb, rho=8000.)
    print(mesh.Bounds)
    # exit()


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN
        # X_0 = np.isclose(mesh.points[:,1],-12.5)
        # boundary_data[:,1] = 0.

        X_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[:,1] = 0.
        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],2))+np.NAN
        # X_0 = np.isclose(mesh.points[:,1],10)
        # boundary_data[X_0,0] = 9.e3
        boundary_data[:,0] = 5.e3

        return boundary_data


    def NeumannFuncDyn(mesh, time_step):
        boundary_data = np.zeros((mesh.points.shape[0],2, time_step))+np.NAN

        mag = 5e5
        n = 2000
        d1 = np.ones(n)*mag
        d2 = np.zeros(time_step-n)
        d = np.concatenate((d1,d2))
        # X_0 = np.isclose(mesh.points[:,1],10.0)
        # boundary_data[X_0,0,:] = d
        boundary_data[:,0,:] = d

        boundary_data[:,1] = -5000.

        return boundary_data


    # time_step = 5000
    time_step = 6000
    # time_step = 250
    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)
    # boundary_condition.SetDirichletCriteria(DirichletFuncDyn, mesh, time_step)
    boundary_condition.SetNeumannCriteria(NeumannFuncDyn, mesh, time_step)

    # solver = LinearSolver(linear_solver="direct", linear_solver_type="pardiso", dont_switch_solver=True)
    solver = LinearSolver(linear_solver="direct", linear_solver_type="mumps", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack", dont_switch_solver=True)
    # solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg", dont_switch_solver=True)
    # solver = None

    formulation = DisplacementFormulation(mesh)
    contact_formulation = ExplicitPenaltyContactFormulation(mesh, np.array([-1.,0.]), 180., 1e7)
    # contact_formulation = None
    fem_solver = FEMSolver(total_time=9,number_of_load_increments=time_step,
        analysis_type="dynamic",
        # analysis_type="static",
        # analysis_subtype="implicit",
        analysis_subtype="explicit",
        mass_type="lumped",
        # mass_type="consistent",
        analysis_nature="nonlinear", newton_raphson_tolerance=1e-3, has_low_level_dispatcher=True,
        print_incremental_log=True, parallelise=False,
        save_frequency=50,
        # compute_linear_momentum_dissipation=True,
        break_at_increment=252000)

    solution = fem_solver.Solve(formulation=formulation, material=material, mesh=mesh,
        boundary_condition=boundary_condition, solver=solver, contact_formulation=contact_formulation)

    solution.WriteVTK("Car2D", quantity=0, interpolation_degree=5)



if __name__ == "__main__":
    explicit_dynamics_mechanics()






