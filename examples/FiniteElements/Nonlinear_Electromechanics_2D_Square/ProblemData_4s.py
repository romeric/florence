import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero




def ProblemData_2D_mech(*args, **kwargs):

    ndim=2
    p=2

    material = NeoHookean_2(ndim,mu=1.0,lamb=2.3)
    # material = MooneyRivlin_0(ndim,mu1=1.0,mu2=1.0,lamb=2.3)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=10., poissons_ratio=0.45)


    ProblemPath = PWD(__file__)
    cad_file = ProblemPath + "/Plate_Hole_2D.iges"

    mesh = Mesh()
    mesh.ReadHDF5(ProblemPath+"/Mesh_Plate_Hole_Curved_P"+str(p)+".mat")
    # mesh.Reader(ProblemPath+"/Mesh_Plate_Hole_2D_2.dat","tri")
    # mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points)


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0
        boundary_data[Y_0,1] = 0

        Y_1 = np.isclose(mesh.points[:,1],10.)

        boundary_data[Y_1,1] = 3
        return boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,parallelise=False,
        newton_raphson_tolerance=1.0e-07)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    print np.linalg.norm(solution.sol)
    # solution.Plot(configuration="deformed")



def ProblemData_2D_electro_mech(*args, **kwargs):

    ndim=2
    p=1


    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    material = IsotropicElectroMechanics_3(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0, eps_2=1.0e3)
    # material = SteinmannModel(ndim,youngs_modulus=10.0, poissons_ratio=0.3, c1=10.0, c2=10.0, eps_1=10.)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1., mu2=0.5, lamb=1.4, eps_1=41., eps_2=12.)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1, mu2=0.5, lamb=1.4, eps_1=41, eps_2=12.)
    # material = IsotropicElectroMechanics_107(ndim,mu1=1,mu2=0.5, mue=1.5, lamb=1.4, eps_1=41, eps_2=12., eps_e=100)

    ProblemPath = PWD(__file__)
    cad_file = ProblemPath + "/Plate_Hole_2D.iges"

    mesh = Mesh()
    # mesh.ReadHDF5(ProblemPath+"/Mesh_Plate_Hole_Curved_P"+str(p)+".mat")
    mesh.Reader(ProblemPath+"/Mesh_Plate_Hole_2D_2.dat","tri")
    mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points)
    # mesh.SimplePlot()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = -0.001
        # boundary_data[Y_0,2] = 0

        Y_1 = np.isclose(mesh.points[:,0],2.)
        boundary_data[Y_1,2] = 0.001

        # Y_1 = np.isclose(mesh.points[:,1],10.)
        # boundary_data[Y_1,1] = -1.

        # boundary_data[:,2] = 0. # fix all electro

        return boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)

    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2,parallelise=False, 
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    print np.linalg.norm(solution.sol)
    # solution.Plot()



def ProblemData_3D_mech(*args, **kwargs):

    ndim=3
    p=3

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_Holes.dat'
    # filename = ProblemPath + '/Mesh_Cyl.dat'
    # filename = ProblemPath + '/Mesh_OneHole.dat'
    filename = ProblemPath + '/Mesh_Cyl_P'+str(p)+'.mat'
    # filename = ProblemPath + '/Mesh_OneHole_P'+str(p)+'.mat'
    cad_file = ProblemPath + '/Cylinder.iges'
    # cad_file = ProblemPath + '/OneHole.iges'


    mesh = Mesh()
    # mesh.ReadHDF5(filename)
    mesh.Cube(side_length=100,element_type="hex",n=2)
    mesh.GetHighOrderMesh(p=p)

    material = MooneyRivlin_0(ndim,mu1=1.0,mu2=0.5, lamb=2.0)
    # material = AnisotropicMooneyRivlin_1(ndim,mu1=1.,mu2=1.5, mu3=2.5,lamb=5.5)
    # material.anisotropic_orientations = np.zeros((mesh.nelem,mesh.points.shape[1]))
    # material.anisotropic_orientations[:,0] = 1.



    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Y_0 = np.where(mesh.points[:,2] == 0)[0]
        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.isclose(mesh.points[:,2], 100)
        boundary_data[Y_1,0] = 0.
        boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = -50.


        return boundary_data


    def NeumannFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN
        Y_0 = np.isclose(mesh.points[:,2],100.)
        boundary_data[Y_0,2] = .01

        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=5, newton_raphson_tolerance=1e-05,parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    print np.linalg.norm(solution.sol)
    solution.Animate(configuration="deformed")
    # solution.WriteVTK("/home/roman/ZPlots/Cylz.vtu", quantity=0)





def ProblemData_3D_electro_mech(*args, **kwargs):

    ndim=3
    p=2

    mesh = Mesh()

    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_114.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_200.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_1116.dat", "tet")
    mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_Hex.dat", "hex")
 
    mesh.GetHighOrderMesh(p=p)
    # print mesh.Bounds
    # exit()

    material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_107(ndim,mu1=1,mu2=0.5, mue=1.5, lamb=1.4, eps_1=41, eps_2=12, eps_e=100)
    # material = Piezoelectric_100(ndim,mu1=1,mu2=0.5, mu3=1.5, lamb=1.4, eps_1=41, eps_2=12, eps_3=100)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1,mu2=0.5, lamb=1.4, eps_1=41., eps_2=12.)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1,mu2=0.5, lamb=1.4, eps_1=41., eps_2=12.)


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN


        # 3D
        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.


        Y_1 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_1,3] = 0.

        Y_1 = np.isclose(mesh.points[:,2],2.)
        boundary_data[Y_1,3] = 0.2

        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(newton_raphson_tolerance=1e-02, number_of_load_increments=2, parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    print np.linalg.norm(solution.sol)





def LargestSegmentBenchmark():

    mesh = Mesh()

    # -------- TRI ----------
    # mesh.Square(n=1)
    # mesh.GetHighOrderMesh(p=2)
    # mesh.points[-2,:] = [0.4,1.4]
    # mesh.points[-3,:] = [0.9,.0]
    # mesh.LargestSegment(nsamples=50, plot_segment=True, plot_element=True)
    # ------------------------

    # -------- QUAD ----------
    mesh.Square(n=1, element_type="quad")
    mesh.GetHighOrderMesh(p=3)
    mesh.points[6,:] = [1.4,0.27]
    mesh.points[7,:] = [1.5,0.72]
    mesh.points[-3,:] = [0.72,0.8]
    mesh.points[-8,:] = [0.27,0.05]
    mesh.points[-1,:] = [1.0,0.68]
    mesh.points[-2,:] = [1.0,0.29]
    mesh.LargestSegment(nsamples=50, plot_segment=True, plot_element=True)
    # ------------------------

    # -------- TET ----------    
    # p=5
    # filename = PWD(__file__) + '/Mesh_Cyl_P'+str(p)+'.mat'
    # mesh.ReadHDF5(filename)
    # mesh.LargestSegment(nsamples=25, plot_segment=True, smallest_element=False)
    # ------------------------

    # -------- Hex ----------    
    # mesh.Cube(element_type="hex",nx=1,ny=1,nz=1)
    # mesh.GetHighOrderMesh(p=2)
    # mesh.points[5,:] = [-0.2,0.5,1.0]
    # mesh.points[11,:] = [0.5,-0.2,1.0]
    # mesh.points[14,:] = [0.5,0.5,1.4]
    # mesh.points[-4,:] = [1.2,0.5,1.0]
    # mesh.points[-10,:] = [0.5,1.2,1.0]
    # mesh.points[12,:] = [0.5,0.5,-0.5]
    # mesh.LargestSegment(nsamples=25, plot_segment=True)
    # ------------------------







def Problem_Parallelepiped():

    mesh = Mesh()
    # mesh.Parallelepiped(element_type="hex",nx=8,ny=8,nz=8)
    # mesh.Parallelepiped(element_type="hex", nx=6,ny=7,nz=8)
    # mesh.Parallelepiped(element_type="tet")
    mesh.Cube(element_type="hex",n=4)
    mesh.GetHighOrderMesh(p=2)
    # mesh.Cube()
    # mesh.SimplePlot()
    ndim = mesh.points.shape[1]



    material = NeoHookean_2(ndim,youngs_modulus=10, poissons_ratio=0.4)
    # material = MooneyRivlin_0(ndim,mu1=1.0,mu2=0.5, lamb=2.0)
    # material = AnisotropicMooneyRivlin_1(ndim, mu1=1.,mu2=1.5, mu3=2.5,lamb=5.5)
    # material.anisotropic_orientations = np.zeros((mesh.nelem,mesh.points.shape[1]))
    # material.anisotropic_orientations = np.random.rand(mesh.nelem,mesh.points.shape[1])
    # material.anisotropic_orientations[:,0] = 1.



    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.isclose(mesh.points[:,2], 1.)
        boundary_data[Y_1,0] = 0.
        boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = -.6


        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)
    formulation = DisplacementFormulation(mesh)

    # solver = LinearSolver(linear_solver = "multigrid")
    solver = None
    fem_solver = FEMSolver(number_of_load_increments=5, newton_raphson_tolerance=1e-05,parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition, solver=solver)

    print np.linalg.norm(solution.sol)
    solution.Animate(configuration="deformed", quantity=18)
    # solution.WriteVTK("/home/roman/ZPlots/Cylz.vtu", quantity=0)





if __name__ == "__main__":

    # ProblemData_2D_mech()
    # ProblemData_2D_electro_mech()
    # ProblemData_3D_mech()
    # ProblemData_3D_electro_mech()

    
    from cProfile import run
    # run('ProblemData_2D_mech()')
    # run('ProblemData_2D_electro_mech()')
    # run('ProblemData_3D_mech()')
    # run('ProblemData_3D_electro_mech()')


    LargestSegmentBenchmark()
    # Problem_Parallelepiped()
    # run('Problem_Parallelepiped()')

