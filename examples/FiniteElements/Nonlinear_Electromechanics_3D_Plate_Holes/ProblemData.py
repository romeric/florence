import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from scipy.io import loadmat
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def GetMeshes(*args, **kwargs):

    assert len(args) == 1
    MainData = args[0]
    ndim = 3

    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/Mesh_Holes.dat'

    ##
    # filename = ProblemPath + '/Mesh_Holes_P'+str(MainData.C+1)+'.mat'
    # dd = loadmat(filename)
    # mesh = Mesh()
    # mesh.elements = dd['elements']
    # mesh.points = dd['points']
    # mesh.faces = dd['faces']
    # mesh.element_type = "tet"
    # mesh.nelem = mesh.elements.shape[0]
    # # TotalDisp = dd['TotalDisp']
    # TotalDisp = np.zeros_like(mesh.points)
    # TotalDisp = TotalDisp[:,:,None]

    # post_process = PostProcess(3,3)
    # post_process.SetMesh(mesh)
    # post_process.SetSolution(TotalDisp)        
    # post_process.CurvilinearPlot()
    # # post_process.CurvilinearPlot(plot_edges=False)
    # # post_process.CurvilinearPlot(plot_points=True,point_radius=0.2)
    # exit()
    ##



    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")
    mesh.GetHighOrderMesh(p=MainData.C+1, Decimals=8)

    cad_file = ProblemPath + '/Plate_With_Holes.iges'

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_nature="linear")

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    solution.CurvilinearPlot()
    mesh.points += solution.sol[:,:formulation.ndim,-1]
    mesh.WriteHDF5(ProblemPath+"/Mesh_Holes_P"+str(MainData.C+1)+".mat")






def ProblemData(p=2):

    ndim=3


    mesh = Mesh()
    ProblemPath = PWD(__file__)
    if p>1:
        filename = ProblemPath + '/Mesh_Holes_P'+str(p)+'.mat'
        mesh.ReadHDF5(filename)
    else:
        filename = ProblemPath + '/Mesh_Holes.dat'
        mesh.Reader(filename, "tet")

    # print mesh.Bounds
    # exit()

    material = NeoHookean_2(ndim,youngs_modulus=10.0, poissons_ratio=0.3)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=1e8, poissons_ratio=0.3, eps_1=1e-6, eps_2=1e-6)
    # material = IsotropicElectroMechanics_100(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_101(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_102(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_103(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_104(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1e7,mu2=1e7,lamb=2e7, eps_1=1e-5, eps_2=1e-5)
    # material = IsotropicElectroMechanics_106(ndim,mu1=10.0,mu2=10.0,lamb=20., eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_107(ndim,mu1=10.0,mu2=10.0,mue=5.0,lamb=20., eps_1=1.0, eps_2=1.0, eps_e=1.0)

    mesh.points /=1000.
    e0 = 8.85*1e-12
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=0.5, lamb=495.0, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1e9,mu2=0.5e9, mu3=1.5e9, lamb=495e6, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.005, mu3=0.5, lamb=.495, eps_1=1.e-1*e0, eps_2=1e-1*e0, eps_3=1e2*e0)


    # filename = ProblemPath + '/Compound_Mesh_1.dat'
    # mesh.Reader(filename, "tet")
    # mesh.SimplePlot()
    # print mesh.Bounds
    # print mesh.points[np.where(mesh.points[:,0]==0)[0],:]
    # exit()

    material.anisotropic_orientations = np.zeros((mesh.nelem,ndim))
    # material.anisotropic_orientations[:,0] = -1.

    a,b,c=0.5,0.5,0.5
    material.anisotropic_orientations[:,:] = np.array([a,b,c])/np.sqrt(a**2+b**2+c**2)


    print mesh.Bounds
    # post_process = PostProcess(3,3)
    # post_process.CurvilinearPlotTet(mesh,np.zeros_like(mesh.points),plot_edges=False)
    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanical
        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_0 = np.isclose(mesh.points[:,0],0.1)
        # boundary_data[Y_0,0] = 0.00
        # boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = -0.06
        


        # ###
        # # Energy harvesting
        # Y_0 = np.isclose(mesh.points[:,0],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        # # boundary_data[Y_0,3] = 0.

        # Y_1 = np.isclose(mesh.points[:,2],0.)
        # boundary_data[Y_1,3] = 0.0

        # Y_2 = np.isclose(mesh.points[:,2],6.)
        # boundary_data[Y_1,3] = 3e-4
        # ####

        # Y_0 = np.isclose(mesh.points[:,0],100.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        # # boundary_data[Y_0,3] = 0.

        # Y_1 = np.isclose(mesh.points[:,0],100.)
        # boundary_data[Y_1,0] = 50.0
        # boundary_data[Y_1,1] = 0.0
        # boundary_data[Y_1,2] = 0.0
        # # boundary_data[Y_1,3] = 0.0

        # Actuation
        # boundary_data[np.where(mesh.points[:,0]==0)[0],:3] = 0.

        # Y_0 = np.where(mesh.points[:,2] == 0)[0]
        # boundary_data[Y_0,3] = 0.

        # Y_1 = np.where(mesh.points[:,2] == 6.)[0]
        # boundary_data[Y_1,3] = 3.0

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics


        # Compound
        # boundary_data[np.where(mesh.points[:,0]==0)[0],:3] = 0.
        # Y_0 = np.where(mesh.points[:,2] == 0)[0]
        # boundary_data[Y_0,3] = 0.
        # Y_1 = np.where(mesh.points[:,2] == 2.)[0]
        # boundary_data[Y_1,3] = -30000.0

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1, newton_raphson_tolerance=1e-02, parallelise=True)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=6,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.sol *= 1e5 
    # sol = np.copy(solution.sol[:,:,-1])
    # makezero(sol,tol=1.0e-9)
    # print repr(sol)
    # print sol
    # solution.Plot(configuration="deformed",quantity=0, plot_points=True)
    solution.Plot(configuration="deformed",quantity=0, plot_points=False,colorbar=False, point_radius=0.0005, plot_edges=False)
    # solution.Animate(configuration="deformed",quantity=0, plot_points=True, save=True, filename="/home/roman/ZPlots/PP.mp4")
    # solution.WriteVTK(quantity=0, filename="/home/roman/ZPlots/NN.vtu")
    # solution.Plot(configuration="original",quantity=35)






if __name__ == "__main__":
    class MainData():
        C = 2
    # GetMeshes(MainData)

    ProblemData(p=MainData.C+1)