import os, sys
from scipy.io import loadmat
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def GetMeshes(*args, **kwargs):

    assert len(args) == 1
    MainData = args[0]
    ndim = 3

    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    ProblemPath = PWD(__file__)

    # FOR GETTING HIGH ORDER MESHES
    # filename_0 = ProblemPath + '/FingerMech.dat'
    # filename_1 = ProblemPath + '/FingerElec.dat'

    filename_0 = ProblemPath + '/FingerMech_Straight.dat'
    filename_1 = ProblemPath + '/FingerElec_Straight.dat'
    mesh = Mesh()
    mesh.Reader(filename=filename_0, element_type="tet")
    mesh_1 = Mesh()
    mesh_1.Reader(filename=filename_1, element_type="tet")
    # mesh.MergeWith(mesh_1)
    mesh = mesh + mesh_1
    mesh.GetHighOrderMesh(p=MainData.C+1, Decimals=8)
    # mesh.SimplePlot()
    # exit()

    cad_file = ProblemPath + '/FingerHarvester_Straight.iges'

    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_nature="linear")

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.CurvilinearPlot()
    mesh.points += solution.sol[:,:formulation.ndim,-1]
    # mesh.WriteHDF5(ProblemPath+"/FingerHavester_P"+str(MainData.C+1)+".mat")
    mesh.WriteHDF5(ProblemPath+"/FingerHarvester_Straight_P"+str(MainData.C+1)+".mat")






def ProblemData(p=2):

    ndim=3

    # material = NeoHookean_2(ndim,youngs_modulus=10.0, poissons_ratio=0.3)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=1e8, poissons_ratio=0.3, eps_1=1e-6, eps_2=1e-6)
    # material = IsotropicElectroMechanics_100(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_101(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_102(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_103(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_104(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1e7,mu2=1e7,lamb=2e7, eps_1=1e-5, eps_2=1e-5)
    material = IsotropicElectroMechanics_106(ndim,mu1=10.0,mu2=10.0,lamb=20., eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_107(ndim,mu1=10.0,mu2=10.0,mue=5.0,lamb=20., eps_1=1.0, eps_2=1.0, eps_e=1.0)


    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/FingerHavester_P'+str(p)+'.mat'
    filename = ProblemPath + '/FingerHarvester_Straight_P'+str(p)+'.mat'
    
    mesh = Mesh()
    mesh.ReadHDF5(filename)
    # print mesh.Bounds
    # exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        # Y_0 = np.isclose(mesh.points[:,0],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        # Y_1 = np.isclose(mesh.points[:,0],100.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = -50.0
        # boundary_data[Y_1,2] = 0.0

        # Electromechanics
        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        Y_1 = np.isclose(mesh.points[:,1],10.)
        boundary_data[Y_1,3] = 16.0
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        Y_2 = np.isclose(mesh.points[:,1],12.)
        # boundary_data[Y_2,0] = 0.0
        # boundary_data[Y_2,1] = 0.0
        # boundary_data[Y_2,2] = 0.0
        boundary_data[Y_2,3] = -16.0

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementPotentialFormulation(mesh)
    # formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=20,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-01)
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
    # solution.Plot(configuration="deformed",quantity=3, plot_points=True)
    # solution.Animate(configuration="deformed",quantity=1, plot_points=True, save=True, filename="/home/roman/ZZZchecker/QQ.mp4")
    # solution.WriteVTK(configuration="deformed",quantity=1, filename="/home/roman/ZZZchecker/SS.vtu")
    # solution.WriteHDF5(filename="/home/roman/ZZZchecker/TT.mat")
    solution.WriteHDF5(filename="/home/roman/Dropbox/TT.mat")
    # solution.Plot(configuration="original",quantity=35)



def ReadData(p=2):

    from scipy.io import loadmat
    from Florence.PostProcessing import PostProcess

    # dd = loadmat("/home/roman/Dropbox/TT.mat")
    dd = loadmat("/home/roman/ZPlots/TT.mat")
    # print dd.keys()

    ProblemPath = PWD(__file__) 
    filename = ProblemPath + '/FingerHarvester_Straight_P'+str(p)+'.mat'
    
    mesh = Mesh()
    mesh.ReadHDF5(filename)

    post_process = PostProcess(3,4)
    post_process.SetSolution(dd['Solution'])
    post_process.SetMesh(mesh)

    post_process.WriteVTK("/home/roman/ZPlots/TT.vtu", quantity=1)
    # post_process.Animate(configuration="deformed",quantity=0)






if __name__ == "__main__":
    class MainData():
        C = 1
    # GetMeshes(MainData)

    # ProblemData(p=MainData.C+1)

    ReadData(p=MainData.C+1)