import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero




def ProblemData(*args, **kwargs):

    ndim=2
    p=2

    # material = IsotropicElectroMechanics_1(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=1.0)
    # material = MooneyRivlin(ndim,youngs_modulus=1000.0,poissons_ratio=0.3)
    material = NeoHookean_2(ndim,youngs_modulus=1.0e0,poissons_ratio=0.3, rho=1.0)
    # material = LinearModel(ndim,youngs_modulus=1.0,poissons_ratio=0.495, rho=1.0)
    # material = IncrementalLinearElastic(ndim,youngs_modulus=1.0,poissons_ratio=0.495, rho=1.0)
    # material = NeoHookean_2(ndim,lame_parameter_1=0.4,lame_parameter_2=0.6)
    # print(material.mu, material.lamb)
    # print material.nvar

    ProblemPath = PWD(__file__)
    cad_file = ProblemPath + "/Plate_Hole_2D.iges"

    mesh = Mesh()
    # mesh.Square(n=5)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=10,ny=10)
    # mesh.Reader(ProblemPath+"/Mesh_Plate_Hole_2D.dat","tri")
    # mesh.Reader(ProblemPath+"/Mesh_Plate_Hole_2D_2.dat","tri")
    # mesh.GetHighOrderMesh(p=p)
    mesh.ReadHDF5(ProblemPath+"/Mesh_Plate_Hole_Curved_P"+str(p)+".mat")
    # mesh.ReadHDF5(ProblemPath+"/Mesh_Plate_Hole_Curved_2_P"+str(p)+".mat")
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # exit()
    makezero(mesh.points)

    # print(mesh.elements)
    # mesh.PlotMeshNumbering()
    # exit()


    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0
        boundary_data[Y_0,1] = 0
        # boundary_data[Y_0,2] = 0

        # Y_1 = np.where(mesh.points[:,1] == 1)[0]
        Y_1 = np.where(mesh.points[:,1] == 10)[0]
        # boundary_data[Y_1,0] = 0
        # boundary_data[Y_1,1] = -6
        boundary_data[Y_1,1] = 6
        # boundary_data[Y_1,2] = 1

        # boundary_data[2::material.nvar,:] = 0

        # Y_0 = np.where(mesh.points[:,0] == 2)[0]
        # boundary_data[Y_0,0] = 0

        return boundary_data

    boundary_condition = BoundaryCondition()
    # boundary_condition.SetCADProjectionParameters(cad_file,scale=1000.,condition=1e10)
    # boundary_condition.GetProjectionCriteria(mesh)

    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # exit()

    # solution.StressRecovery()
    # qq = formulation.quadrature_rules[0]
    # print qq.weights, qq.points
    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0], formulation.quadrature_rules[0],mesh,solution.sol)

    # print solution.sol[:,:,-1]+mesh.points
    # print solution.sol[:,:,-1]
    # solution.sol = solution.sol[:,:,None]
    # print solution.sol[:,:,-1]+mesh.points
    # solution.WriteVTK("/home/roman/ZZZchecker/HHH", quantity=1)

    # solution.Plot(configuration="deformed", quantity=20)
    solution.Plot(configuration="original", quantity=20, plot_points=False, plot_on_curvilinear_mesh=True)
    # solution.Plot(configuration="deformed", quantity=20, plot_points=True)
    # solution.Plot(configuration="original", quantity=20, plot_points=True)
    # solution.Plot(configuration="deformed", quantity=20, save=False, filename="/home/roman/ZZZchecker/HHH")

    # solution.Animate(configuration="original", quantity=20, plot_edges=False, plot_on_curvilinear_mesh=True, colorbar=True)
    # solution.Animate(configuration="original", quantity=20, plot_points=True)
    # solution.Animate(configuration="deformed", quantity=20, save=True, filename="/home/roman/ZZZchecker/HHH.gif", plot_edges=False, colorbar=False)
    # solution.Animate(configuration="deformed", quantity=20, save=True, filename="/home/roman/ZZZchecker/HHH.gif", plot_edges=False)
    # solution.Animate(configuration="deformed", quantity=20, plot_on_curvilinear_mesh=True, plot_points=True)
    # exit()

    # mesh.points += solution.sol[:,:,-1]
    # mesh.WriteHDF5(ProblemPath+"/Mesh_Plate_Hole_Curved_P"+str(p)+".mat")
    # solution.CurvilinearPlot()





def ProblemData_2(*args, **kwargs):

    ndim=2
    p=2

    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=100.5)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=1.0e6,poissons_ratio=0.3, eps_1=10.5e-4, eps_2=1e-4)
    # material = Steinmann(ndim,youngs_modulus=1.0e6,poissons_ratio=0.3, c1=10.5e-4, c2=1e-4, eps_1=1e-5)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0,mu2=0.5, lamb=1.0, eps_1=10.5, eps_2=2.1)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0e6,mu2=0.5e6, lamb=2.0e6, eps_1=1e-03, eps_2=1e-03)
    # material = IsotropicElectroMechanics_107(ndim,mu1=1.0,mu2=0.5, lamb=1.0, eps_1=10.5, eps_2=200.1, mue=1.0, eps_e=1.0)


    # material = IsotropicElectroMechanics_101(ndim,mu=1.0, lamb=2.0, eps_1=1e4)
    # material = IsotropicElectroMechanics_102(ndim,mu=1.0, lamb=2.0, eps_1=1e4)
    # material = IsotropicElectroMechanics_104(ndim,mu=1.0, lamb=2.0, eps_1=1e4, eps_2=1e4)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4)
    material = IsotropicElectroMechanics_106(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4)

    mesh = Mesh()
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=6,ny=6)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=5,ny=5)
    mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()

    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.where(mesh.points[:,1] == 4)[0]
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = 4.0
        boundary_data[Y_1,2] = 0.01

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics
        # print boundary_data[2::material.nvar,:]
        # exit()

        return boundary_data

    def NeumannFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 4)[0]
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 2e5
        # boundary_data[Y_0,1] = 2.
        # boundary_data[Y_0,2] = 0.01

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc,mesh)

    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(newton_raphson_tolerance=1e-04, number_of_load_increments=2)
    # fem_solver = StaggeredFEMSolver(newton_raphson_tolerance=1e-04, number_of_load_increments=2)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    sol = np.copy(solution.sol[:,:,-1])
    # makezero(sol,tol=1.0e-8)
    # print sol
    # print formulation.fields
    # exit()
    # print repr(sol)
    # print solution.sol[:,2,-1]
    # print solution.sol[:,1,-1]
    # solution.sol = solution.sol[:,:2,:]
    # solution.Plot(configuration="deformed",quantity=1)
    # solution.CurvilinearPlot(QuantityToPlot=solution.sol[:,1,-1])
    # solution.Animate(configuration="deformed", quantity=20)

    # solution.Plot()

    # print np.linalg.norm(solution.sol[:,:2,-1])
    # print np.linalg.norm(solution.sol[:,-1,-1])
    solution.Plot(configuration="original",quantity=21)
    # solution.Animate(configuration="original",quantity=21)


    # 21.0460889142
    # 21.0861327294 21.0822980885 21.0841743637

    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # import matplotlib.cm as cm
    # from matplotlib import rc
    # import itertools

    # from mpl_toolkits.mplot3d import Axes3D
    # from matplotlib.mlab import griddata

    # marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
    # linestyle = itertools.cycle(('-','-.','--',':'))

    # rc('font',**{'family':'serif','serif':['Palatino'],'size':26})
    # rc('text', usetex=True)
    # params = {'text.latex.preamble' : [r'\usepackage{mathtools}']}
    # plt.rcParams.update(params)

    # # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
    # rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
    # # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})


    # colors = ['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D',
    #         '#4D5C75','#FFF056','#558C89','#F5CCBA','#A2AB58','#7E8F7C','#005A31']



    # 

    # import matplotlib.pyplot as plt
    # n = np.array([9.68466010533, 9.68466010533, 9.68466010533, 9.68466010533, 9.68466010533])
    # l = np.array([9.68349405202, 9.68423064463, 9.68459212411, 9.68465365067, 9.6846594632])
    # norm = np.linalg.norm
    # log = np.log10
    # e = np.sqrt((n-l)**2/n**2)
    # # print e
    # font_size = 20
    # plt.plot(log([1,2,10,100,1000]),log(e),'-ko',linewidth=2); plt.grid('on')
    # plt.ylabel(r'log$_{10}(\mathcal{L}^2\;error)$',fontsize=font_size)
    # plt.xlabel(r'log$_{10}(Increments)$',fontsize=font_size)
    # plt.savefig("/home/roman/l2_stagg",format='eps',dpi=300,bbox_inches='tight',pad_inches=0.12)
    # plt.show()




def ProblemData_3(*args, **kwargs):

    ndim=2
    p=2

    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=100.5)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10000.0,poissons_ratio=0.3, eps_1=1.5e-02)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=10.5, eps_2=200.1)

    mesh = Mesh()
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=2,ny=2)
    mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()

    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        # Y_1 = np.where(mesh.points[:,1] == 1)[0]
        Y_1 = np.where(mesh.points[:,1] == 10)[0]
        boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = 1.0
        boundary_data[Y_1,2] = 0.0

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0
        # boundary_data[:,:2] = 0 # fix all mechanics
        # print boundary_data[2::material.nvar,:]
        # exit()

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    # boundary_condition.SetDirichletCriteria(DirichletCriteria, mesh.points)

    formulation = DisplacementPotentialFormulation(mesh)
    # formulation = DisplacementFormulation(mesh)

    # from Florence.Solver.FEMSolver import StaggeredFEMSolver
    fem_solver = StaggeredFEMSolver(number_of_load_increments=10,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-06)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    sol = np.copy(solution.sol[:,:,-1])
    makezero(sol,tol=1.0e-9)
    # print sol
    print repr(sol)
    # print solution.sol[:,2,-1]
    # print solution.sol[:,1,-1]
    # solution.Plot(configuration="deformed",quantity=2)
    # solution.Animate(configuration="deformed", quantity=2)


    # e = np.array([ -1.00001207e-01,  -1.00040808e-02,  -9.99777745e-04,  -8.78721423e-05])
    # nincr = np.array([   10,   100,  1000, 10000])
    # import matplotlib.pyplot as plt
    # plt.loglog(nincr,np.abs(e),'-ko'); 
    # plt.grid('on'); plt.xlabel(r'Number of Increments'); 
    # plt.ylabel(r'$e=\frac{u_{nonlinear} - u_{linear}}{u_{nonlinear}}$'); 
    # plt.show()




def ProblemData_3D(*args, **kwargs):

    ndim=3
    p=3


    # from Florence.PostProcessing import PostProcess
    # post_process = PostProcess(3,4)
    # post_process.QuantityNamer(15)
    # exit()

    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    # material = Steinmann(ndim,youngs_modulus=10000.0,poissons_ratio=0.3, c1=1.0, c2=1.0, eps_1=0.05)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=100.5)
    # material = IsotropicElectroMechanics_100(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0, eps_2=10.0)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0,poissons_ratio=0.41)
    material = NeoHookean_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_Holes.dat'
    # filename = ProblemPath + '/Mesh_Cyl.dat'
    # filename = ProblemPath + '/Mesh_OneHole.dat'
    filename = ProblemPath + '/Mesh_Cyl_P'+str(p)+'.mat'
    # filename = ProblemPath + '/Mesh_OneHole_P'+str(p)+'.mat'
    cad_file = ProblemPath + '/Cylinder.iges'
    # cad_file = ProblemPath + '/OneHole.iges'

    mesh = Mesh()
    # mesh.Reader(filename=filename, element_type="tet")
    # makezero(mesh.points)
    # mesh.GetHighOrderMesh(p=p)
    mesh.ReadHDF5(filename)
    # print mesh.points[155,:]
    # print mesh.points[154,:]
    # exit()
    # makezero(mesh.points, tol=1e-7)
    # mesh.SimplePlot()
    # print mesh.Bounds
    # print mesh.points.shape
    # mesh.PlotMeshNumbering()
    # exit()


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
        boundary_data[Y_1,2] = 50.
        # boundary_data[Y_1,2] = -20.

        return boundary_data


    # def DirichletFunc(mesh):

    #     boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

    #     Y_0 = np.where(mesh.points[:,2] == 0)[0]
    #     boundary_data[Y_0,0] = 0.
    #     boundary_data[Y_0,1] = 0.
    #     boundary_data[Y_0,2] = 0.
    #     # boundary_data[Y_0,3] = 0.

    #     Y_1 = np.where(mesh.points[:,2] == 100)[0]
    #     boundary_data[Y_1,0] = 0.
    #     boundary_data[Y_1,1] = 0.
    #     boundary_data[Y_1,2] = 1.
    #     # boundary_data[Y_1,3] = 5.

    #     # boundary_data[:,3] = 0
    #     # boundary_data[Y_0,3] = 0; boundary_data[Y_1,3] = 10.1

    #     return boundary_data


    def NeumannFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN
        Y_0 = np.where(mesh.points[:,2] == 100)[0]
        boundary_data[Y_0,2] = .01

        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)
    # boundary_condition.SetCADProjectionParameters(cad_file,scale=1000.,condition=1e10, 
    #     solve_for_planar_faces=True, project_on_curves=True)
    # boundary_condition.GetProjectionCriteria(mesh)

    # boundary_condition.SetNeumannCriteria(NeumannFunc,mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=20,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-02)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=10,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-04)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # sol = np.copy(solution.sol[:,:,-1])
    # makezero(sol,tol=1.0e-9)
    # print repr(sol)


    # solution.CurvilinearPlot(mesh,solution.sol[:,:,-1])


    # solution.Plot(configuration="original", quantity=0, increment=1)
    # solution.Plot(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True)
    # solution.Plot(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True, 
        # save=True, filename="/home/roman/ZZZchecker/xx.png")

    # solution.Animate(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True, 
        # save=True, filename="/home/roman/ZZZchecker/yy.mp4")
    # solution.Animate(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True)
    # solution.Animate(configuration="original", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True)
    # solution.QuantityNamer(2)
    # solution.Plot(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True)
    # solution.StressRecovery()
    solution.WriteVTK("/home/roman/ZZZchecker/GG.vtu", configuration="deformed", quantity=2)


    # solution.CurvilinearPlot()
    # mesh.points += solution.sol[:,:,-1]
    # mesh.WriteHDF5(ProblemPath+"/Mesh_Cyl_P"+str(p)+".mat")
    # mesh.WriteHDF5(ProblemPath+"/Mesh_OneHole_P"+str(p)+".mat")

    # solution.WriteVTK("/home/roman/ZZZchecker/ff.vtu", quantity=2)




def ProblemData_4(*args, **kwargs):

    ndim=2
    p=2

    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_100(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0)
    # material = IsotropicElectroMechanics_101(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_102(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0)
    # material = IsotropicElectroMechanics_103(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_104(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_105(ndim,mu1=10.0,mu2=10.0,lamb=20., eps_1=1.0, eps_2=1.0)
    # material = IsotropicElectroMechanics_106(ndim,mu1=10.0,mu2=10.0,lamb=20., eps_1=1.0, eps_2=1.0)
    material = IsotropicElectroMechanics_107(ndim,mu1=10.0,mu2=10.0,mue=5.0,lamb=20., eps_1=1.0, eps_2=1.0, eps_e=1.0)

    mesh = Mesh()
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=2,ny=2)
    mesh.GetHighOrderMesh(p=p)
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()


    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.where(mesh.points[:,1] == 10)[0]
        # boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = 5.0
        boundary_data[Y_1,2] = 0.0

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    # boundary_condition.SetDirichletCriteria(DirichletCriteria, mesh.points)

    formulation = DisplacementPotentialFormulation(mesh)

    # fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-03)
    fem_solver = StaggeredFEMSolver(number_of_load_increments=3,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-03)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    sol = np.copy(solution.sol[:,:,-1])
    makezero(sol,tol=1.0e-9)
    # print repr(sol)
    # print sol
    print np.linalg.norm(solution.sol[:,:2,-1])
    # print np.linalg.norm(solution.sol[:,-1,-1])
    # solution.Plot(configuration="deformed",quantity=1)

    # n = np.array([9.63882100718])
    # l = np.array[9.65172916172, 9.93955034453, 9.64257268545]



if __name__ == "__main__":

    # ProblemData()
    # ProblemData_2()
    # ProblemData_3()
    ProblemData_3D()

    # ProblemData_4()
    
    # from cProfile import run
    # run('ProblemData_3D()')
