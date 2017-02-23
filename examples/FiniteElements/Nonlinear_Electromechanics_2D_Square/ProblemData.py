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
    # material = NeoHookean_2(ndim,youngs_modulus=1.0e0,poissons_ratio=0.3, rho=1.0)
    # material = LinearModel(ndim,youngs_modulus=1.0,poissons_ratio=0.495, rho=1.0)
    # material = IncrementalLinearElastic(ndim,youngs_modulus=1.0,poissons_ratio=0.495, rho=1.0)
    # material = NeoHookean_2(ndim,lame_parameter_1=0.4,lame_parameter_2=0.6)
    material = MooneyRivlin_0(ndim,mu1=1.0,mu2=1.0,lamb=2.3)
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

        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0
        boundary_data[Y_0,1] = 0
        # boundary_data[Y_0,2] = 0

        # Y_1 = np.where(mesh.points[:,1] == 1)[0]
        Y_1 = np.isclose(mesh.points[:,1],10.)
        # boundary_data[Y_1,0] = 0
        # boundary_data[Y_1,1] = -6
        boundary_data[Y_1,1] = 3
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
    # solution.Plot(configuration="original", quantity=20, plot_points=False)
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

    print np.linalg.norm(solution.sol)





def ProblemData_2(*args, **kwargs):

    ndim=3
    p=3

    mesh = Mesh()
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=1,ny=1)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=4,ny=8)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=7,ny=14, element_type="quad")
    # mesh.Rectangle(lower_left_point=(-1,-1),upper_right_point=(1,1),nx=2,ny=2)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),element_type="quad",nx=2,ny=2)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=5,ny=5)

    # mesh.Sphere()
    # mesh.SimplePlot()
    # material.GetFibresOrientation(mesh,plot=False)
    # exit()
    # print mesh.points.shape[0], mesh.nelem
    # mesh.PlotMeshNumbering()
    # mesh.Sphere()
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements/MechanicalComponent2D/MechanicalComponent2D_192.dat")
    # mesh.ReadGIDMesh("/home/roman/Dropbox/florence/examples/FiniteElements/MechanicalComponent3D/mechanicalComplex.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements/Cylinder/Hollow_Cylinder.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements/Nonlinear_Electromechanics_Benchmark/Patch_26807.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements/Nonlinear_3D_Cube/Mesh_Cube_12_Tet.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_114.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_200.dat", "tet")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_1116.dat", "tet")
    mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements//Nonlinear_Electromechanics_2D_Square/Mesh_NormalPlate_Hex.dat", "hex")
    # mesh.Reader("/home/roman/Dropbox/florence/examples/FiniteElements/Nonlinear_Electromechanics_3D_Beam/Mesh_125.dat", "hex")


    # mesh.elements = np.array([[0,1,2,3,4,5,6,7]])
    # mesh.points = np.array([
    #     [ 0,  0, 0.],
    #     [ 100., 0, 0.],
    #     [100,  50., 0.],
    #     [0, 50., 0.],
    #     [ 0,  0, 2.],
    #     [ 100., 0, 2.],
    #     [100.,  50., 2.],
    #     [0, 50., 2.],
    #     ])


    # mesh.elements = np.array([[0,1,2,3,4,5,6,7], [4,5,6,7,8,9,10,11]])
    # mesh.points = np.array([
    #     [ 0,  0, 0.],
    #     [ 100., 0, 0.],
    #     [100,  50., 0.],
    #     [0, 50., 0.],
    #     [ 0,  0, 1.],
    #     [ 100., 0, 1.],
    #     [100.,  50., 1.],
    #     [0, 50., 1.],
    #     [ 0,  0, 2.],
    #     [ 100., 0, 2.],
    #     [100.,  50., 2.],
    #     [0, 50., 2.],
    #     ])
    # mesh.nelem = mesh.elements.shape[0]
    # mesh.element_type = "hex"
    # mesh.GetBoundaryFaces()
    # mesh.GetBoundaryEdges()


    # print mesh.Bounds
    # exit()

    # mesh.elements = np.array([[0,1,2,3]])
    # mesh.points = np.array([
    # exit()
    #     [-1,-1,-1.],
    #     [ 1,-1,-1.],
    #     [-1, 1,-1.],
    #     [-1,-1, 1.],
    #     ])
    # mesh.nelem = 1
    # mesh.element_type = "tet"
    # mesh.GetBoundaryFacesTet()
    # mesh.GetBoundaryEdgesTet()

    # mesh.OneElementCylinder(nz=1)

    # print mesh.points

    # print mesh.points.shape, mesh.elements.shape
    # mesh.ConvertTrisToQuads()
    # mesh.ConvertQuadsToTris()
    # mesh.ConvertHexesToTets()
    # mesh.ConvertTetsToHexes()
    # mesh.GetHighOrderMesh(p=p)
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # from Florence.PostProcessing import PostProcess
    # PostProcess.TessellateHexes(mesh,np.zeros_like(mesh.points),interpolation_degree=0,plot_points=True)
    # exit()
    mesh.GetHighOrderMesh(p=p)
    # x = mesh.points
    # makezero(x,tol=1e-8)
    # mesh.points = x

    # print mesh.points.shape[0]*4
    # exit()


    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=100.5)
    # material = IsotropicElectroMechanics_3(ndim,youngs_modulus=1.0e6,poissons_ratio=0.3, eps_1=10.5e-4, eps_2=1e-4)
    # material = Steinmann(ndim,youngs_modulus=1.0e6,poissons_ratio=0.3, c1=10.5e-4, c2=1e-4, eps_1=1e-5)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0,mu2=0.5, lamb=1.0, eps_1=10.5, eps_2=2.1)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0e6,mu2=0.5e6, lamb=2.0e6, eps_1=1e-03, eps_2=1e-03)
    # material = IsotropicElectroMechanics_107(ndim,mu1=1.0,mu2=0.5, lamb=1.0, eps_1=10.5, eps_2=200.1, mue=1.0, eps_e=1.0)
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=1.0, lamb=1.0, eps_1=1.5, eps_2=1.1, eps_3=1.0)

    mesh.points /=1000.
    e0 = 8.85*1e-12
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=0.5, lamb=495.0, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    material = Piezoelectric_100(ndim,mu1=1e9,mu2=0.5e9, mu3=1.5e9, lamb=495e6, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.005, mu3=0.5, lamb=.495, eps_1=1.e-1*e0, eps_2=1e-1*e0, eps_3=1e2*e0)

    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.5, mu3=0.25, lamb=2., eps_1=1., eps_2=1., eps_3=1.)


    # material = IsotropicElectroMechanics_0(ndim,mu=1.0, lamb=2.0, eps_1=1.)
    # material = IsotropicElectroMechanics_3(ndim,mu=1.0, lamb=2.0, eps_1=1., eps_2=1.)
    # material = IsotropicElectroMechanics_100(ndim,mu=1.0, lamb=2.0, eps_1=1.)
    # material = IsotropicElectroMechanics_101(ndim,mu=1.e9, lamb=0.5e9, eps_1=2.e1*e0)
    # material = IsotropicElectroMechanics_102(ndim,mu=1.0, lamb=2.0, eps_1=1e4)
    # material = IsotropicElectroMechanics_104(ndim,mu=1.0, lamb=2.0, eps_1=1e4, eps_2=1e4)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1.0,mu2=0.5, lamb=2.0, eps_1=10., eps_2=10.)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4) ###
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.005, mu3=1.0, lamb=2.0, eps_1=1e4, eps_2=1e4) 
    # material = NeoHookean_2(ndim,youngs_modulus=2.3*1e4, poissons_ratio=0.3)
    # material = AnisotropicMooneyRivlin(ndim, alpha=1.0, beta=1.0, lamb=1.0) 
    # material = AnisotropicMooneyRivlin_0(ndim, mu=1.0, lamb=1.0)
    # material = AnisotropicMooneyRivlin_1(ndim, mu1=1.0, mu2=1.0, mu3=100.0, lamb=2.0)

    # material.Linearise({'invariant':'u(J-1)**2','coefficient':'u','kinematics':'C','constants':'I'})
    # material.LineariseInvariant({'invariant':'F:F','coefficient':'lamb / 2 '})
    # material.LineariseInvariant({'invariant':'lambda / 2* ( J - 1 )**2','coefficient':'lambda/2'})
    # material.Linearise("u1*C:I+u2*G:I-2*(mu1+2*mu2)*lnJ")
    # material.GetFibresOrientation(mesh,plot=False)
    # exit()

    # material = IsotropicElectroMechanics_200(ndim,mu1=1., mu2=1., lamb=1.0, eps_1=2.)
    # material = IsotropicElectroMechanics_201(ndim,mu1=1., mu2=1., lamb=1.0, eps_1=2.)

    material.anisotropic_orientations = np.zeros((mesh.nelem,ndim))
    # material.anisotropic_orientations[:,0] = -1.

    a,b,c=0.5,0.5,0.5
    material.anisotropic_orientations[:,:] = np.array([a,b,c])/np.sqrt(a**2+b**2+c**2)

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Y_0 = np.isclose(mesh.points[:,1],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # # boundary_data[Y_0,2] = 0

        # Y_1 = np.isclose(mesh.points[:,0],0.)
        # boundary_data[Y_1,2] = -1e8

        # Y_1 = np.isclose(mesh.points[:,0],2/1000.)
        # # boundary_data[Y_1,0] = 0.0
        # # boundary_data[Y_1,1] = 2.0
        # # boundary_data[Y_1,2] = 0.01
        # boundary_data[Y_1,2] = 1e8

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics
        # print boundary_data[2::material.nvar,:]
        # exit()

        # 3D
        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        # boundary_data[Y_0,3] = -1e-2
        # boundary_data[Y_0,3] = -0.

        Y_1 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_1,3] = 0.

        Y_1 = np.isclose(mesh.points[:,2],2./1000.)
        # Y_1 = np.isclose(mesh.points[:,0],100.)
        # boundary_data[Y_1,0] = 200.0
        # boundary_data[Y_1,1] = 0.0
        # boundary_data[Y_1,2] = 2.
        # boundary_data[Y_1,3] = 1e-2
        boundary_data[Y_1,3] = 2e8
        # print boundary_data
        # exit()

        # boundary_data[:,3] = 0. # fix all electrostatics
        # boundary_data[:,:3] = 0 # fix all mechanics

        # print boundary_data

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
    # formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(newton_raphson_tolerance=1e-02, number_of_load_increments=20)
    # fem_solver = StaggeredFEMSolver(newton_raphson_tolerance=1e-02, number_of_load_increments=5)
    # fem_solver = StaggeredElectromechanicSolver(newton_raphson_tolerance=1e-02, number_of_load_increments=5)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    # print (2e-10/2.)/np.sqrt(0.5/1e2*e0)
    # 50 incr
    # 0.0302276310449 femstag
    # 0.0320120473061 elecstag
    # 0.0307196172242 mono

    # 0.0170649814719 0.0363506882861

    # sol = np.copy(solution.sol[:,:,-1])
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

    print np.linalg.norm(solution.sol[:,:ndim,-1])
    print solution.sol[:,0,-1].max(), solution.sol[:,1,-1].max(), solution.sol[:,2,-1].max(), solution.sol[:,3,-1].max()
    # print np.linalg.norm(solution.sol[:,-1,-1])
    # solution.Plot(configuration="original",quantity=42)
    solution.WriteVTK(quantity=42, filename="/home/roman/ZPlots/GG.vtu")
    # solution.Animate(configuration="original",quantity=21)


    # 0.0122204689158
    # 0.0239793477991 0.0141085965428 0.0134785934614


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





def ProblemData_22(*args, **kwargs):

    ndim=2
    p=2

    mesh = Mesh()
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=2,ny=10,"")
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=3,ny=5)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=4,ny=8)
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=2,ny=100, element_type="quad")
    # mesh.Rectangle(lower_left_point=(-1,-1),upper_right_point=(1,1),nx=2,ny=2)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),element_type="quad",nx=2,ny=2)
    # mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,4),nx=5,ny=5)
    # print mesh.elements.shape[1]
    # print mesh.InferSpatialDimension()

    material = MooneyRivlin_0(ndim,mu1=1.0,mu2=1.0,lamb=1000.3)
    mesh.GetHighOrderMesh(p=p)

    # exit()
    # exit()


    # mesh.points /=1000.
    e0 = 8.85*1e-12
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=0.5, lamb=495.0, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1e9,mu2=0.5e9, mu3=0.5e9, lamb=495e6, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.005, mu3=0.5, lamb=.495, eps_1=1.e-1*e0, eps_2=1e-1*e0, eps_3=1e2*e0)

    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.5, mu3=0.25, lamb=2., eps_1=1., eps_2=1., eps_3=1.)


    # material = IsotropicElectroMechanics_0(ndim,mu=1.0, lamb=2.0, eps_1=1.)
    # material = IsotropicElectroMechanics_3(ndim,mu=1.0, lamb=2.0, eps_1=1., eps_2=1.)
    # material = IsotropicElectroMechanics_100(ndim,mu=1.0, lamb=2.0, eps_1=1.)
    # material = IsotropicElectroMechanics_101(ndim,mu=1.e9, lamb=0.5e9, eps_1=2.e1*e0)
    # material = IsotropicElectroMechanics_102(ndim,mu=1.0, lamb=2.0, eps_1=1e4)
    # material = IsotropicElectroMechanics_104(ndim,mu=1.0, lamb=2.0, eps_1=1e4, eps_2=1e4)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1.0,mu2=0.5, lamb=2.0, eps_1=10., eps_2=10.)
    # material = IsotropicElectroMechanics_105(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4)
    # material = IsotropicElectroMechanics_106(ndim,mu1=1.0,mu2=0.005, lamb=2.0, eps_1=1e4, eps_2=1e4) ###
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.005, mu3=1.0, lamb=2.0, eps_1=1e4, eps_2=1e4) 
    # material = NeoHookean_2(ndim,youngs_modulus=2.3*1e4, poissons_ratio=0.3)
    # material = AnisotropicMooneyRivlin(ndim, alpha=1.0, beta=1.0, lamb=1.0) 
    # material = AnisotropicMooneyRivlin_0(ndim, mu=1.0, lamb=1.0)
    # material = AnisotropicMooneyRivlin_1(ndim, mu1=1.0, mu2=1.0, mu3=100.0, lamb=2.0)

    # material = IsotropicElectroMechanics_200(ndim,mu1=1., mu2=0., lamb=0.0, eps_1=2.)
    # material = IsotropicElectroMechanics_201(ndim,mu1=1., mu2=1., lamb=1.0, eps_1=2.)

    material.anisotropic_orientations = np.zeros((mesh.nelem,ndim))
    material.anisotropic_orientations[:,1] = -1.
    # a,b=0.1,0.9
    # material.anisotropic_orientations[:,:] = np.array([a,b])/np.sqrt(a**2+b**2)

    # mesh.SimplePlot()
    # exit()


    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0

        Y_1 = np.isclose(mesh.points[:,1],10.)
        boundary_data[Y_1,1] = -5.

        # Y_1 = np.isclose(mesh.points[:,0],2./1000.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = 2.0
        # boundary_data[Y_1,2] = 5e-3

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics
        # print boundary_data[2::material.nvar,:]
        # exit()

        return boundary_data


    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc,mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(newton_raphson_tolerance=1e-04, number_of_load_increments=10, parallelise=False)
    # fem_solver = StaggeredFEMSolver(newton_raphson_tolerance=1e-04, number_of_load_increments=10)
    # fem_solver = StaggeredElectromechanicSolver(newton_raphson_tolerance=1e-04, number_of_load_increments=4)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # sol = np.copy(solution.sol[:,:,-1])
    # makezero(sol,tol=1.0e-8)
    # print sol
    # solution.Animate(configuration="deformed", quantity=20)

    # solution.sol *= 1000.
    # mesh.points *= 1000.
    # print np.linalg.norm(solution.sol[:,:ndim,-1])
    # print solution.sol[:,0,-1].max(), solution.sol[:,1,-1].max(), solution.sol[:,2,-1].max()

    # solution.Plot(configuration="deformed",quantity=1)
    solution.Animate(configuration="deformed",quantity=1, colorbar=False)

    # 3.3978065921
    # 1.22803727354 0.241734624567 5.0




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
    p=1


    # from Florence.PostProcessing import PostProcess
    # post_process = PostProcess(3,4)
    # post_process.QuantityNamer(15)
    # exit()

    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    # material = Steinmann(ndim,youngs_modulus=10000.0,poissons_ratio=0.3, c1=1.0, c2=1.0, eps_1=0.05)
    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=100.5)
    # material = IsotropicElectroMechanics_100(ndim,youngs_modulus=10.0, poissons_ratio=0.3, eps_1=10.0, eps_2=10.0)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0,poissons_ratio=0.41)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3)
    # material = MooneyRivlin_0(ndim,mu1=1.0,mu2=1.0,lamb=2.3)

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
    mesh.ReadHDF5(filename)
    # makezero(mesh.points,tol=1e-9)
    # mesh.GetHighOrderMesh(p=p)
    # print mesh.points[155,:]
    # print mesh.points[154,:]
    # exit()
    # from Florence.PostProcessing import PostProcess
    # PostProcess.CurvilinearPlotTet(mesh,np.zeros_like(mesh.points))
    # makezero(mesh.points, tol=1e-7)
    # mesh.SimplePlot()
    # print mesh.Bounds
    # print mesh.points.shape
    # mesh.PlotMeshNumbering()
    # exit()

    # mesh.points /=1000.
    e0 = 8.85*1e-12
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=0.5, lamb=495.0, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    material = Piezoelectric_100(ndim,mu1=1e9,mu2=0.5e9, mu3=0.5e9, lamb=495e6, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.005, mu3=0.5, lamb=.495, eps_1=1.e-1*e0, eps_2=1e-1*e0, eps_3=1e2*e0)

    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.5, mu3=0.25, lamb=2., eps_1=1., eps_2=1., eps_3=1.)

    material.anisotropic_orientations = np.zeros((mesh.nelem,ndim))
    material.anisotropic_orientations[:,1] = -1.
    # # a,b=0.1,0.9
    # material.anisotropic_orientations[:,:] = np.array([a,b])/np.sqrt(a**2+b**2)

    material = IsotropicElectroMechanics_107(ndim,mu1=1.0,mu2=0.5, lamb=2.0, eps_1=1e4, eps_2=1e4, mue=1., eps_e=1e4) 



    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Y_0 = np.where(mesh.points[:,2] == 0)[0]
        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = -2e2

        Y_1 = np.isclose(mesh.points[:,2], 100)
        boundary_data[Y_1,0] = 0.
        boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = 0.
        boundary_data[Y_1,3] = 2e2

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

    formulation = DisplacementPotentialFormulation(mesh)
    # formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1, newton_raphson_tolerance=1e-05,parallelise=False)
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
    # solution.Plot(configuration="deformed", quantity=2, plot_points=True)
    # solution.Plot(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True, 
        # save=True, filename="/home/roman/ZZZchecker/xx.png")

    # solution.Animate(configuration="deformed", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True, 
        # save=True, filename="/home/roman/ZZZchecker/yy.mp4")
    # solution.Animate(configuration="deformed", quantity=2, plot_points=True)
    # solution.Animate(configuration="original", quantity=2, plot_on_curvilinear_mesh=True, plot_points=True)
    # solution.QuantityNamer(2)
    # solution.Plot(configuration="deformed", quantity=2)
    # solution.WriteVTK("/home/roman/ZZZchecker/GG.vtu", configuration="deformed", quantity=2)





def ProblemData_4(*args, **kwargs):

    ndim=2
    p=1

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
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=1,ny=1)
    mesh.GetHighOrderMesh(p=p)
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # print mesh.Bounds


    # mesh.points /=1000.
    e0 = 8.85*1e-12
    # material = Piezoelectric_100(ndim,mu1=1.0,mu2=0.5, mu3=0.5, lamb=495.0, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1e9,mu2=0.5e9, mu3=1.5e9, lamb=495e6, eps_1=4.68*e0, eps_2=1e6*e0, eps_3=1e3*e0)
    # material = Piezoelectric_100(ndim,mu1=1.,mu2=0.005, mu3=0.5, lamb=.495, eps_1=1.e-1*e0, eps_2=1e-1*e0, eps_3=1e2*e0)

    # material.anisotropic_orientations = np.zeros((mesh.nelem,ndim))
    # material.anisotropic_orientations[:,0] = -1.
    # a,b,c=0.5,0.5,0.5
    # material.anisotropic_orientations[:,:] = np.array([a,b,c])/np.sqrt(a**2+b**2+c**2)



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.isclose(mesh.points[:,1],10.)
        # boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = 5.0
        boundary_data[Y_1,2] = 0.0

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0. # fix all electrostatics
        # boundary_data[:,:2] = 0 # fix all mechanics

        # print boundary_data

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1, analysis_nature="nonlinear",parallelise=False,
        newton_raphson_tolerance=1.0e-08)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=3,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-03)

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








def ProblemData_Rogelio(*args, **kwargs):

    ndim=3
    p=2

    # material = MooneyRivlin_0(ndim, mu1=126.0, mu2=252.0, lamb=81.512)
    material = NeoHookean_2(ndim, mu=2*126.0, lamb=81.512)

    ProblemPath = PWD(__file__)
    from scipy.io import loadmat
    dd = loadmat(ProblemPath+"/connectivity.mat")
    dd1 = loadmat(ProblemPath+"/nodes.mat")
    # print dir(dd)
    mesh = Mesh()
    mesh.elements = np.ascontiguousarray(dd['connectivity'][:,1:5] -1)
    tpoints = np.ascontiguousarray(dd1['nodes'][:,1:])
    # print mesh.points
    mesh.points = tpoints[np.unique(mesh.elements),:]
    mesh.element_type = "tet"
    mesh.nelem = mesh.elements.shape[0]
    mesh.GetBoundaryFaces()
    mesh.GetBoundaryEdges()
    # print mesh.elements
    mesh.GetHighOrderMesh(p=p,Decimals=8)
    # exit()
    # from Florence.PostProcessing import PostProcess
    # PostProcess.CurvilinearPlotTet(mesh,np.zeros_like(mesh.points),plot_points=True)
    # mesh.SimplePlot()

    # print mesh.Bounds
    # mesh.PlotMeshNumbering()

    # from Florence.QuadratureRules import FeketePoints
    # exit()
    # from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet
    # print FeketePointsTet(1)
    # tmesh = mesh.GetLocalisedMesh(0)
    # tmesh.PlotMeshNumbering()
    print mesh.points.shape
    exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,0],10.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = -0.37037

        return boundary_data


    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=4, analysis_nature="nonlinear",parallelise=False,
        newton_raphson_tolerance=1.0e-06)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    sol = np.copy(solution.sol[:,:,-1])
    makezero(sol,tol=1.0e-9)
    # print repr(sol)
    # print sol

    # Y_0 = np.where(mesh.points[:,0]==10.)
    # print mesh.points[Y_0]
    # print mesh.points
    # solution.Animate(configuration="deformed")
    # print Y_0
    # sol = mesh.points + solution.sol[:,:ndim]
    print solution.sol[196,[0,2]]










def ProblemData_Rogelio2(*args, **kwargs):

    ndim=3
    p=1

    material = MooneyRivlin_0(ndim, mu1=126.0, mu2=252.0, lamb=815.12)
    # material = NeoHookean_2(ndim, mu=2*126.0, lamb=81.512)
    mesh = Mesh()

    mesh.Parallelepiped(element_type="hex",upper_right_front_point=(1,1,10),nx=2,ny=2,nz=10)

    # mesh.SimplePlot()
    # mesh.WriteHDF5("/home/roman/BeamMesh.mat")

    # print np.where(np.isclose(mesh.points[:,0],0.5))
    # print mesh.points[94,:]
    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        # Y_1 = np.isclose(mesh.points[:,2],10.)
        # boundary_data[Y_1,2] = 10.

        return boundary_data


    def NeumannFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,2],10.)
        boundary_data[Y_0,0] = -0.37037
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        return boundary_data


    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2, analysis_nature="nonlinear",parallelise=False,
        newton_raphson_tolerance=1.0e-06)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    print solution.sol[94,:]




def ProblemData_Rogelio3(*args, **kwargs):

    ndim=3
    p=1

    # e0 = 8.8541e-12
    e0=1.0
    # material = IsotropicElectroMechanics_105(ndim, mu1=126.0/2., mu2=0.0, lamb=800., eps_1=1.0e5*e0, eps_2=1.0e5*e0)
    material = IsotropicElectroMechanics_105(ndim, mu1=126.0/2., mu2=0.0, lamb=800., eps_1=1000.0*e0, eps_2=1000.0*e0)
    # material = IsotropicElectroMechanics_101(ndim, mu=126.0, lamb=800., eps_1=1000.0*e0)
    # material = IsotropicElectroMechanics_101(ndim, mu1=126.0, mu2, lamb=800., eps_1=1000.0*e0)
    # material = IsotropicElectroMechanics_200(ndim, mu1=126.0, mu2=0.0, lamb=800., eps_1=-1000.0*e0)
    # material = IsotropicElectroMechanics_201(ndim, mu1=126.0, mu2=0.0, lamb=800., eps_1=1000.0*e0)
    
    # material = IsotropicElectroMechanics_106(ndim, mu1=126.0/2., mu2=0.0, lamb=800., eps_1=1000.0*e0, eps_2=1000.0*e0)
    # material = IsotropicElectroMechanics_107(ndim, mu1=126.0/2., mu2=0.0, mue=0.1,lamb=800., eps_1=1000.0*e0, eps_2=1000.0*e0, eps_e=1000.0*e0)

    mesh = Mesh()
    mesh.Parallelepiped(element_type="hex",upper_right_front_point=(1,1,10),nx=2,ny=2,nz=10)
    # mesh.Parallelepiped(element_type="hex",upper_right_front_point=(1,1,10),nx=4,ny=4,nz=400)
    # mesh.Cube(element_type="hex",n=1)
    # mesh.SimplePlot()
    # mesh.WriteHDF5("/home/roman/Dropbox/OneCube.mat")
    # print mesh.Bounds
    # print mesh.nelem
    # exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN


        Y_0 = np.isclose(mesh.points[:,0],0.)
        boundary_data[Y_0,3] = 0.
        Y_0 = np.isclose(mesh.points[:,0],1.)
        # boundary_data[Y_0,3] = 1e7
        boundary_data[Y_0,3] = 1e-1

        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        # boundary_data[:,:3] = 0.

        # print boundary_data

        return boundary_data


    # def NeumannFunc(mesh):
    #     boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

    #     Y_0 = np.isclose(mesh.points[:,2],10.)
    #     boundary_data[Y_0,0] = -0.37037
    #     boundary_data[Y_0,1] = 0.
    #     boundary_data[Y_0,2] = 0.

    #     return boundary_data


    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)
    # boundary_condition.SetNeumannCriteria(NeumannFunc, mesh)

    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2, analysis_nature="nonlinear",parallelise=False,
        newton_raphson_tolerance=1.0e-6)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    print solution.sol[94,:]
    # solution.sol *= 2.
    # print solution.sol[:,:,-1]
    # solution.Animate(configuration="deformed",quantity=1)

# /home/roman/PROFILER_CPP/templight_binary/metashell/3rd/templight/build/bin
# cmake -DCUSTOM_BOOST_PATH:PATH=/media/MATLAB/boost_1_62_0 -DBOOST_LIBRARYDIR=/media/MATLAB/boost_1_62_0/stage/lib ..
# /usr/bin/time -v clang++ -fconstexpr-steps=16000000 compilation_pair_contraction.cpp -O3 -mavx -std=c++11 -I../../../ -DNINE_INDEX -DCONTRACT_OPT=0

# BinaryMatMulOp<BinaryMatMulOp<UnaryInvOp<BinaryMatMulOp<Tensor<T, ndim, nodeperelem>, Tensor<T, nodeperelem, ndim> > >, 
    # Tensor<T, ndim, nodeperelem> >, Tensor<T, nodeperelem, ndim> >



# /media/MATLAB/intel_2017/bin/icpc benchmark_quadrature_scalar.cpp -O3 -xHost -std=c++11 -I../ -I../../../../ -DVECTORISED_CLASSIC_OVERLOADS -DPOLYDEG=1
# /media/MATLAB/intel_2017/bin/icpc benchmark_quadrature_scalar.cpp -O3 -mavx -std=c++11 -I../ -I../../../../ -DVECTORISED_CLASSIC_OVERLOADS -DPOLYDEG=1
# BinaryMatMulOp<BinaryMatMulOp<Tensor<T, I, J>, Tensor<T, J, K> >, BinarySubOp<BinaryAddOp<Tensor<double, 2ul>, Tensor<double, 2ul>, 1ul>, double, 1ul> >

# BinaryMatMulOp<BinaryMatMulOp<Tensor<T, 2, 2>, Tensor<double, 2, 2> >,
#       Fastor::BinaryAddOp<Fastor::Tensor<double, 2>, Fastor::Tensor<double, 2>, 1> >

# git rev-list --all | while read rev; do git ls-tree -lr $rev | cut -c54- | grep -v '^ '; done | sort -u | sort -k 2 > /tmp/files.txt



if __name__ == "__main__":

    # ProblemData()
    # ProblemData_2()
    # ProblemData_22()
    # ProblemData_3()
    # ProblemData_3D()

    # ProblemData_4()
    # ProblemData_Rogelio()
    # ProblemData_Rogelio2()
    ProblemData_Rogelio3()
    
    from cProfile import run
    # run('ProblemData_3D()')
    # run('ProblemData_2()')
    # run('ProblemData_22()')
    # run('ProblemData_4()')


 #[[  1.13797860e-13  -3.60150509e-09]
 # [ -6.07680573e-13  -2.53353449e-11]
 # [  7.11162206e-03   2.85507799e-02]
 # [  2.50000000e-02   4.99999985e-02]]

# Mech
# [[ -2.12371894e+00  -3.79030902e+00]
#  [  4.54570825e-11   1.18958177e-11]
#  [ -2.72262514e-01  -8.98698882e-01]]