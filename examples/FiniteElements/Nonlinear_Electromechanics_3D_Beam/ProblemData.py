import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from scipy.io import loadmat
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def ProblemData(p=1):

    ndim = 3

    # material = Steinmann(ndim,mu=2.3*10e+04,lamb=8.0*10.0e+04, eps_1=1505*10.0e-11, c1=0.0, c2=0.0, rho=7.5*10e-6)
    # material = NeoHookean_2(ndim, mu=2.3*10e+04, lamb=8.0*10.0e+04)
    material = NeoHookean_2(ndim, youngs_modulus=2.3*10e+04, poissons_ratio=0.4)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_125.dat'
    # filename = ProblemPath + '/Mesh_1.dat'                    
    filename = ProblemPath + '/Mesh_8.dat'                    
    # filename = ProblemPath + '/Mesh_64.dat'                   
    # filename = ProblemPath + '/Mesh_1000.dat'
    # filename = ProblemPath + '/Mesh_8000.dat'


    mesh = Mesh()
    # mesh.Reader(filename, "hex")
    mesh.OneElementCylinder(nz=3)
    # mesh.Areas()

    mesh.GetHighOrderMesh(p=4)

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        Y_1 = np.isclose(mesh.points[:,2],10.)
        boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = -5.0
        boundary_data[Y_1,1] = 0.0
        boundary_data[Y_1,2] = 5.0
        # boundary_data[Y_1,2] = 0.0

        # Electromechanics
        # Y_0 = np.isclose(mesh.points[:,2],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        # boundary_data[Y_0,3] = 1.
        # Y_1 = np.isclose(mesh.points[:,2],10.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = -5.0
        # boundary_data[Y_1,2] = 0.0
        # boundary_data[Y_1,3] = -1.0

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0],formulation.quadrature_rules[0],mesh)
    # exit()

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=6,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.CurvilinearPlot(QuantityToPlot=solution.sol[:,1,-1],plot_on_faces=False, plot_points=True, point_radius=.08)
    solution.CurvilinearPlot(interpolation_degree=20, plot_points=True, point_radius=.08)
    # solution.WriteVTK(filename="/home/roman/ZZZchecker/HE.vtu", quantity=10)
    # solution.WriteVTK(filename="/home/roman/Dropbox/HE.vtu", quantity=10)





def GetMeshesOneCylinder(p=1):

    ndim = 3
    material = NeoHookean_2(ndim, youngs_modulus=2.3*10e+04, poissons_ratio=0.4)

    mesh = Mesh()
    mesh.OneElementCylinder(nz=10)
    # mesh.Areas()
    mesh.GetHighOrderMesh(p=p)



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Cylinder curved
        #######################################################
        Y = np.unique(mesh.faces)
        points = mesh.points[Y,:]
        r = np.sqrt(points[:,0]**2 + points[:,1]**2)
        theta = np.arctan2(points[:,1],points[:,0])

        boundary_data[Y,0] = (1-r)*np.cos(theta)
        boundary_data[Y,1] = (1-r)*np.sin(theta)
        boundary_data[Y,2] = 0.0

        makezero(boundary_data)
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex
        node_arranger = NodeArrangementHex(mesh.InferPolynomialDegree()-1)[0]

        Z0 = mesh.elements[0,node_arranger[0,:]]
        Y0 = Z0[~np.isclose(mesh.points[Z0,0],-np.sin(np.pi/4))]
        Y1 = Y0[~np.isclose(mesh.points[Y0,0],np.sin(np.pi/4))]
        Y2 = Y1[~np.isclose(mesh.points[Y1,1],-np.sin(np.pi/4))]
        Y3 = Y2[~np.isclose(mesh.points[Y2,1],np.sin(np.pi/4))]
        # boundary_data[Y3,:] = 0.0
        boundary_data[Y3,2] = 0.0
        boundary_data[Y3,:2] = np.NAN

        ZF = mesh.elements[-1,node_arranger[1,:]]
        Y0 = ZF[~np.isclose(mesh.points[ZF,0],-np.sin(np.pi/4))]
        Y1 = Y0[~np.isclose(mesh.points[Y0,0],np.sin(np.pi/4))]
        Y2 = Y1[~np.isclose(mesh.points[Y1,1],-np.sin(np.pi/4))]
        Y3 = Y2[~np.isclose(mesh.points[Y2,1],np.sin(np.pi/4))]
        # boundary_data[Y3,:] = 0.0
        boundary_data[Y3,2] = 0.0
        boundary_data[Y3,:2] = np.NAN


        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementFormulation(mesh,compute_post_quadrature=False)

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    solution.CurvilinearPlot(QuantityToPlot=solution.sol[:,1,-1],plot_on_faces=False, plot_points=True, point_radius=.08)
    # solution.Plot(configuration="deformed",quantity=20,plot_points=True, point_radius=.08)
    # solution.Animate(configuration="original",quantity=2,plot_points=True)
    # solution.CurvilinearPlot(interpolation_degree=20, plot_points=True, point_radius=.08)

    # mesh.points += solution.sol[:,:ndim,-1]
    # mesh.WriteHDF5(filename=PWD(__file__)+"/OneCylinder10_P"+str(p)+".mat")






def ProblemOneCylinder(p=1, incr=0):

    ndim = 3
    material = NeoHookean_2(ndim, youngs_modulus=2e6, poissons_ratio=0.3)

    mesh = Mesh()

    if incr == 0:
        if p==1:
            mesh.OneElementCylinder(nz=10)
        else:
            mesh.ReadHDF5(filename=PWD(__file__)+"/OneCylinder10_P"+str(p)+".mat")
            # mesh.GetHighOrderMesh()
    else:
        mesh.ReadHDF5("/home/roman/ZZZchecker/Cylinder_Config_"+str(incr-1)+"_P"+str(p)+".mat")      
    makezero(mesh.points)





    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        # # Y_1 = np.isclose(mesh.points[:,2],100.)
        # # boundary_data[Y_1,0] = 1.0
        # # boundary_data[Y_1,1] = 0.0
        # # boundary_data[Y_1,2] = 0.0

        # APPLY TORSIONAL FORCE
        C = mesh.InferPolynomialDegree()-1
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex, NodeArrangementQuad
        node_arranger = NodeArrangementHex(C)[0]

        ZF = mesh.elements[-1,node_arranger[1,:]]
        tpoints = mesh.points[ZF,:]
        # r = np.sqrt(tpoints[:,0]**2+tpoints[:,1]**2)
        # Y = ~np.isclose(r,1.)
        # print repr(Y)
        # exit()
        if C==0:
            Y = [False, False, False, False]
        elif C==1:
            Y = [False, False, False, False, False ,False,  True, False, False] # for p=2
        elif C==3:
            Y = np.array([False, False, False, False, False, False, False, False,  True,
                True,  True, False, False,  True,  True,  True, False, False,
                True,  True,  True, False, False, False, False], dtype=bool)
        boundary_data[ZF[Y],:] = np.NAN


        ZF_Y = ZF[~np.in1d(ZF,ZF[Y])]
        if C==0:
            ZF_Y = ZF
        # print ZF
        # print ZF[[Y]]
        # print ZF_Y
        # exit()
        tpoints = mesh.points[ZF_Y,:]
        ref = np.copy(tpoints)
        ref[:,-1] = 0

        normals = np.zeros_like(ref)
        # normals[:,0] = -ref[:,1]
        # normals[:,1] = ref[:,0]
        normals[:,0] = ref[:,1]
        normals[:,1] = -ref[:,0]

        dyf = 0.1
        boundary_data[ZF_Y,:] = dyf*normals

        # print boundary_data
        # exit()


        return boundary_data


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    formulation = DisplacementFormulation(mesh,compute_post_quadrature=False)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    solution.Plot(configuration="deformed",quantity=20, plot_points=True, colorbar=False, point_radius=.14)
    # solution.Plot(configuration="deformed",quantity=0,plot_points=True, colorbar=False, save=True, 
        # filename="/home/roman/ZZZchecker/twister_P"+str(p)+"_incr_%03d.png" % incr,show_plot=False, point_radius=.14)
    # solution.Animate(configuration="deformed",quantity=20,plot_points=True, point_radius=.1)

    # mesh.points += solution.sol[:,:ndim,-1]
    # mesh.WriteHDF5("/home/roman/ZZZchecker/Cylinder_Config_"+str(incr)+"_P"+str(p)+".mat")


    # solution.WriteVTK("/home/roman/ZPlots/HH.vtu", quantity=0)


if __name__ == "__main__":
    
    p = 1

    # ProblemData(p=p)

    # GetMeshesOneCylinder(p=p)

    ProblemOneCylinder(p=p)


    # for incr in range(50):
    #     print "INCREMENT ", incr
    #     ProblemOneCylinder(p=p,incr=incr)




