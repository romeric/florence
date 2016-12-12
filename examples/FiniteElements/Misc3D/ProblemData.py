import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    ndim=3
    p=2

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)


    ProblemPath = PWD(__file__)

    # filename = ProblemPath + '/Torus_612.dat' # Torus
    # filename = ProblemPath + '/Hollow_Cylinder.dat'
    # filename = ProblemPath + '/TPipe_4006.dat'
    # filename = ProblemPath + '/TPipe_2262.dat'
    # filename = ProblemPath + "/TPipe_2_1302.dat"
    # filename = ProblemPath + "/TPipe_2_1247.dat"
    # filename = ProblemPath + "/FullTPipe.dat"
    # filename = ProblemPath + '/Cylinder.dat'
    # filename = ProblemPath + '/Revolution_1.dat'
    filename = ProblemPath + '/Extrusion_116.dat'
    # filename = ProblemPath + '/Extrusion_2_416.dat'
    # filename = ProblemPath + '/ufc_206.dat'
    # filename = ProblemPath + '/ucp_206.dat'
    # filename = ProblemPath + '/penc.dat'
    # filename = ProblemPath + '/gopro.dat' #
    # filename = ProblemPath + '/bracketH0.dat' #

    # filename = ProblemPath + '/form1.dat'

    # filename = ProblemPath + '/hand.mesh'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")
    # mesh.ReadGmsh(filename)
    # mesh.element_type="tet"
    mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/mesh_hand_3d_points_p"+str(p)+".dat",
    #     mesh.points,fmt="%9.5f")
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/mesh_hand_3d_elements_p"+str(p)+".dat",
    #     mesh.elements,fmt="%d")

    # exit()



    # cad_file = ProblemPath + '/Torus.igs'
    # cad_file = ProblemPath + '/Hollow_Cylinder.igs'
    # cad_file = ProblemPath + '/PipeTShape.igs'
    # cad_file = ProblemPath + '/TPipe_2.igs'
    # cad_file = ProblemPath + '/FullTPipe.igs'
    # cad_file = ProblemPath + '/Cylinder.igs'
    # cad_file = ProblemPath + '/Revolution_1.igs'
    cad_file = ProblemPath + '/Extrusion.igs'
    # cad_file = ProblemPath + '/Extrusion_2.igs'
    # cad_file = ProblemPath + '/ufc_206.igs'
    # cad_file = ProblemPath + '/ucp_206.igs'
    # cad_file = ProblemPath + '/Porta_Canetas.igs'
    # cad_file = ProblemPath + '/gopro.igs' #
    # cad_file = ProblemPath + '/bracket.igs' #

    # cad_file = ProblemPath + '/form1.igs'

    # sphere
    # scale = 1000.
    # condition = 1000.

    # torus
    # scale = 1000.
    # condition = 1.0e20

    # pipe t-shape, hollow cylinder, torus
    scale = 1000.
    condition = 1.e020

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=scale,condition=condition,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)

    formulation = DisplacementFormulation(mesh)
    # function_space = FunctionSpace(mesh, quadrature, p=C+1)
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p"+str(p)+"_3d_Jm.dat",
    #     formulation.function_spaces[0].Jm.flatten(),fmt="%9.6f")
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p"+str(p)+"_3d_AllGauss.dat",
    #     formulation.function_spaces[0].AllGauss,fmt="%9.6f")

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.sol = np.zeros_like(solution.sol)
    # solution.sol = solution.sol/1.5
    solution.CurvilinearPlot()
    # solution.CurvilinearPlot(plot_surfaces=False)

    # mesh.points += solution.sol[:,:ndim,-1]
    # mesh.WriteHDF5(PWD(__file__)+"/Extrusion.mat")





def ProblemData_curved(*args, **kwargs):

    ndim=3
    p=2

    material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)

    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/Extrusion.mat'
    mesh = Mesh()
    mesh.ReadHDF5(filename)
    mesh.GetFaces()
    # print mesh.Bounds

    # from Florence.Tensor import itemfreq
    # lmesh = mesh.GetLinearMesh()
    # for i in range(lmesh.nelem):
    #     coords = lmesh.points[lmesh.elements[i,:],:]
    #     xx = itemfreq(coords[:,2])
    #     for j in range(xx.shape[0]):
    #         if np.isclose(xx[j,0],8.0) and xx[j,1]==3:
    #             print i, ",",

    # from Florence.Tensor import itemfreq
    # lmesh = mesh.GetLinearMesh()
    # for i in range(lmesh.nelem):
    #     coords = lmesh.points[lmesh.elements[i,:],:]
    #     xx = itemfreq(coords[:,1])
    #     for j in range(xx.shape[0]):
    #         if np.isclose(xx[j,0],30.0) and xx[j,1]==2:
    #             print i, ",",

    # exit()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.isclose(mesh.points[:,0],-10.99952585)
        boundary_data[Y_0,0] = 5.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_0 = np.isclose(mesh.points[:,0],10.99952585)
        boundary_data[Y_0,0] = -5.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = -2.0
        boundary_data[Y_1,2] = 0.0

        Y_1 = np.isclose(mesh.points[:,1],30.)
        boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = 2.0
        boundary_data[Y_1,2] = 0.0

        Y_1 = np.isclose(mesh.points[:,2],8.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = 0.0
        boundary_data[Y_1,2] = -1.0

        return boundary_data

    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=1, analysis_nature="nonlinear")

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.sol = solution.sol/1.5
    # solution.sol = np.zeros_like(solution.sol)
    
    # solution.Plot(configuration="deformed")
    # solution.CurvilinearPlot(plot_surfaces=False)
    # solution.CurvilinearPlot()
    

    # To make this work just change the following line in CurvilinearPlotTet:
        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars = Uplot,
                    # line_width=point_line_width,colormap="summer")
    # to this:
        # trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot, scalars = Uplot,
                    # line_width=point_line_width,color=color)

    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab

    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))

    solution.CurvilinearPlotTet(mesh, solution.sol, 
        QuantityToPlot=np.zeros(mesh.all_faces.shape[0]), show_plot=False, figure=figure)

    rmesh, rsol = mesh.GetLocalisedMesh([25,70],solution=solution.sol)
    rmesh.GetFaces()
    solution.CurvilinearPlotTet(rmesh, rsol, 
        QuantityToPlot=np.zeros(rmesh.all_faces.shape[0]),
        ProjectionFlags=np.ones(rmesh.all_faces.shape[0]),color=(227./255, 66./255, 52./255),show_plot=False,figure=figure)

    bmesh, bsol = mesh.GetLocalisedMesh(73,solution=solution.sol)
    bmesh.GetFaces()
    solution.CurvilinearPlotTet(bmesh, bsol, 
        QuantityToPlot=np.zeros(bmesh.all_faces.shape[0]),
        ProjectionFlags=np.ones(bmesh.all_faces.shape[0]),color=(84./255, 90./255, 167./255),show_plot=False,figure=figure)

    mlab.show()
    
    # 25, 70 are the elements with red faces
    # 73 is the element with blue faces





if __name__ == "__main__":
    # ProblemData()
    ProblemData_curved()
