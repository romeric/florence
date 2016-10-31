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
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
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
    solution.sol = solution.sol/1.5
    # solution.CurvilinearPlot()
    solution.CurvilinearPlot(plot_surfaces=False)


if __name__ == "__main__":
    ProblemData()
