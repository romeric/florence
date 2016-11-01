import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    p = 4
    ndim = 2

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = NeoHookean_2(ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # Reader = 'UniformHollowCircle'

    # FileName = ProblemPath + '/TwoArcs_18.dat'
    # FileName = ProblemPath + '/Half_Circle_23.dat'
    # FileName = ProblemPath + '/Half_Circle_348.dat'

    # FileName = ProblemPath + '/Mech2D_Seg0_350.dat'
    # FileName = ProblemPath + '/Mech2D_Seg0_70.dat'
    FileName = ProblemPath + '/Mech2D_Seg2_6.dat'
    # FileName = ProblemPath + '/Mesh_LeftPartWithCircle_56.dat'
    # FileName = ProblemPath + '/LeftCircle_12.dat'
    # FileName = ProblemPath + '/Leaf_2.dat'
    # FileName = ProblemPath + '/Two_Hole.dat'
    # FileName = ProblemPath + '/Two_Hole2.dat'
    # FileName = ProblemPath + '/Two_Hole3.dat'
    # FileName = ProblemPath + '/5_Hole.dat'
    # FileName = ProblemPath + '/5_Hole_273.dat'

    mesh = Mesh()
    mesh.Reader(filename=FileName,element_type="tri")
    mesh.points *=1000.
    mesh.GetHighOrderMesh(p=p)

    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/mesh_2d_points_p"+str(p)+".dat",
    #     mesh.points,fmt="%9.5f")
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/mesh_2d_elements_p"+str(p)+".dat",
    #     mesh.elements,fmt="%d")
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # exit()


    # cad_file = ProblemPath + '/Two_Arcs.iges'
    # cad_file = ProblemPath + '/Half_Circle.igs'
    # cad_file = ProblemPath + '/Mech2D_Seg0.igs'
    # cad_file = ProblemPath + '/LeftPartWithCircle.igs'
    # cad_file = ProblemPath + '/LeftCircle.iges'
    cad_file = ProblemPath + '/Mech2D_Seg2.igs'
    # cad_file = ProblemPath + '/Leaf_2.igs'
    # cad_file = ProblemPath + '/Two_Hole.igs'
    # cad_file = ProblemPath + '/Two_Hole3.igs'
    # cad_file = ProblemPath + '/5_Hole.igs'
# 
    # two arcs and half circle
    # scale = 1000.
    # condition = 3000.
    # condition = 3000000.

    # mech2d_seg0 also activate multiply by 1000 in pre-process
    scale = 1.
    condition = 1e10 
        
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file, scale=scale,condition=condition)
    boundary_condition.GetProjectionCriteria(mesh)

    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(analysis_nature="nonlinear",number_of_load_increments = 10,)

    # print formulation.function_spaces[0].Jm.flatten().shape
    # print formulation.function_spaces[0].AllGauss.shape
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p"+str(p)+"_2d_Jm.dat",
    #     formulation.function_spaces[0].Jm.flatten(),fmt="%9.6f")
    # np.savetxt("/home/roman/Dropbox/Fastor/benchmark/benchmark_academic/kernel_quadrature/meshes/p"+str(p)+"_2d_AllGauss.dat",
    #     formulation.function_spaces[0].AllGauss,fmt="%9.6f")
    # exit()

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
        material=material, boundary_condition=boundary_condition)

    # solution.CurvilinearPlot(QuantityToPlot=fem_solver.ScaledJacobian, plot_points=True)
    solution.CurvilinearPlot(save=True,filename="/home/roman/Dropbox/2d_mesh.eps")

if __name__ == "__main__":
    ProblemData()
