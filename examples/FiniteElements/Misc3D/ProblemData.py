import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    ndim=3
    p=3

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
    filename = ProblemPath + '/Hollow_Cylinder.dat'
    # filename = ProblemPath + '/TPipe_4006.dat'
    # filename = ProblemPath + '/TPipe_2262.dat'
    # filename = ProblemPath + "/TPipe_2_1302.dat"
    # filename = ProblemPath + "/TPipe_2_1247.dat"
    # filename = ProblemPath + "/FullTPipe.dat"
    # filename = ProblemPath + '/Cylinder.dat'
    # filename = ProblemPath + '/Revolution_1.dat'
    # filename = ProblemPath + '/Extrusion_116.dat'
    # filename = ProblemPath + '/Extrusion_2_416.dat'
    # filename = ProblemPath + '/ufc_206.dat'
    # filename = ProblemPath + '/ucp_206.dat'
    # filename = ProblemPath + '/penc.dat'
    # filename = ProblemPath + '/gopro.dat' #
    # filename = ProblemPath + '/bracketH0.dat' #

    # filename = ProblemPath + '/form1.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")
    mesh.GetHighOrderMesh(p=p)
    # mesh.SimplePlot()


    # cad_file = ProblemPath + '/Torus.igs'
    cad_file = ProblemPath + '/Hollow_Cylinder.igs'
    # IGES_File = ProblemPath + '/PipeTShape.igs'
    # IGES_File = ProblemPath + '/TPipe_2.igs'
    # IGES_File = ProblemPath + '/FullTPipe.igs'
    # IGES_File = ProblemPath + '/Cylinder.igs'
    # IGES_File = ProblemPath + '/Revolution_1.igs'
    # IGES_File = ProblemPath + '/Extrusion.igs'
    # IGES_File = ProblemPath + '/Extrusion_2.igs'
    # IGES_File = ProblemPath + '/ufc_206.igs'
    # IGES_File = ProblemPath + '/ucp_206.igs'
    # IGES_File = ProblemPath + '/Porta_Canetas.igs'
    # IGES_File = ProblemPath + '/gopro.igs' #
    # IGES_File = ProblemPath + '/bracket.igs' #

    # IGES_File = ProblemPath + '/form1.igs'

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
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    solution.CurvilinearPlot()


if __name__ == "__main__":
    ProblemData()
