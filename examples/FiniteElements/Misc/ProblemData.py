import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    p = 3
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
    # FileName = ProblemPath + '/Mech2D_Seg2_6.dat'
    # FileName = ProblemPath + '/Mesh_LeftPartWithCircle_56.dat'
    # FileName = ProblemPath + '/LeftCircle_12.dat'
    # FileName = ProblemPath + '/Leaf_2.dat'
    # FileName = ProblemPath + '/Two_Hole.dat'
    # FileName = ProblemPath + '/Two_Hole2.dat'
    # FileName = ProblemPath + '/Two_Hole3.dat'
    # FileName = ProblemPath + '/5_Hole.dat'
    FileName = ProblemPath + '/5_Hole_273.dat'

    mesh = Mesh()
    mesh.Reader(filename=FileName,element_type="tri")
    mesh.points *=1000.
    mesh.GetHighOrderMesh(p=p)


    # cad_file = ProblemPath + '/Two_Arcs.iges'
    # cad_file = ProblemPath + '/Half_Circle.igs'
    # cad_file = ProblemPath + '/Mech2D_Seg0.igs'
    # cad_file = ProblemPath + '/LeftPartWithCircle.igs'
    # cad_file = ProblemPath + '/LeftCircle.iges'
    # cad_file = ProblemPath + '/Mech2D_Seg2.igs'
    # cad_file = ProblemPath + '/Leaf_2.igs'
    # cad_file = ProblemPath + '/Two_Hole.igs'
    # cad_file = ProblemPath + '/Two_Hole3.igs'
    cad_file = ProblemPath + '/5_Hole.igs'

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
    fem_solver = FEMSolver(analysis_nature="linear")

    TotalDisp = fem_solver.Solve(formulation=formulation, mesh=mesh, 
        material=material, boundary_condition=boundary_condition)

if __name__ == "__main__":
    ProblemData()
