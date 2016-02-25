import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    # ndim - Dimension of the problem - 1D, 2D, 3D
    MainData.ndim = 2
    
    MainData.Fields = 'Mechanics'
    # MainData.Fields = 'ElectroMechanics'
    
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.Analysis = 'Dynamic'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
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

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    MainData.solver = solver

    return mesh, material, boundary_condition
