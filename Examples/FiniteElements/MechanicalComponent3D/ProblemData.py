import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    # MainData.MeshInfo.Reader = "ReadHDF5"

    # MainData.MeshInfo.FileName = ProblemPath + '/mechanicalComplex.dat'
    filename = ProblemPath + '/mechanicalComplex_P'+str(MainData.C+1)+'.mat'

    # if MainData.CurrentIncr == 1:
    #     MainData.MeshInfo.FileName = ProblemPath + '/mechanicalComplex_P'+str(MainData.C+1)+'.mat'
    # else:
    #     MainData.MeshInfo.FileName = '/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'.mat'

    # MainData.MeshInfo.IsHighOrder = True 

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type="ReadHDF5")
    face_to_surface = np.loadtxt(ProblemPath+"/face_to_surface_mapped.dat").astype(np.int64)  


    cad_file = ProblemPath + '/mechanicalComplex.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=1.0,project_on_curves=True,solve_for_planar_faces=True, modify_linear_mesh_on_projection=False)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver
        
        
    return mesh, material, boundary_condition