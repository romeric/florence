import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(*args, **kwargs):

    ndim = 3
    p = 2

    # material = LinearModel(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.45,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = PWD(__file__)
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    # MainData.MeshInfo.Reader = "ReadHDF5"

    # MainData.MeshInfo.FileName = ProblemPath + '/mechanicalComplex.dat'
    filename = ProblemPath + '/mechanicalComplex_P'+str(p)+'.mat'

    # if MainData.CurrentIncr == 1:
    #     MainData.MeshInfo.FileName = ProblemPath + '/mechanicalComplex_P'+str(MainData.C+1)+'.mat'
    # else:
    #     MainData.MeshInfo.FileName = '/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'.mat'

    # MainData.MeshInfo.IsHighOrder = True 

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type="ReadHDF5")
    face_to_surface = np.loadtxt(ProblemPath+"/face_to_surface_mapped.dat").astype(np.int64)
    mesh.GetHighOrderMesh(p=p)


    cad_file = ProblemPath + '/mechanicalComplex.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=1.0,project_on_curves=True,solve_for_planar_faces=True, modify_linear_mesh_on_projection=False)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    TotalDisp = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition, solver=solver)


if __name__ == "__main__":
    ProblemData()