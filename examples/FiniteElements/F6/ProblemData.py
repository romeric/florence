import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(*args, **kwargs):


    ndim = 3
    p = 2

    # material = LinearModel(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    # MainData.MeshInfo.Reader = "ReadHDF5"

    # filename = ProblemPath + '/f6.dat'
    # filename = '/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/f6BoundaryLayer/f6BL.dat'
    # filename = '/home/roman/f6BL.dat'
    filename = '/media/MATLAB/f6BL.dat'
    # filename = "/home/roman/Dropbox/f6BLayer1.mat"
    # filename = "/home/roman/f6BL_P3.mat"
    # filename = "/home/roman/f6BL_P4.mat"
    # filename = ProblemPath + '/f6_iso.dat'
    # filename = ProblemPath + '/f6_iso_P4.mat'
    # filename = "/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/f6Isotropic/f6_iso.dat"
    # filename = "/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/f6Isotropic/f6_iso_P"+str(MainData.C+1)+'.mat'

    # filename = ProblemPath + '/f6_P'+str(MainData.C+1)+'.mat'

    # Layers for anistropic mesh
    # filename = "/home/roman/LayerSolution/Layer_dd/f6BL_Layer_dd_P3.mat"

    # MainData.MeshInfo.IsHighOrder = True


    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type="GID")
    mesh.GetHighOrderMesh(p=p)

    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/face_to_surface_mapped.dat").astype(np.int64)
    # mesh.face_to_surface = np.loadtxt(ProblemPath+"/f6_iso_face_to_surface_mapped.dat").astype(np.int64)
    mesh.face_to_surface = np.loadtxt(ProblemPath+"/f6BL_face_to_surface_mapped.dat").astype(np.int64)


    def ProjectionCriteria(mesh):
        projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
        num = mesh.faces.shape[1]
        for iface in range(mesh.faces.shape[0]):
            Y = np.where(abs(mesh.points[mesh.faces[iface,:3],1])<1e-07)[0]
            if Y.shape[0]!=3:
                projection_faces[iface]=1
        
        return projection_faces


    cad_file = ProblemPath + '/f6.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=1.0,project_on_curves=False,solve_for_planar_faces=False)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=False)
        


    #     def ProjectionCriteria(self,mesh):
    #         projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    #         num = mesh.faces.shape[1]
    #         for iface in range(mesh.faces.shape[0]):
    #             Y = np.where(abs(mesh.points[mesh.faces[iface,:3],1])<1e-07)[0]
    #             if Y.shape[0]!=3:
    #                 projection_faces[iface]=1
            
    #         return projection_faces

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    TotalDisp = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition, solver=solver)


if __name__ == "__main__":
    ProblemData()