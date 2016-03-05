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

    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/f6_iso.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type="Read",reader_type_format="GID")
    mesh.face_to_surface = np.loadtxt(ProblemPath+"/f6_iso_face_to_surface_mapped.dat").astype(np.int64)


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


    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver

    return mesh, material, boundary_condition