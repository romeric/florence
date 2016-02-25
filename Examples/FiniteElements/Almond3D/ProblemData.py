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

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485)
    material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.485,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    # MainData.MeshInfo.Reader = "ReadHDF5"

    filename = ProblemPath + '/almond_H1.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/almond_H2.dat'

    # MainData.MeshInfo.FileName = ProblemPath + '/Almond3D_H1.mat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Almond3D_H2.mat' 

    # MainData.MeshInfo.FileName = ProblemPath + '/Almond3D_H1_P'+str(MainData.C+1)+'.mat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Almond3D_H2_P'+str(MainData.C+1)+'.mat'

    # MainData.MeshInfo.IsHighOrder = True 

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type_format="GID")

    def ProjectionCriteria(mesh,boundary_condition):
        projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
        num = mesh.faces.shape[1]
        scale = boundary_condition.scale_value_on_projection
        for iface in range(mesh.faces.shape[0]):
            x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
            y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
            z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
            x *= scale
            y *= scale
            z *= scale 
            if x > -2.5*scale and x < 2.5*scale and y > -2.*scale \
                and y < 2.*scale and z > -2.*scale and z < 2.*scale:    
                projection_faces[iface]=1
        
        return projection_faces

    cad_file = ProblemPath + '/almond.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=25.4,project_on_curves=False,solve_for_planar_faces=False)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=True)
        


    # class BoundaryData(object):
    #     # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
    #     Type = 'nurbs'
    #     RequiresCAD = True
    #     ProjectionType = 'orthogonal'

    #     scale = 25.4
    #     condition = 2000. # this condition it not used

    #     IGES_File = ProblemPath + '/almond.igs'


    #     def ProjectionCriteria(self,mesh):
    #         projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    #         num = mesh.faces.shape[1]
    #         for iface in range(mesh.faces.shape[0]):
    #             x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
    #             y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
    #             z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
    #             x *= self.scale
    #             y *= self.scale
    #             z *= self.scale 
    #             # if np.sqrt(x*x+y*y+z*z)< self.condition:
    #             if x > -2.5*self.scale and x < 2.5*self.scale and y > -2.*self.scale \
    #                 and y < 2.*self.scale and z > -2.*self.scale and z < 2.*self.scale:    
    #                 projection_faces[iface]=1
            
    #         return projection_faces


    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver


    return mesh, material, boundary_condition