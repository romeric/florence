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

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    # MainData.MeshInfo.Reader = "ReadHDF5"

    # MainData.MeshInfo.FileName = ProblemPath + '/falcon_iso.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Falcon3DIso.mat'
    # MainData.MeshInfo.FileName = '/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/falcon/falcon_big.dat'

    # MainData.MeshInfo.FileName = ProblemPath + '/falcon_iso_P'+str(MainData.C+1)+'.mat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Falcon3DIso_P'+str(MainData.C+1)+'.mat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Falcon3DIso_P'+str(MainData.C+1)+'_New.mat'
    filename = ProblemPath + '/falcon_big_P'+str(MainData.C+1)+'.mat'

    # if MainData.CurrentIncr == 1:
    #     MainData.MeshInfo.FileName = ProblemPath + '/falcon_big_P'+str(MainData.C+1)+'.mat'
    # else:
    #     MainData.MeshInfo.FileName = '/home/roman/Dropbox/Falcon3DBig_P'+str(MainData.C+1)+'.mat'

    # MainData.MeshInfo.IsHighOrder = True

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type="ReadHDF5")

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
            # if x > -20*self.scale and x < 40*self.scale and y > -30.*self.scale \
                # and y < 30.*self.scale and z > -20.*self.scale and z < 20.*self.scale:  
            if x > -10*scale and x < 30*scale and y > -20.*scale \
                and y < 20.*scale and z > -15.*scale and z < 15.*scale:   
                projection_faces[iface]=1

        return projection_faces

    cad_file = ProblemPath + '/falcon.igs'
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
    #     condition = 1.0e20 # this condition it not used

    #     IGES_File = ProblemPath + '/falcon.igs'


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
    #             # if x > -20*self.scale and x < 40*self.scale and y > -30.*self.scale \
    #                 # and y < 30.*self.scale and z > -20.*self.scale and z < 20.*self.scale:  
    #             if x > -10*self.scale and x < 30*self.scale and y > -20.*self.scale \
    #                 and y < 20.*self.scale and z > -15.*self.scale and z < 15.*self.scale:   
    #                 projection_faces[iface]=1
            
    #         return projection_faces

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver


    return mesh, material, boundary_condition