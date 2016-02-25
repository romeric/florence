import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 2   
    MainData.Fields = 'Mechanics'
    # MainData.Fields = 'ElectroMechanics'
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.Analysis = 'Dynamic'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)



    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # reader_type = "Read"
    # MainData.MeshInfo.Reader = "ReadHighOrderMesh"
    # MainData.MeshInfo.Format = "GID"
    
    # filename = ProblemPath + '/sd7003_Stretch25.dat'
    # filename = ProblemPath + '/sd7003_Stretch50.dat'
    # filename = ProblemPath + '/sd7003_Stretch100.dat'
    # filename = ProblemPath + '/sd7003_Stretch200.dat'
    # filename = ProblemPath + '/sd7003_Stretch400.dat'
    # filename = ProblemPath + '/sd7003_Stretch800.dat'
    filename = ProblemPath + '/sd7003_Stretch1600.dat'

    # MainData.MeshInfo.IsHighOrder = True

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Read",
        reader_type_format="GID")


    def ProjectionCriteria(mesh,boundary_condition):
        projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
        num = mesh.edges.shape[1]
        for iedge in range(mesh.edges.shape[0]):
            x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
            y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
            x *= boundary_condition.scale_value_on_projection
            y *= boundary_condition.scale_value_on_projection 
            # if x > -510 and x < 510 and y > -10 and y < 10:
            if x > -630 and x < 830 and y > -100 and y < 300:   
                projection_edges[iedge]=1
        
        return projection_edges

    IGES_File = ProblemPath + '/sd7003.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(IGES_File,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=True)

    solver = LinearSolver()
    MainData.solver = solver

    # class BoundaryData(object):
    #     # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
    #     Type = 'nurbs'
    #     RequiresCAD = True
    #     # ProjectionType = 'orthogonal' # gives worse quality for wing 2D - checked
    #     ProjectionType = 'arc_length'
    #     CurvilinearMeshNodalSpacing = 'fekete'

    #     scale = 1000.
    #     condition = 1 # this condition is not used

    #     IGES_File = ProblemPath + '/sd7003.igs'

    #     class DirichArgs(object):
    #         node = 0
    #         Applied_at = 'node' 
                                    

    #     class NeuArgs(object):
    #         pass
                

    #     def DirichletCriterion(self,DirichArgs):
    #         pass


    #     def NURBSParameterisation(self):
    #         pass

    #     def NURBSCondition(self,x):
    #         return np.sqrt(x[:,0]**2 + x[:,1]**2) < 0.1
            

    #     def ProjectionCriteria(self,mesh):
    #         projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
    #         num = mesh.edges.shape[1]
    #         for iedge in range(mesh.edges.shape[0]):
    #             x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
    #             y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
    #             x *= self.scale
    #             y *= self.scale 
    #             # if x > -510 and x < 510 and y > -10 and y < 10:
    #             if x > -630 and x < 830 and y > -100 and y < 300:   
    #                 projection_edges[iedge]=1
            
    #         return projection_edges

        
    #     def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
    #         pass


    return mesh, material, boundary_condition
