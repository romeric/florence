import numpy as np 
import os, imp
from Core.MaterialLibrary import *
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver

def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'

    material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.45)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    MainData.MeshInfo.MeshType = "tet"
    MainData.MeshInfo.Reader = "Read"
    MainData.MeshInfo.Format = "GID"

    MainData.MeshInfo.FileName = ProblemPath + '/f6_iso.dat'       


    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 1.0
        condition = 1.0e10 # this condition it not used

        IGES_File = ProblemPath + '/f6.igs'


        def ProjectionCriteria(self,mesh):
            projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
            num = mesh.faces.shape[1]
            for iface in range(mesh.faces.shape[0]):
                Y = np.where(abs(mesh.points[mesh.faces[iface,:3],1])<1e-07)[0]
                if Y.shape[0]!=3:
                    projection_faces[iface]=1
            
            return projection_faces



        class DirichArgs(object):
            pass
                                    
        class NeuArgs(object):
            pass

        def DirichletCriterion(self,DirichArgs):
            pass

        def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
            pass


    class AnalyticalSolution(object):
        class Args(object):
            node = 0
            points = 0

        def Get(self,Args):
            pass

            
    # PLACE THEM ALL INSIDE THE MAIN CLASS
    MainData.BoundaryData = BoundaryData
    MainData.AnalyticalSolution = AnalyticalSolution