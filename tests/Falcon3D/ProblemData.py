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

    MainData.MeshInfo.FileName = ProblemPath + '/falcon_big.dat'

    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 25.4
        condition = 1.0e20 # this condition it not used

        IGES_File = ProblemPath + '/falcon.igs'


        def ProjectionCriteria(self,mesh):
            projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
            num = mesh.faces.shape[1]
            for iface in range(mesh.faces.shape[0]):
                x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
                y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
                z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
                x *= self.scale
                y *= self.scale
                z *= self.scale
                # if x > -20*self.scale and x < 40*self.scale and y > -30.*self.scale \
                    # and y < 30.*self.scale and z > -20.*self.scale and z < 20.*self.scale:  
                if x > -10*self.scale and x < 30*self.scale and y > -20.*self.scale \
                    and y < 20.*self.scale and z > -15.*self.scale and z < 15.*self.scale:   
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