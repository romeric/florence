import numpy as np 
import os, imp
from Core.MaterialLibrary import *


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

    MainData.MeshInfo.FileName = ProblemPath + '/mechanicalComplex.dat'
        


    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 1.
        condition = 1.0e10 

        IGES_File = ProblemPath + '/mechanicalComplex.igs'


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
                if np.sqrt(x*x+y*y+z*z)< self.condition:    
                    projection_faces[iface]=1
            
            return projection_faces


        def PlottingCriteria(self,mesh):
            """Which faces need plotting"""

            plotting_faces = np.zeros((mesh.all_faces.shape[0],1),dtype=np.uint64)
            num = mesh.all_faces.shape[1]
            for iface in range(mesh.all_faces.shape[0]):
                x = np.sum(mesh.points[mesh.all_faces[iface,:],0])/num
                y = np.sum(mesh.points[mesh.all_faces[iface,:],1])/num
                z = np.sum(mesh.points[mesh.all_faces[iface,:],2])/num

                # x = np.min(mesh.points[mesh.faces[iface,:],0])
                x *= self.scale
                y *= self.scale
                z *= self.scale 
                if z < -1.0:
                    plotting_faces[iface]=1

            return plotting_faces



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

    return material