import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver

def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.AnalysisType = 'Linear'
    MainData.AnalysisType = 'Nonlinear'

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    MainData.MeshInfo.Reader = "Sphere"
    
    MainData.MeshInfo.FileName = ProblemPath + '/Sphere.dat'
        
        


    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 1000.
        condition = 1e10

        IGES_File = ProblemPath + '/Sphere.igs'


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

            mesh.GetFacesTet()
            corr_faces = mesh.faces
            # corr_faces = mesh.all_faces
            plotting_faces = np.zeros((corr_faces.shape[0],1),dtype=np.uint64)
            num = corr_faces.shape[1]
            for iface in range(corr_faces.shape[0]):
                x = np.sum(mesh.points[corr_faces[iface,:],0])/num
                y = np.sum(mesh.points[corr_faces[iface,:],1])/num
                z = np.sum(mesh.points[corr_faces[iface,:],2])/num

                # x = np.min(mesh.points[mesh.faces[iface,:],0])
                x *= self.scale
                y *= self.scale
                z *= self.scale 
                if x < 0.:
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