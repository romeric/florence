import numpy as np 
import os, imp


def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # MATERIAL INPUT DATA 
    # MainData.MaterialArgs.Type = 'LinearModel'
    MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
    # MainData.MaterialArgs.Type = 'NeoHookean_1'
    # MainData.MaterialArgs.Type = 'NeoHookean_2'
    # MainData.MaterialArgs.Type = 'MooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
    # MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicHyperElastic'
    # MainData.MaterialArgs.Type = 'JavierTranservselyIsotropicHyperElastic'

    MainData.MaterialArgs.E  = 1.0e5
    MainData.MaterialArgs.nu = 0.42

    MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E

    E = MainData.MaterialArgs.E
    nu = MainData.MaterialArgs.nu

    # GET LAME CONSTANTS
    MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    MainData.MaterialArgs.mu = E/2./(1+nu)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    # MainData.MeshInfo.Format = "GID"
    MainData.MeshInfo.Reader = "ReadHDF5"

    # MainData.MeshInfo.FileName = ProblemPath + '/drill.dat'

    MainData.MeshInfo.FileName = ProblemPath + '/drill_P'+str(MainData.C+1)+'.mat'

    MainData.MeshInfo.IsHighOrder = True
        


    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 1.0
        condition = 1.0e10 # this condition it not used

        IGES_File = ProblemPath + '/drill.igs'


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