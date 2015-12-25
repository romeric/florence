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
    # MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedNeoHookean'
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedMooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
    # MainData.MaterialArgs.Type = 'NeoHookean_1'
    # MainData.MaterialArgs.Type = 'NeoHookean_2'
    MainData.MaterialArgs.Type = 'MooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
    # MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicHyperElastic'
    # MainData.MaterialArgs.Type = 'JavierTranservselyIsotropicHyperElastic'

    MainData.MaterialArgs.E  = 1.0e5
    MainData.MaterialArgs.nu = 0.35

    MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E

    E = MainData.MaterialArgs.E
    nu = MainData.MaterialArgs.nu

    # GET LAME CONSTANTS
    MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    MainData.MaterialArgs.mu = E/2./(1+nu)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    MainData.MeshInfo.MeshType = "tet"
    # MainData.MeshInfo.Reader = "Read"
    MainData.MeshInfo.Reader = "Sphere"

    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_393401.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_123962.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_56407.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_1473.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_181.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cube_Tet_12.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Sphere_8219.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Sphere_1483.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Sphere_880.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Sphere_115.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Sphere_16.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Circular_Holes.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Mesh_Cyl_Hole_1528.dat'
    # MainData.MeshInfo.FileName = ProblemPath + '/Sphere.dat'
        
        


    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'orthogonal'

        scale = 1000.
        condition = 1000.

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