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
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedNeoHookean'
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedMooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
    # MainData.MaterialArgs.Type = 'NeoHookean_1'
    # MainData.MaterialArgs.Type = 'NeoHookean_2'
    # MainData.MaterialArgs.Type = 'MooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
    # MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicHyperElastic'
    # MainData.MaterialArgs.Type = 'JavierTranservselyIsotropicHyperElastic'

    MainData.MaterialArgs.E  = 1.0e1
    MainData.MaterialArgs.nu = 0.4

    # MainData.MaterialArgs.E = MainData.E 
    # MainData.MaterialArgs.nu = MainData.nu
    # print 'Poisson ratio is:', MainData.MaterialArgs.nu


    E = MainData.MaterialArgs.E
    nu = MainData.MaterialArgs.nu

    # GET LAME CONSTANTS
    MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    MainData.MaterialArgs.mu = E/2./(1+nu)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    class MeshInfo(object):
        # MeshType = 'tri'
        Nature = 'straight'
        Reader = 'Read'
        # Format = 'GID'
        # Reader = 'UniformHollowCircle'

        MeshType = 'tet'
        # FileName = ProblemPath + '/Mesh_Cube_Tet_181.dat'
        # FileName = ProblemPath + '/Sphere_1483.dat'
        # FileName = ProblemPath + '/Torus_612.dat' # Torus
        # FileName = ProblemPath + '/Torus_check.dat'
        # FileName = ProblemPath + '/TPipe_4006.dat'
        # FileName = ProblemPath + '/TPipe_2262.dat'
        FileName = ProblemPath + '/Hollow_Cylinder.dat'
        # FileName = ProblemPath + "/TPipe_2_1302.dat"
        # FileName = ProblemPath + "/TPipe_2_1247.dat"
        # FileName = ProblemPath + "/FullTPipe.dat"
        # FileName = ProblemPath + '/Cylinder.dat'
        # FileName = ProblemPath + '/Revolution_1.dat'
        # FileName = ProblemPath + '/Extrusion_116.dat'
        # FileName = ProblemPath + '/Extrusion_2_416.dat'
        # FileName = ProblemPath + '/ufc_206.dat'
        # FileName = ProblemPath + '/ucp_206.dat'
        # FileName = ProblemPath + '/penc.dat'
        # FileName = ProblemPath + '/gopro.dat' #
        # FileName = ProblemPath + '/bracketH0.dat' #

        # FileName = ProblemPath + '/form1.dat'


    class BoundaryData(object):
        Type = 'nurbs'
        RequiresCAD = True
        CurvilinearMeshNodalSpacing = 'equal'
        
        # IGES_File = ProblemPath + '/Sphere.igs'
        # IGES_File = ProblemPath + '/Torus.igs'
        # IGES_File = ProblemPath + '/PipeTShape.igs'
        # IGES_File = ProblemPath + '/TPipe_2.igs'
        # IGES_File = ProblemPath + '/FullTPipe.igs'
        IGES_File = ProblemPath + '/Hollow_Cylinder.igs'
        # IGES_File = ProblemPath + '/Cylinder.igs'
        # IGES_File = ProblemPath + '/Revolution_1.igs'
        # IGES_File = ProblemPath + '/Extrusion.igs'
        # IGES_File = ProblemPath + '/Extrusion_2.igs'
        # IGES_File = ProblemPath + '/ufc_206.igs'
        # IGES_File = ProblemPath + '/ucp_206.igs'
        # IGES_File = ProblemPath + '/Porta_Canetas.igs'
        # IGES_File = ProblemPath + '/gopro.igs' #
        # IGES_File = ProblemPath + '/bracket.igs' #

        # IGES_File = ProblemPath + '/form1.igs'

        # sphere
        # scale = 1000.
        # condition = 1000.

        # torus
        # scale = 1000.
        # condition = 1.0e20

        # pipe t-shape
        scale = 1000.
        condition = 1.e020

        class DirichArgs(object):
            node = 0
            Applied_at = 'node' 
                                    

        class NeuArgs(object):
            pass

        def DirichletCriterion(self,DirichArgs):
            pass

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
    MainData.MeshInfo = MeshInfo

