import numpy as np 
import os, imp


def ProblemData(MainData):

    # ndim - Dimension of the problem - 1D, 2D, 3D
    MainData.ndim = 2
    
    MainData.Fields = 'Mechanics'
    # MainData.Fields = 'ElectroMechanics'
    
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.Analysis = 'Dynamic'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'


    # MainData.MaterialArgs.Type = 'LinearModel'
    # MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedNeoHookean'
    # MainData.MaterialArgs.Type = 'IncrementallyLinearisedMooneyRivlin'
    # MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin_1'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
    # MainData.MaterialArgs.Type = 'MooneyRivlin'
    MainData.MaterialArgs.Type = 'NeoHookean_2'


    MainData.MaterialArgs.E  = 1.0e1
    MainData.MaterialArgs.nu = 0.30

    # MainData.MaterialArgs.E = MainData.E 
    # MainData.MaterialArgs.nu = MainData.nu 

    E = MainData.MaterialArgs.E
    nu = MainData.MaterialArgs.nu
    # GET LAME CONSTANTS
    MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    MainData.MaterialArgs.mu = E/2./(1+nu)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    class MeshInfo(object):
        MeshType = 'tri'
        Nature = 'straight'
        Reader = 'Read'
        # Reader = 'UniformHollowCircle'

        FileName = ProblemPath + '/TwoArcs_18.dat'
        # FileName = ProblemPath + '/Leaf_2.dat'
        


    class BoundaryData(object):
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'arc_length'
        CurvilinearMeshNodalSpacing = 'fekete'
        
        IGES_File = ProblemPath + '/Two_Arcs.iges'
        # IGES_File = ProblemPath + '/Leaf_2.igs'

        scale = 1000.
        condition = 3000.


        class DirichArgs(object):
            node = 0
            Applied_at = 'node' 
                                    

        class NeuArgs(object):
            pass
                

        def DirichletCriterion(self,DirichArgs):
            pass

        def ProjectionCriteria(self,mesh):
            projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
            num = mesh.edges.shape[1]
            for iedge in range(mesh.edges.shape[0]):
                x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
                y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
                x *= self.scale
                y *= self.scale
                if np.sqrt(x*x+y*y)< self.condition:
                    projection_edges[iedge]=1
            
            return projection_edges


        
        def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
            pass


    class AnalyticalSolution(object):
        class Args(object):
            pass
        def Get(self,Args):
            pass

            
    # PLACE THEM ALL INSIDE THE MAIN CLASS
    MainData.BoundaryData = BoundaryData
    MainData.AnalyticalSolution = AnalyticalSolution
    MainData.MeshInfo = MeshInfo
