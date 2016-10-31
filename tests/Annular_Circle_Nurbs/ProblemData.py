import numpy as np 
import os, imp


def ProblemData(MainData):

    MainData.ndim = 2   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # MATERIAL INPUT DATA 
    # MainData.MaterialArgs.Type = 'LinearModel'
    MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
    # MainData.MaterialArgs.Type = 'NeoHookean_1'
    # MainData.MaterialArgs.Type = 'NeoHookean_2'
    # MainData.MaterialArgs.Type = 'MooneyRivlin'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleNeoHookean'
    # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
    # MainData.MaterialArgs.Type = 'AnisotropicMooneyRivlin' 
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
    # MainData.MaterialArgs.Type = 'TranservselyIsotropicHyperElastic'
    # MainData.MaterialArgs.Type = 'BonetTranservselyIsotropicHyperElastic'
        
    
    MainData.MaterialArgs.E  = 1.0
    MainData.MaterialArgs.nu = 0.40

    # MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E
    # # # MainData.MaterialArgs.G_A = (E*(E_A*nu - E_A + E_A*nu**2 + E*nu**2))/(2*(nu + 1)*(2*E*nu**2 + E_A*nu - E_A))
    # MainData.MaterialArgs.G_A = MainData.MaterialArgs.E/2.


    # MainData.MaterialArgs.E = MainData.E 
    # MainData.MaterialArgs.nu = MainData.nu
    print 'Poisson ratio is:', MainData.MaterialArgs.nu


    E = MainData.MaterialArgs.E
    nu = MainData.MaterialArgs.nu
    # E_A = MainData.MaterialArgs.E_A    


    # GET LAME CONSTANTS
    MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    MainData.MaterialArgs.mu = E/2./(1+nu)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    class MeshInfo(object):
        MeshType = 'tri'
        Nature = 'straight'
        Reader = 'ReadSeparate'

        ConnectivityFile = ProblemPath + '/elements_circle.dat'
        CoordinatesFile = ProblemPath +'/points_circle.dat'
        MatFile = ProblemPath + '/circleTest.mat'
        EdgesFile = None
        FacesFile = None



    class BoundaryData(object):
        # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
        Type = 'nurbs'
        RequiresCAD = True
        # ProjectionType = 'orthogonal'
        ProjectionType = 'arc_length'
        # CurvilinearMeshNodalSpacing = 'fekete'
        CurvilinearMeshNodalSpacing = 'equal'

        IGES_File = ProblemPath + '/Circle.igs'
        condition = 2000.
        scale = 1000.

        class DirichArgs(object):
            node = 0
            Applied_at = 'node' 
                                    

        class NeuArgs(object):
            pass
                

        def DirichletCriterion(self,DirichArgs):
            pass


        def NURBSParameterisation(self):
            import Core.Supplementary.nurbs.cad as iga 

            circle = iga.circle(radius=1, center=None, angle=None)
            dum=np.array([4,3,2,1,0,7,6,5,8])
            control = circle.control[dum,:];    control[-1,0]=-1
            points = circle.points[dum,:];      points[-1,0] = -1


            # circle = iga.circle(radius=1000, center=None, angle=None)
            # dum=np.array([4,3,2,1,0,7,6,5,8])
            # control = circle.control[dum,:];  control[-1,0]=-1000.
            # points = circle.points[dum,:];        points[-1,0] = -1000.

            # control = circle.control; points=circle.points

            return [({'U':circle.knots,'Pw':control,'start':0,'end':1,'points':points,'weights':circle.weights,'degree':2})]

        def NURBSCondition(self,x):
            return np.sqrt(x[:,0]**2 + x[:,1]**2) < 2
            # return np.sqrt(x[:,0]**2 + x[:,1]**2) < 2000


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
            # USING THIS APPROACH YOU EITHER NEED TO APPLY FORCE (N) OR YOU SHOULD KNOW THE VALUE OF AREA (M^2)
            node = NeuArgs.node
            # Area should be specified for as many physical (no meshed faces i.e. not mesh.faces) as Neumann is applied 
            area = 1.0*np.array([4.,4.,100.])

            t=[]
            for i in range(0,len(NeuArgs.cond)):
                no_nodes = 1.0*NeuArgs.no_nodes[i] 
                if Analysis != 'Static':
                    if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
                        t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
                    else:
                        t = [[],[],[],[]]

                # Static Analysis 
                if Analysis=='Static':
                    if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
                        t = NeuArgs.Loads[i,:]*area[i]/no_nodes
                    else:
                        t = [[],[],[],[]]

            return t


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

    # return MainData, MeshInfo, AnalyticalSolution 