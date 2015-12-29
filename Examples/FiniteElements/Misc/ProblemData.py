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
    # MainData.AnalysisType = 'Linear'
    MainData.AnalysisType = 'Nonlinear'


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

        # FileName = ProblemPath + '/TwoArcs_18.dat'
        # FileName = ProblemPath + '/Half_Circle_23.dat'
        # FileName = ProblemPath + '/Half_Circle_348.dat'

        # FileName = ProblemPath + '/Mech2D_Seg0_350.dat'
        # FileName = ProblemPath + '/Mech2D_Seg0_70.dat'
        FileName = ProblemPath + '/Mech2D_Seg2_6.dat'
        # FileName = ProblemPath + '/Mesh_LeftPartWithCircle_56.dat'
        # FileName = ProblemPath + '/LeftCircle_12.dat'
        


    class BoundaryData(object):
        Type = 'nurbs'
        RequiresCAD = True
        ProjectionType = 'arc_length'
        CurvilinearMeshNodalSpacing = 'fekete'
        
        # Type = 'straight'
        # Type = 'mixed'
        # IGES_File = ProblemPath + '/Two_Arcs.iges'
        # IGES_File = ProblemPath + '/Half_Circle.igs'
        # IGES_File = ProblemPath + '/Mech2D_Seg0.igs'
        # IGES_File = ProblemPath + '/LeftPartWithCircle.igs'
        # IGES_File = ProblemPath + '/LeftCircle.iges'
        IGES_File = ProblemPath + '/Mech2D_Seg2.igs'

        # two arcs and half circle
        scale = 1000.
        condition = 3000.
        # condition = 300000.

        # mech2d_seg0 also activate multiply by 1000 in pre-process
        scale = 1.
        condition = 1e10 

        class DirichArgs(object):
            node = 0
            Applied_at = 'node' 
                                    

        class NeuArgs(object):
            points=0
            node = 0
            # Applied_at = 'face'
            Applied_at = 'node'
            #--------------------------------------------------------------------------------------------------------------------------#
            # The condition upon which Neumann is applied 
            # - tuple (first is the coordinate direction x=0,y=1,z=2 and second is value of coordinate in that direction e.g. x, y or z) 
            cond = np.array([[1,2.]])
            Loads = np.array([
                [0.2,0.,0.],
                ])
            # Number of nodes is necessary
            no_nodes = 0.
            #--------------------------------------------------------------------------------------------------------------------------#


        # Dynamic Data
        nstep = 100
        dt = 1./nstep
        drange = np.linspace(0.,60.,nstep)
        Amp = 10000.0
        DynLoad = Amp*np.sin(drange)
                

        def DirichletCriterion(self,DirichArgs):
            
            node = DirichArgs.node 
            points = DirichArgs.points 

            # REMOVE THIS
            #----------------------------------
            edges = DirichArgs.edges 
            unedges = np.unique(edges)
            inode = DirichArgs.inode

            r  = 0.5
            # r=1
            rn = np.sqrt(node[0]**2+node[1]**2)
            tol_radius = 0.1
            
            # if rn < 0.5+tol_radius and rn > 0.5 - tol_radius:
            if rn < r+tol_radius and rn > r - tol_radius:

                # print node[0], node[1]
                theta = np.arctan(node[1]/node[0])

                # Is this node on the edge
                p = np.where(unedges==inode)[0]
                if p.shape[0]!=0:
                    # Now we are on the edge
                    # x = rn*np.cos(theta)
                    # y = rn*np.sin(theta)
                    x=node[0]
                    y=node[1]
                    Lx = 1.0*r/rn*x
                    Ly = 1.0*r/rn*y
                    # print x, np.sign(x)
                    ux = np.sign(x)*abs(abs(Lx)-abs(x))
                    uy = np.sign(y)*abs(abs(Ly)-abs(y))

                    b = np.array([ux,uy])
                else: 
                    b = [None,None] 
            # elif rn < 2.0+tol_radius and rn > 2.0 - tol_radius:

            #   # print node[0], node[1]
            #   theta = np.arctan(node[1]/node[0])

            #   # Is this node on the edge
            #   p = np.where(unedges==inode)[0]
            #   if p.shape[0]!=0:
            #       # Now we are on the edge
            #       b = np.array([0.,0.])
            #   else: 
            #       b = [None,None] 
            else:   
                b = [None,None] 
            
        
            return b

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


        # class DynamicData(object):
            # nstep = 100
            # dt = 1./nstep
            # drange = np.linspace(0.,60.,nstep)
            # Amp = 100.0
            # DynLoad = Amp*np.sin(drange)





    class AnalyticalSolution(object):
        class Args(object):
            node = 0
            points = 0




        def Get(self,Args):
            node = Args.node
            ndim = 2
            nvar = 3

            m=2
            if node.size==2:
                x = node[0]
                y = node[1]
                # sol = np.array([ 0.0,x*np.sin(y),0.0])
                sol = np.array([ 0.0,0.1*y**m])
            else:
                x = node[:,0]
                y = node[:,1]
                ux = y*0.0
                # uy = x*np.sin(y)
                uy = 0.1*y**m
                sol = np.zeros((node.shape[0],nvar))
                sol[:,0] = ux
                sol[:,1] = uy

            return sol 

            
    # PLACE THEM ALL INSIDE THE MAIN CLASS
    MainData.BoundaryData = BoundaryData
    MainData.AnalyticalSolution = AnalyticalSolution
    MainData.MeshInfo = MeshInfo

    # return MainData, MeshInfo, AnalyticalSolution 