import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 2   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.AnalysisType = 'Linear'
    MainData.AnalysisType = 'Nonlinear'

    # material = Material("LinearModel",MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
    #     E_A=2.5e05,G_A=5.0e04)
    material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        E_A=2.5e05,G_A=5.0e04)

    # MATERIAL INPUT DATA 
    # # MainData.MaterialArgs.Type = 'LinearModel'
    # # MainData.MaterialArgs.Type = 'IncrementalLinearElastic'
    # # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
    # MainData.MaterialArgs.Type = 'NeoHookean_2'
    # # MainData.MaterialArgs.Type = 'MooneyRivlin'
    # # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
    # # MainData.MaterialArgs.Type = 'BonetTranservselyIsotropicHyperElastic'
        
    

    # MainData.MaterialArgs.E  = 1.0e5
    # MainData.MaterialArgs.nu = 0.40

    # MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E
    # MainData.MaterialArgs.G_A = MainData.MaterialArgs.E/2.


    # # MainData.MaterialArgs.E = MainData.E 
    # # MainData.MaterialArgs.nu = MainData.nu
    # print 'Poisson ratio is:', MainData.MaterialArgs.nu


    # E = MainData.MaterialArgs.E
    # nu = MainData.MaterialArgs.nu
    # # E_A = MainData.MaterialArgs.E_A    


    # # GET LAME CONSTANTS
    # MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    # MainData.MaterialArgs.mu = E/2./(1+nu)


    # print (MaterialArgs.lamb)/2./(MaterialArgs.lamb+MaterialArgs.mu)

    # mu = 23.3*1000   # N/mm^2
    # lamb = 79.4*1000 # N/mm^2
    # eps_1 = 1.5*10e-11  # C/mm^2


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    # class MeshInfo(object):
    #     MeshType = 'tri'
    #     Nature = 'straight'
    #     Reader = 'Read'
    #     # Reader = 'UniformHollowCircle'

    # FileName = ProblemPath + '/Mesh_Annular_Circle_6135253.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_382526.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_23365.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_5716.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_502.dat'
    # FileName = ProblemPath + '/Mesh_Annular_Circle_312.dat'
    filename = ProblemPath + '/Mesh_Annular_Circle_75.dat'


        # MeshType = 'tet'
        # FileName = ProblemPath + '/Mesh_Cube_Tet_181.dat'
        # MeshType = 'quad'
        # FileName = ProblemPath + '/Mesh_Square_Quad_64.dat'
        # MeshType = 'hex'
        # FileName = ProblemPath + '/Rectangular_Beam_Mesh_64.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Read")


    cad_file = ProblemPath + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=1000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    MainData.solver = solver
    
        


    # class BoundaryData(object):
    #     Type = 'nurbs'
    #     RequiresCAD = True
        
    #     IGES_File = ProblemPath + '/Circle.igs'

    #     condition = 1000.
    #     scale = 1000.

    #     class DirichArgs(object):
    #         node = 0
    #         Applied_at = 'node' 
                                    

    #     class NeuArgs(object):
    #         points=0
    #         node = 0
    #         # Applied_at = 'face'
    #         Applied_at = 'node'
    #         #--------------------------------------------------------------------------------------------------------------------------#
    #         # The condition upon which Neumann is applied 
    #         # - tuple (first is the coordinate direction x=0,y=1,z=2 and second is value of coordinate in that direction e.g. x, y or z) 
    #         cond = np.array([[1,2.]])
    #         Loads = np.array([
    #             [0.2,0.,0.],
    #             ])
    #         # Number of nodes is necessary
    #         no_nodes = 0.
    #         #--------------------------------------------------------------------------------------------------------------------------#


    #     # Dynamic Data
    #     nstep = 100
    #     dt = 1./nstep
    #     drange = np.linspace(0.,60.,nstep)
    #     Amp = 10000.0
    #     DynLoad = Amp*np.sin(drange)
                

    #     def DirichletCriterion(self,DirichArgs):
            
    #         node = DirichArgs.node 
    #         points = DirichArgs.points 

    #         # REMOVE THIS
    #         #----------------------------------
    #         edges = DirichArgs.edges 
    #         unedges = np.unique(edges)
    #         inode = DirichArgs.inode

    #         r  = 0.5
    #         # r=1
    #         rn = np.sqrt(node[0]**2+node[1]**2)
    #         tol_radius = 0.1
            
    #         # if rn < 0.5+tol_radius and rn > 0.5 - tol_radius:
    #         if rn < r+tol_radius and rn > r - tol_radius:

    #             # print node[0], node[1]
    #             theta = np.arctan(node[1]/node[0])

    #             # Is this node on the edge
    #             p = np.where(unedges==inode)[0]
    #             if p.shape[0]!=0:
    #                 # Now we are on the edge
    #                 # x = rn*np.cos(theta)
    #                 # y = rn*np.sin(theta)
    #                 x=node[0]
    #                 y=node[1]
    #                 Lx = 1.0*r/rn*x
    #                 Ly = 1.0*r/rn*y
    #                 # print x, np.sign(x)
    #                 ux = np.sign(x)*abs(abs(Lx)-abs(x))
    #                 uy = np.sign(y)*abs(abs(Ly)-abs(y))

    #                 b = np.array([ux,uy])
    #             else: 
    #                 b = [None,None] 
    #         # elif rn < 2.0+tol_radius and rn > 2.0 - tol_radius:

    #         #   # print node[0], node[1]
    #         #   theta = np.arctan(node[1]/node[0])

    #         #   # Is this node on the edge
    #         #   p = np.where(unedges==inode)[0]
    #         #   if p.shape[0]!=0:
    #         #       # Now we are on the edge
    #         #       b = np.array([0.,0.])
    #         #   else: 
    #         #       b = [None,None] 
    #         else:   
    #             b = [None,None] 
            
        
    #         return b

    #     def ProjectionCriteria(self,mesh):
    #         projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
    #         num = mesh.edges.shape[1]
    #         for iedge in range(mesh.edges.shape[0]):
    #             x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
    #             y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
    #             x *= self.scale
    #             y *= self.scale
    #             if np.sqrt(x*x+y*y)< self.condition:
    #                 projection_edges[iedge,0]=1
            
    #         return projection_edges


        
    #     def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
    #         # USING THIS APPROACH YOU EITHER NEED TO APPLY FORCE (N) OR YOU SHOULD KNOW THE VALUE OF AREA (M^2)
    #         node = NeuArgs.node
    #         # Area should be specified for as many physical (no meshed faces i.e. not mesh.faces) as Neumann is applied 
    #         area = 1.0*np.array([4.,4.,100.])

    #         t=[]
    #         for i in range(0,len(NeuArgs.cond)):
    #             no_nodes = 1.0*NeuArgs.no_nodes[i] 
    #             if Analysis != 'Static':
    #                 if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
    #                     t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
    #                 else:
    #                     t = [[],[],[],[]]

    #             # Static Analysis 
    #             if Analysis=='Static':
    #                 if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
    #                     t = NeuArgs.Loads[i,:]*area[i]/no_nodes
    #                 else:
    #                     t = [[],[],[],[]]

    #         return t


    #     # class DynamicData(object):
    #         # nstep = 100
    #         # dt = 1./nstep
    #         # drange = np.linspace(0.,60.,nstep)
    #         # Amp = 100.0
    #         # DynLoad = Amp*np.sin(drange)





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
    MainData.AnalyticalSolution = AnalyticalSolution

    return mesh, material, boundary_condition