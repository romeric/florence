import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from scipy.io import loadmat
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def ProblemData(p=1):


    # from time import time
    from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex
    from Florence.QuadratureRules.GaussLobattoPoints import GaussLobattoPointsHex
    from Florence.FunctionSpace import HexLagrangeGaussLobatto as Hex

    # Hex.LagrangeGaussLobatto(0,-1,1,1)
    # print Hex.GradLagrangeGaussLobatto(0,-1,-1,-1)

    # NodeArrangementHex(1)
    # t = time()
    # GaussLobattoPointsHex(20)
    # print time() - t

    # mesh = Mesh()
    # mesh.Wing()



    ndim = 3

    # material = Steinmann(ndim,mu=2.3*10e+04,lamb=8.0*10.0e+04, eps_1=1505*10.0e-11, c1=0.0, c2=0.0, rho=7.5*10e-6)
    material = NeoHookean_2(ndim, mu=2.3*10e+04, lamb=8.0*10.0e+04)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_125.dat'
    # filename = ProblemPath + '/Mesh_1.dat'                    
    filename = ProblemPath + '/Mesh_8.dat'                    
    # filename = ProblemPath + '/Mesh_64.dat'                   
    # filename = ProblemPath + '/Mesh_1000.dat'
    # filename = ProblemPath + '/Mesh_8000.dat'


    mesh = Mesh()
    mesh.Reader(filename, "hex")
    # print mesh.edges.shape
    # mesh.GetHighOrderMesh(p=p)
    # print mesh.Bounds
    # print mesh.elements
    # print mesh.faces
    # mesh.faces = None
    # mesh.GetFacesHex()
    # mesh.GetEdgesHex()
    # mesh.GetBoundaryEdgesHex()
    # mesh.GetBoundaryEdgesHex()
    # print mesh.all_edges
    # print mesh.edges.shape
    # print mesh.all_faces.shape
    # mesh.Areas()

    mesh.GetHighOrderMesh(p=2)
    # print mesh.faces.shape
    # exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        Y_0 = np.isclose(mesh.points[:,2],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        Y_1 = np.isclose(mesh.points[:,2],10.)
        boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = -5.0
        # boundary_data[Y_1,2] = -5.0
        boundary_data[Y_1,2] = 0.0

        # Electromechanics
        # Y_0 = np.isclose(mesh.points[:,2],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 0.
        # boundary_data[Y_0,3] = 1.
        # Y_1 = np.isclose(mesh.points[:,2],10.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = -5.0
        # boundary_data[Y_1,2] = 0.0
        # boundary_data[Y_1,3] = -1.0

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0],formulation.quadrature_rules[0],mesh)
    # exit()

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=6,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.sol *= 1e5 
    # sol = np.copy(solution.sol[:,:,-1])
    # makezero(sol,tol=1.0e-9)
    # print repr(sol)
    # print sol
    
    # import os
    # os.environ['ETS_TOOLKIT'] = 'qt4'
    # from mayavi import mlab

    # x,y,z = np.meshgrid(mesh.points[:,0],mesh.points[:,1],mesh.points[:,2])
    # x,y = np.meshgrid(mesh.points[:,0],mesh.points[:,1])
    # z = np.sin(x)
    # print x.shape
    # mlab.mesh(x,y,z)
    # mlab.show()

    solution.CurvilinearPlot(QuantityToPlot=solution.sol[:,1,-1],plot_on_faces=False)
    # solution.WriteVTK(filename="/home/roman/ZZZchecker/HE.vtu", quantity=10)
    # solution.WriteVTK(filename="/home/roman/Dropbox/HE.vtu", quantity=10)


if __name__ == "__main__":
    class MainData():
        C = 1

    ProblemData(p=MainData.C+1)













   # class BoundaryData(object):
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
    #         # cond = np.array([[2,10.],[1,2.],[0,2.]])
    #         # cond = np.array([[1,2.]])
    #         # cond = np.array([[0,2.]])
    #         cond = np.array([[2,10.]])  
    #         # cond = np.array([[1,2.],[1,0.]])  
    #         # Loads corresponding to cond
    #         # Loads = np.array([
    #         #   [0.,0.,20000.e0,-0.],
    #         #   ])
    #         Loads = np.array([
    #             [0.,3500.,0.e0,0.01],
    #             ])
    #         # Loads = np.array([
    #         #   [0.,0.,0.2e0,-0.],
    #         #   [0.,0.,0.e0,-0.1]
    #         #   ])
    #         # Loads = np.array([
    #         #   [0.,0.,0.e0,-0.05],
    #         #   [0.,0.,0.e0,-0.05],
    #         #   [0.,0.,0.e0,-0.2]
    #         #   ])  
    #         # Number of nodes is necessary
    #         no_nodes = 0.
    #         #--------------------------------------------------------------------------------------------------------------------------#
                

    #     def DirichletCriterion(self,DirichArgs):
    #         node = DirichArgs.node 
    #         mesh_points = DirichArgs.points 

    #         if np.allclose(node[2],0.0):
    #             b = np.array([0.,0.,0.,0.])
    #             # b = np.array([0.,0.,0.,1e-06])
    #         else:
    #             b = [[],[],[],[]]

    #             # b = [0.,0.,0.,[]]
    #             # All mechanical variables fixed
    #             # b = np.array([[[],0.,0.,0.]]); b = np.fliplr(b); b=b.reshape(4)
    #             # All electric variables fixed
    #             # b = np.array([[],[],[],0.])

    #         return b


        
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
    #                     # t = np.array([0.,0.,200000.01e0,0.1e-06])*area/no_nodes
    #                     t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
    #                 else:
    #                     t = [[],[],[],[]]

    #             # Static Analysis 
    #             if Analysis=='Static':
    #                 if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
    #                     # print node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]
    #                     # t = np.array([0.,0.,0.e0,-0.2])*area[i]/no_nodes
    #                     t = NeuArgs.Loads[i,:]*area[i]/no_nodes
    #                 else:
    #                     t = [[],[],[],[]]

    #         return t