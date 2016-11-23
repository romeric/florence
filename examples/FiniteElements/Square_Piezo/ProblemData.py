import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from scipy.io import loadmat
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def ProblemData(p=1):

    # p=2

    # from Florence.FunctionSpace.GetBases import GetBases, GetBases3D, GetBasesBoundary, GetBasesAtNodes
    # from Florence import QuadratureRule

    # quadrature = QuadratureRule(norder=p+1, mesh_type="quad")
    # Domain = GetBases(p-1,quadrature,"quad")
    # exit()

    ndim = 2

    mesh = Mesh()
    mesh.Rectangle(upper_right_point=(2,10), element_type="quad", nx=4, ny=5)
    # mesh.Square(side_length=2, element_type="quad", n=7)
    # mesh.Square(side_length=2, n=7)

    # print mesh.Areas()
    # elements = np.copy(mesh.elements)
    # elements[:,0] = mesh.elements[:,3]
    # elements[:,1] = mesh.elements[:,2]
    # elements[:,2] = mesh.elements[:,1]
    # elements[:,3] = mesh.elements[:,0]
    # mesh.elements = elements
    # mesh.CheckNodeNumbering()
    # print mesh.AspectRatios()
    # print 3333333333333
    # mesh.GetEdgesQuad()
    # exit()

    # material = Steinmann(ndim,mu=2.3*10e+04,lamb=8.0*10.0e+04, eps_1=1505*10.0e-11, c1=0.0, c2=0.0, rho=7.5*10e-6)
    # material = NeoHookean_2(ndim, youngs_modulus=2.3*1e4, poissons_ratio=0.499999999999)
    material = NeoHookean_2(ndim, youngs_modulus=2.3*1e4, poissons_ratio=0.4)

    # ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_Square_9.dat'                   

    # mesh = Mesh()
    # mesh.Reader(filename, "quad")
    # mesh.GetHighOrderMesh(p=p)
    # print mesh.Bounds
    # print mesh.elements.shape
    # mesh.SimplePlot()
    # print mesh.elements
    # print mesh.edges
    # mesh.PlotMeshNumbering()





    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        Y_1 = np.isclose(mesh.points[:,1],10.)
        boundary_data[Y_1,0] = 0.0
        boundary_data[Y_1,1] = 20.

        # Electromechanics
        # Y_0 = np.isclose(mesh.points[:,1],0.)
        # boundary_data[Y_0,0] = 0.
        # boundary_data[Y_0,1] = 0.
        # boundary_data[Y_0,2] = 5.
        # Y_1 = np.isclose(mesh.points[:,1],2.)
        # boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = 1.0
        # boundary_data[Y_1,2] = -5.0

        return boundary_data

    boundary_condition.SetDirichletCriteria(DirichletFunc, mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=5,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-06)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=6,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0], formulation.quadrature_rules[0], mesh)


    solution.Plot(configuration="deformed", quantity=1, plot_points=True, point_radius=2)
    # solution.WriteVTK(filename="/home/roman/ZZZchecker/QE.vtu", quantity=1)
    # solution.WriteVTK(filename="/home/roman/Dropbox/HE.vtu", quantity=10)

    # elements = np.concatenate((mesh.elements[:,:3],mesh.elements[:,[0,1,3]],
    #         mesh.elements[:,[0,2,3]],mesh.elements[:,[1,2,3]]),axis=0)
    # tmesh = Mesh()
    # tmesh.elements = elements
    # tmesh.element_type = "tri"
    # tmesh.points = mesh.points
    # tmesh.nelem = tmesh.elements.shape[0]
    # tmesh.edges = tmesh.GetBoundaryEdgesTri()
    # from Florence.PostProcessing import PostProcess
    # # tmesh.GetHighOrderMesh(p=3)
    # # post_process = PostProcess(2,2)
    # # post_process
    # # PostProcess.CurvilinearPlotTri(tmesh,np.zeros_like(tmesh.points))
    # PostProcess.CurvilinearPlotTri(tmesh,solution.sol[:,:2,-1])
    # exit()


if __name__ == "__main__":
    class MainData():
        C = 3

    ProblemData(p=MainData.C+1)