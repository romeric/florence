import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')
from Florence import *
from Florence.VariationalPrinciple import *
from scipy.io import loadmat
from Florence.PostProcessing import PostProcess
from Florence.Tensor import makezero


def ProblemData(p=1):

    ndim = 2

    mesh = Mesh()
    mesh.Rectangle(upper_right_point=(2,12), element_type="quad", nx=2, ny=12)
    # mesh.Rectangle(upper_right_point=(2,12), element_type="tri", nx=2, ny=12)
    # mesh.Square(side_length=2, element_type="quad", n=7)
    # mesh.Square(side_length=2, n=7)

    # mesh.all_edges = None
    # mesh.GetEdgesQuad()
    # exit()
    # mesh.edges = None
    # mesh.all_edges = None
    # mesh.GetBoundaryEdgesQuad()
    # mesh.GetEdgesQuad()
    mesh.GetHighOrderMesh(p=p)

    # material = Steinmann(ndim,mu=2.3*10e+04,lamb=8.0*10.0e+04, eps_1=1505*10.0e-11, c1=0.0, c2=0.0, rho=7.5*10e-6)
    # material = NeoHookean_2(ndim, youngs_modulus=2.3*1e4, poissons_ratio=0.499999999999)
    material = NeoHookean_2(ndim, youngs_modulus=2.3*1e4, poissons_ratio=0.45)




    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        # Mechanics
        Y_0 = np.isclose(mesh.points[:,1],0.)
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        Y_1 = np.isclose(mesh.points[:,1],12.)
        boundary_data[Y_1,0] = 0.0
        # boundary_data[Y_1,1] = -4.1
        boundary_data[Y_1,1] = -2.8

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
    # exit()

    fem_solver = FEMSolver(number_of_load_increments=5,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-03)
    # fem_solver = StaggeredFEMSolver(number_of_load_increments=6,analysis_type="static",
    #     analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
    #     newton_raphson_tolerance=1.0e-02)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0], formulation.quadrature_rules[0], mesh)


    # solution.Plot(configuration="deformed", quantity=1, plot_points=True, point_radius=2)
    solution.Animate(configuration="deformed", quantity=1, plot_points=True, point_radius=2)
    # solution.Animate(configuration="deformed", quantity=1, plot_points=True, point_radius=2, save=True, 
        # filename="/home/roman/ZZZchecker/column_compression0_p"+str(p)+".gif", colorbar=False)
    # solution.WriteVTK(filename="/home/roman/ZZZchecker/QE.vtu", quantity=1)
    # solution.WriteVTK(filename="/home/roman/Dropbox/HE.vtu", quantity=10)

    # solution.CurvilinearPlot(plot_on_faces=False, QuantityToPlot=solution.sol[:,1])


if __name__ == "__main__":
    class MainData():
        C = 7

    ProblemData(p=MainData.C+1)