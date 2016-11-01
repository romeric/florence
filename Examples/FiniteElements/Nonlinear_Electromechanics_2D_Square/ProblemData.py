import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *




def ProblemData(*args, **kwargs):

    ndim=2
    p=9

    # material = IsotropicElectroMechanics_1(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=1.0)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0,poissons_ratio=0.31)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3)
    material = NeoHookean_2(ndim,lame_parameter_1=0.4,lame_parameter_2=0.6)
    # print(material.mu, material.lamb)
    # print material.nvar


    mesh = Mesh()
    # mesh.Square(n=5)
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=4,ny=6)
    mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # exit()
    from Florence.Tensor import makezero
    makezero(mesh.points)

    # print(mesh.elements)
    # mesh.PlotMeshNumbering()
    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0
        boundary_data[Y_0,1] = 0
        # boundary_data[Y_0,2] = 0

        # Y_1 = np.where(mesh.points[:,1] == 1)[0]
        Y_1 = np.where(mesh.points[:,1] == 10)[0]
        # boundary_data[Y_1,0] = 0
        boundary_data[Y_1,1] = 20
        # boundary_data[Y_1,2] = 1

        # boundary_data[2::material.nvar,:] = 0

        # Y_0 = np.where(mesh.points[:,0] == 2)[0]
        # boundary_data[Y_0,0] = 0

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=10,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)


    # solution.StressRecovery()
    # qq = formulation.quadrature_rules[0]
    # print qq.weights, qq.points
    # from Florence.Utils import debug
    # debug(formulation.function_spaces[0], formulation.quadrature_rules[0],mesh,solution.sol)

    # print solution.sol+mesh.points
    # exit()
    solution.sol = solution.sol[:,:,None]
    # print solution.sol[:,:,-1]+mesh.points
    solution.Plot(configuration="deformed", quantity=1)
    # solution.Animate(configuration="deformed")





def ProblemData_2(*args, **kwargs):

    ndim=2
    p=2

    material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)

    mesh = Mesh()
    mesh.Rectangle(lower_left_point=(0,0),upper_right_point=(2,10),nx=4,ny=4)
    mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        # Y_1 = np.where(mesh.points[:,1] == 1)[0]
        Y_1 = np.where(mesh.points[:,1] == 10)[0]
        boundary_data[Y_1,0] = 0.1
        boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = 1.1

        # boundary_data[2::material.nvar,:] = 0
        # boundary_data[:,2] = 0
        # print boundary_data[2::material.nvar,:]
        # exit()

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    # boundary_condition.SetDirichletCriteria(DirichletCriteria, mesh.points)

    formulation = DisplacementPotentialFormulation(mesh)
    # formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-04)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.Plot(configuration="deformed")
    # solution.Animate(configuration="deformed", quantity=2)

    # exit()



def ProblemData_3D(*args, **kwargs):

    ndim=3
    p=1

    # material = IsotropicElectroMechanics_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3, c1=1.0, c2=1.0)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0,poissons_ratio=0.41)
    material = NeoHookean_2(ndim,youngs_modulus=1.0,poissons_ratio=0.3)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Mesh_Holes.dat'
    filename = ProblemPath + '/Mesh_Cyl.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tet")
    mesh.GetHighOrderMesh(p=p)
    from Florence.Tensor import makezero
    # mesh.SimplePlot()
    # print mesh.Bounds
    # exit()

    boundary_condition = BoundaryCondition()

    # def DirichletFunc(mesh):

    #     boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

    #     Y_0 = np.where(mesh.points[:,0] == 0)[0]
    #     boundary_data[Y_0,0] = -20.
    #     boundary_data[Y_0,1] = 0.
    #     boundary_data[Y_0,2] = 0.

    #     Y_1 = np.where(mesh.points[:,0] == 100)[0]
    #     boundary_data[Y_1,0] = 20.
    #     boundary_data[Y_1,1] = 0.
    #     boundary_data[Y_1,2] = 0.

    #     # boundary_data[:,3] = 0

    #     return boundary_data



    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,2] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.

        Y_1 = np.where(mesh.points[:,2] == 100)[0]
        boundary_data[Y_1,0] = 0.
        boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = 10.

        # boundary_data[:,3] = 0

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-04)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.Plot(configuration="deformed")
    # solution.CurvilinearPlot(QuantityToPlot=solution.sol[:,0])
    # solution.Animate(configuration="deformed")
    solution.StressRecovery()






if __name__ == "__main__":
    ProblemData()
    # ProblemData_2()
    # ProblemData_3D()
    
    # from cProfile import run
    # run('ProblemData_3D()')
