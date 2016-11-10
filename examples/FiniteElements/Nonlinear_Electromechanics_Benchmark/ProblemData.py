import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero



def BenchmarkMechanics(*args, **kwargs):

    p = kwargs['p']
    ndim=2

    # material = IsotropicElectroMechanics_0(ndim,youngs_modulus=1e4,poissons_ratio=0.3, eps_1=1.0)
    material = MooneyRivlin_2(ndim,mu1=1.0,mu2=1.0,lamb=2.0)
    # material = LinearModel(ndim,youngs_modulus=10,poissons_ratio=0.3)

    mesh = Mesh()
    mesh.Square(n=4)
    mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points, tol=1e-12)

    # print mesh.points
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # exit()



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        un_edges = np.unique(mesh.edges)
        bpoints = mesh.points[un_edges]
        A = 0.1
        B = 0.2
        boundary_data[un_edges,0] = A*bpoints[:,0]**3
        boundary_data[un_edges,1] = B*bpoints[:,1]**3

        return boundary_data


    boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.Animate(configuration="deformed")
    # solution.StressRecovery()
    # solution.WriteVTK("/home/roman/zzchecker/HHH", quantity=1)
    # solution.Plot(configuration="original", quantity=2)
    # solution.Plot(configuration="deformed", quantity=1)
    solution.PlotNewtonRaphsonConvergence()



    def Analytical(mesh):
        A = 0.1
        B = 0.2
        exact_sol = np.zeros_like(mesh.points)
        exact_sol[:,0] = A*mesh.points[:,0]**3
        exact_sol[:,1] = B*mesh.points[:,1]**3
        return exact_sol


    exact_sol = Analytical(mesh)
    sol = solution.sol[:,:,-1]
    # print(sol - exact_sol)
    # error = np.abs((np.linalg.norm(sol) - np.linalg.norm(exact_sol))/np.linalg.norm(exact_sol))
    mm=1
    error = np.abs((np.linalg.norm(sol[:,mm]) - np.linalg.norm(exact_sol[:,mm]))/np.linalg.norm(exact_sol[:,mm]))
    print error, mesh.points.shape[0] 


if __name__ == "__main__":

    p = 1
    BenchmarkMechanics(p=p)
