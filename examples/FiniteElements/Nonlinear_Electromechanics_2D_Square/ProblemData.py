import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    ndim=2
    p=2

    material = IsotropicElectroMechanics_1(ndim,youngs_modulus=1.0,poissons_ratio=0.3, eps_1=1.0)
    # print material.nvar


    mesh = Mesh()
    mesh.Square()
    # mesh.GetHighOrderMesh(p=p)
    # print mesh.points.shape[0], mesh.nelem
    # mesh.SimplePlot()

    # exit()

    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):
        # boundary_data = np.zeros(material.nvar*mesh.points.shape[0])
        # boundary_data = np.array([None]*material.nvar*mesh.points.shape[0])
        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0
        boundary_data[Y_0,1] = 0
        boundary_data[Y_0,2] = 0

        Y_1 = np.where(mesh.points[:,1] == 1)[0]
        boundary_data[Y_1,0] = 1
        boundary_data[Y_1,1] = 1
        boundary_data[Y_1,2] = 1

        return boundary_data

    boundary_condition.dirichlet_flags = DirichletFunc(mesh)

    # print boundary_condition.dirichlet_flags 
    # print dir(boundary_condition)
    # exit()

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)

    formulation = DisplacementElectricPotentialFormulation(mesh)
    # function_space = FunctionSpace(mesh, quadrature, p=C+1)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    exit()

if __name__ == "__main__":
    ProblemData()
