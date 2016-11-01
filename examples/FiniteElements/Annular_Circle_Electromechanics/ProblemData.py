import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData=None):

    ndim = 2
    material = IsotropicElectroMechanics_1(ndim, mu=0.3571, lamb=1.4286, eps_1=1.0, c1=0.,c2=0.)
    ProblemPath = PWD(__file__)

    filename = ProblemPath + '/Mesh_Annular_Circle_23365.dat'
    # filename = ProblemPath + '/Mesh_Annular_Circle_5716.dat'
    # filename = ProblemPath + '/Mesh_Annular_Circle_502.dat'
    # filename = ProblemPath + '/Mesh_Annular_Circle_312.dat'
    filename = ProblemPath + '/Mesh_Annular_Circle_75.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename)
    # mesh.Square(n=1000)

    def DirichletCriteria(points):
        dirichlet_flags = np.zeros((points.shape[0],3))
        for counter, point in enumerate(points):
            if np.allclose(point[1],0.):
                dirichlet_flags[counter] = [0.0,0.0,0.0]
            elif np.allclose(point[1],1.):
                dirichlet_flags[counter] = [1.0,0.0,0.0]
            else:
                dirichlet_flags[counter] = [None,None,0.0]

        return dirichlet_flags


    boundary_condition = BoundaryCondition()
    boundary_condition.SetDirichletCriteria(DirichletCriteria, mesh.points)
    # print mesh.points
    # exit()


    formulation = DisplacementPotentialFormulation(mesh)
    fem_solver = FEMSolver(analysis_type="static", analysis_nature="nonlinear",
        number_of_load_increments=2, compute_mesh_qualities=False, parallelise=False)

    solution = fem_solver.Solve(material=material,formulation=formulation,
        mesh=mesh,boundary_condition=boundary_condition)


if __name__ == "__main__":
    ProblemData()
