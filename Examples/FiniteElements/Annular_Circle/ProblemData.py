import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(*args, **kwargs):

    # CREATE A MESH
    mesh = Mesh()
    # filename = PWD(__file__) + '/Mesh_Annular_Circle_23365.dat'
    # filename = PWD(__file__) + '/Mesh_Annular_Circle_5716.dat'
    # filename = PWD(__file__) + '/Mesh_Annular_Circle_502.dat'
    # filename = PWD(__file__) + '/Mesh_Annular_Circle_312.dat'
    filename = PWD(__file__) + '/Mesh_Annular_Circle_75.dat'

    mesh.Reader(filename=filename, element_type="tri", reader_type="Salome")
    mesh.GetHighOrderMesh(p=2)

    # CHOOSE A MATERIAL MODEL
    ndim = mesh.InferSpatialDimension()

    # material = Material("LinearModel",ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = LinearModel(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)


    

    # SET-UP BOUNDARY CONDITION
    cad_file = PWD(__file__) + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=1000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    # SET UP FORMULATION
    formulation = DisplacementFormulation(mesh)

    # CHOOSE LINEAR SOLVER
    # solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")

    # SET UP FINITE ELEMENT SOLVER
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    solution = fem_solver.Solve(formulation, mesh, material, boundary_condition)

    solution.CurvilinearPlot(QuantityToPlot=fem_solver.ScaledJacobian)


if __name__ == "__main__":
    from time import time
    t = time()
    ProblemData()
    print time() - t

