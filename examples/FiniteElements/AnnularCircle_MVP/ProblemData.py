import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData):

    ndim = 2
    p = 2   

    # material = Material("LinearModel",MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = NearlyIncompressibleNeoHookean(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4999)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.49)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
    #     E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)


    ProblemPath = PWD(__file__)

    # filename = ProblemPath + '/Mesh_Annular_Circle_502.dat'
    # filename = ProblemPath + '/Mesh_Annular_Circle_312.dat'
    filename = ProblemPath + '/Mesh_Annular_Circle_75.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Salome")
    mesh.GetHighOrderMesh(p=p)
    # print mesh.Areas


    cad_file = ProblemPath + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=1000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    # formulation = DisplacementFormulation(mesh)
    formulation = NearlyIncompressibleHuWashizu(mesh)
    fem_solver = FEMSolver(analysis_type="static",analysis_nature="nonlinear",
        number_of_load_increments=1,parallelise=False)

    solution = fem_solver.Solve(material=material,formulation=formulation,
        mesh=mesh,boundary_condition=boundary_condition)

    # solution.CurvilinearPlot(QuantityToPlot=fem_solver.ScaledJacobian)

if __name__ == "__main__":
    ProblemData()