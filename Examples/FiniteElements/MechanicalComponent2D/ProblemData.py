import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData=None):

    p=2
    ndim=2


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    mesh_type = "tri"
    filename = ProblemPath + '/MechanicalComponent2D_192.dat'
    # filename = ProblemPath + '/MechanicalComponent2D_664.dat'
    # filename = ProblemPath + '/MechanicalComponent2D_NonSmooth_321.dat'
    # filename = ProblemPath + '/MechanicalComponent2D_NonSmooth_2672.dat'
    # filename = ProblemPath + '/MechanicalComponent2D_NonSmooth_236.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Salome")
    mesh.points *=1000.
    mesh.GetHighOrderMesh(p=2)
    

    # material = LinearModel(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    

    # cad_file = ProblemPath + '/mechanical2d.igs'
    cad_file = ProblemPath + '/mechanical2D.iges'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1.0,condition=1.0e10)
    boundary_condition.GetProjectionCriteria(mesh)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)
    solution = fem_solver.Solve(mesh=mesh,formulation=formulation,
        material=material,boundary_condition=boundary_condition)

    solution.CurvilinearPlot(QuantityToPlot=fem_solver.ScaledJacobian)


if __name__ == "__main__":
    ProblemData()




