import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(MainData=None):

    ndim = 2
    p = 2

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    ConnectivityFile = ProblemPath + '/elements_circle.dat'
    CoordinatesFile = ProblemPath +'/points_circle.dat'
    # MatFile = ProblemPath + '/circleTest.mat'

    mesh = Mesh()
    mesh.Reader(reader_type="ReadSeparate",element_type="tri",
        connectivity_file=ConnectivityFile,coordinate_file=CoordinatesFile)
    mesh.GetHighOrderMesh(p=p)

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)



    cad_file = ProblemPath + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='equal',scale=1000.0,condition=2000.0)
    boundary_condition.GetProjectionCriteria(mesh)


    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=True)


    fem_solver.Solve(material=material,formulation=formulation,
        mesh=mesh,boundary_condition=boundary_condition)

if __name__ == "__main__":
    ProblemData()