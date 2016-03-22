import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData=None):

    p = 2
    ndim = 2

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.0e05,poissons_ratio=0.001)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.001)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4,
        # E_A=2.5e05,G_A=5.0e04)



    ProblemPath = os.path.dirname(os.path.realpath(__file__))

    filename = ProblemPath + '/sd7003_Stretch25.dat'
    # filename = ProblemPath + '/sd7003_Stretch50.dat'
    # filename = ProblemPath + '/sd7003_Stretch100.dat'
    # filename = ProblemPath + '/sd7003_Stretch200.dat'
    # filename = ProblemPath + '/sd7003_Stretch400.dat'
    # filename = ProblemPath + '/sd7003_Stretch800.dat'
    # filename = ProblemPath + '/sd7003_Stretch1600.dat'

    # MainData.MeshInfo.IsHighOrder = True

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="GID")
    mesh.GetHighOrderMesh(p=p)


    def ProjectionCriteria(mesh,boundary_condition):
        projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
        num = mesh.edges.shape[1]
        for iedge in range(mesh.edges.shape[0]):
            x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
            y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
            x *= boundary_condition.scale_value_on_projection
            y *= boundary_condition.scale_value_on_projection 
            # if x > -510 and x < 510 and y > -10 and y < 10:
            if x > -630 and x < 830 and y > -100 and y < 300:   
                projection_edges[iedge]=1
        
        return projection_edges

    IGES_File = ProblemPath + '/sd7003.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(IGES_File,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=True)

    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False)

    TotalDisp = fem_solver.Solve(formulation=formulation, mesh=mesh, 
        material=material, boundary_condition=boundary_condition)

if __name__ == "__main__":
    ProblemData()