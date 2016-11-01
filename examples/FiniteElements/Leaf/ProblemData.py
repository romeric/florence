import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(*args, **kwargs):

    ndim = 2
    p = 2

    # material = LinearModel(ndim,youngs_modulus=1.0e01,poissons_ratio=0.3)
    # material = IncrementalLinearElastic(ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = NeoHookean_2(ndim,youngs_modulus=1.,poissons_ratio=0.3)
    material = NeoHookean_2(ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.,poissons_ratio=0.3,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.,poissons_ratio=0.3,
        # E_A=2.5,G_A=0.5)


    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/TwoArcs_18.dat'
    # FileName = ProblemPath + '/Leaf_2.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tri",reader_type="Salome")
    mesh.GetHighOrderMesh(p=p)

    def ProjectionCriteria(mesh,boundary_condition):
        projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
        num = mesh.edges.shape[1]
        for iedge in range(mesh.edges.shape[0]):
            x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
            y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
            x *= boundary_condition.scale_value_on_projection
            y *= boundary_condition.scale_value_on_projection
            if np.sqrt(x*x+y*y)< boundary_condition.condition_for_projection:
                projection_edges[iedge,0]=1
        
        return projection_edges


    cad_file = ProblemPath + '/Two_Arcs.iges'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=3000.0)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=True)

    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=10,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False)

    fem_solver.Solve(material=material,formulation=formulation,
        mesh=mesh,boundary_condition=boundary_condition)

if __name__ == "__main__":
    from time import time
    t=time()
    ProblemData()
    print time() - t