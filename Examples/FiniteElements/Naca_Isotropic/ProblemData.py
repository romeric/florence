import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

from Florence import *
from Florence.VariationalPrinciple import *

def ProblemData(*args, **kwargs):

    ndim = 2
    p = 2

    # material = LinearModel(ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = IncrementalLinearElastic(ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NeoHookean_2(ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = MooneyRivlin(ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)

    ProblemPath = PWD(__file__)

    Reader = 'ReadSeparate'
    ConnectivityFile = ProblemPath + '/elements_naca.dat'
    CoordinatesFile = ProblemPath +'/points_naca.dat'

    mesh = Mesh()
    mesh.Reader(reader_type="ReadSeparate",element_type="tri",
        connectivity_file=ConnectivityFile,coordinate_file=CoordinatesFile)
    mesh.GetHighOrderMesh(p=p)


    def NURBSParameterisation():
        control = np.loadtxt(ProblemPath+'/controls_naca.dat',delimiter=',')
        knots = np.loadtxt(ProblemPath+'/knots_naca.dat',delimiter=',')
        return [({'U':(knots,),'Pw':control,'start':0,'end':2.039675505705710,'degree':3})]

    def NURBSCondition(x):
        return np.sqrt(x[:,:,0]**2 + x[:,:,1]**2) < 2

    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(requires_cad=False,scale=1.0,condition=2.0)
    boundary_condition.SetNURBSParameterisation(NURBSParameterisation)
    boundary_condition.SetNURBSCondition(NURBSCondition,mesh.points[mesh.edges[:,:2],:])
        
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear",parallelise=True)

    fem_solver.Solve(material=material,formulation=formulation,
        mesh=mesh,boundary_condition=boundary_condition)

if __name__ == "__main__":
    ProblemData()