import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *
from Florence.VariationalPrinciple import *


def ProblemData(MainData):

    MainData.ndim = 2
    MainData.Fields = 'Mechanics'
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'

    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.4)
    material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.4,
        # E_A=2.5,G_A=0.5)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))

    # MainData.MeshInfo.MeshType = 'tri'
    # MainData.MeshInfo.Nature = 'straight'
    # # MainData.MeshInfo.Reader = 'ReadSeparate'
    # MainData.MeshInfo.Reader = 'ReadHDF5'

    ConnectivityFile = ProblemPath + '/elements_rae.dat'
    CoordinatesFile = ProblemPath +'/points_rae.dat'
    # MainData.MeshInfo.EdgesFile = ProblemPath + '/edges_rae.dat'
    # # FileName = ProblemPath + '/RAE2822.dat'
    # # FileName = ProblemPath +'/RAE2822_Isotropic_90.dat'
    # # FileName = ProblemPath +'/RAE2822_Isotropic_414.dat'
        
    # filename = ProblemPath + '/RAE2822_P'+str(MainData.C+1)+'.mat'
    # MainData.MeshInfo.IsHighOrder = True

    mesh = Mesh()
    mesh.Reader(reader_type="ReadSeparate",element_type="tri",
        connectivity_file=ConnectivityFile,coordinate_file=CoordinatesFile)



    def NURBSParameterisation():
        control = np.loadtxt(ProblemPath+'/controls_rae.dat',delimiter=',')
        knots = np.loadtxt(ProblemPath+'/knots_rae.dat',delimiter=',')
        return [({'U':(knots,),'Pw':control,'start':0,'end':2.04,'degree':3})]

    def NURBSCondition(x):
        return np.sqrt(x[:,:,0]**2 + x[:,:,1]**2) < 2

    cad_file = ProblemPath + '/rae2822.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,requires_cad=False,projection_type='arc_length',
        nodal_spacing='equal',scale=1000.0,condition=500.0)
    boundary_condition.SetNURBSParameterisation(NURBSParameterisation)
    boundary_condition.SetNURBSCondition(NURBSCondition,mesh.points[mesh.edges[:,:2],:])

    solver = LinearSolver(linear_solver="multigrid", linear_solver_type="amg",iterative_solver_tolerance=5.0e-07)


    # class BoundaryData(object):
    #     # NURBS/NON-NURBS TYPE BOUNDARY CONDITION
    #     Type = 'nurbs'
    #     RequiresCAD = True
    #     ProjectionType = 'orthogonal'
    #     # CurvilinearMeshNodalSpacing = 'fekete'
    #     IGES_File = ProblemPath + '/rae2822.igs'

    #     # aniso
    #     # condition = 5.
    #     condition = 0.5*1000.
    #     scale = 1000.

    #     # iso
    #     # condition = 0.6
    #     # scale = 1.


        # def NURBSParameterisation(self):
        #     control = np.loadtxt(ProblemPath+'/controls_rae.dat',delimiter=',')
        #     knots = np.loadtxt(ProblemPath+'/knots_rae.dat',delimiter=',')
        #     return [({'U':(knots,),'Pw':control,'start':0,'end':2.04,'degree':3})]

        # def NURBSCondition(self,x):
        #     return np.sqrt(x[:,0]**2 + x[:,1]**2) < 2


    formulation = DisplacementFormulation(mesh)
    fem_solver = FEMSolver(number_of_load_increments=2,analysis_type="static",
        analysis_nature="linear",parallel=True)



    return formulation, mesh, material, boundary_condition, solver, fem_solver
