import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver
from Florence.MaterialLibrary import *


def ProblemData(MainData):

    MainData.ndim = 2    
    MainData.Fields = 'Mechanics'    
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    # MainData.AnalysisType = 'Linear'
    MainData.AnalysisType = 'Nonlinear'


    # material = LinearModel(MainData.ndim,youngs_modulus=1.0e01,poissons_ratio=0.3)
    # material = IncrementalLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = NeoHookean_2(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3)
    material = NeoHookean_2(MainData.ndim,youngs_modulus=1.0e05,poissons_ratio=0.4)
    # material = MooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = NearlyIncompressibleMooneyRivlin(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3)
    # material = BonetTranservselyIsotropicHyperElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3,
        # E_A=2.5,G_A=0.5)
    # material = TranservselyIsotropicLinearElastic(MainData.ndim,youngs_modulus=1.,poissons_ratio=0.3,
        # E_A=2.5,G_A=0.5)


    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/TwoArcs_18.dat'
    # FileName = ProblemPath + '/Leaf_2.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tri",reader_type="Read")

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

    solver = LinearSolver(linear_solver="direct", linear_solver_type="umfpack")
    MainData.solver = solver

    return mesh, material, boundary_condition
