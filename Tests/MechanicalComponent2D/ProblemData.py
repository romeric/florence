import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver

def ProblemData(MainData):

    MainData.ndim = 2   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'
    # MainData.AnalysisType = 'Nonlinear'

    # MainData.MaterialArgs.E  = 1.0e5
    # MainData.MaterialArgs.nu = 0.4
    # MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E
    # MainData.MaterialArgs.G_A = MainData.MaterialArgs.E/2.

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/MechanicalComponent2D_192.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Read")

    cad_file = ProblemPath + '/mechanical2D.iges'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1.0,condition=1.0e10)
    boundary_condition.GetProjectionCriteria(mesh)

    solver = LinearSolver()
    MainData.solver = solver

    return mesh, material, boundary_condition

