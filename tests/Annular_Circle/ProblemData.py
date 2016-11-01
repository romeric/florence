import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver

def ProblemData(MainData):

    MainData.ndim = 2   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/Mesh_Annular_Circle_75.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename, element_type="tri", reader_type="Read")


    cad_file = ProblemPath + '/Circle.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='arc_length',
        nodal_spacing='fekete',scale=1000.0,condition=1000.0)
    boundary_condition.GetProjectionCriteria(mesh)

    return mesh, boundary_condition