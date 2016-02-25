import numpy as np 
import os, imp
from Florence import Mesh, BoundaryCondition, LinearSolver, FEMSolver


def ProblemData(MainData):

    MainData.ndim = 3   
    MainData.Fields = 'Mechanics'   
    MainData.Formulation = 'DisplacementApproach'
    MainData.Analysis = 'Static'
    MainData.AnalysisType = 'Linear'

    # MainData.MaterialArgs.E  = 1.0e5
    # MainData.MaterialArgs.nu = 0.485


    # E = MainData.MaterialArgs.E
    # nu = MainData.MaterialArgs.nu
    # print 'Poisson ratio is:', MainData.MaterialArgs.nu

    # MainData.MaterialArgs.E_A = 2.5*MainData.MaterialArgs.E
    # E_A = MainData.MaterialArgs.E_A
    # MainData.MaterialArgs.G_A = (E*(E_A*nu - E_A + E_A*nu**2 + E*nu**2))/(2*(nu + 1)*(2*E*nu**2 + E_A*nu - E_A))

    # # GET LAME CONSTANTS
    # MainData.MaterialArgs.lamb = E*nu/(1.+nu)/(1.-2.0*nu)
    # MainData.MaterialArgs.mu = E/2./(1+nu)

    ProblemPath = os.path.dirname(os.path.realpath(__file__))
    filename = ProblemPath + '/almond_H1.dat'

    mesh = Mesh()
    mesh.Reader(filename=filename,element_type="tet",reader_type_format="GID")


    def ProjectionCriteria(mesh,boundary_condition):
        projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
        num = mesh.faces.shape[1]
        scale = boundary_condition.scale_value_on_projection
        for iface in range(mesh.faces.shape[0]):
            x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
            y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
            z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
            x *= scale
            y *= scale
            z *= scale 
            if x > -2.5*scale and x < 2.5*scale and y > -2.*scale \
                and y < 2.*scale and z > -2.*scale and z < 2.*scale:    
                projection_faces[iface]=1
        
        return projection_faces

    cad_file = ProblemPath + '/almond.igs'
    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,projection_type='orthogonal',
        scale=25.4,project_on_curves=False,solve_for_planar_faces=False,modify_linear_mesh_on_projection=False)
    boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh,takes_self=True)

    solver = LinearSolver(iterative_solver_tolerance=5.0e-07)
    MainData.solver = solver

    return mesh, boundary_condition
        