import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero
from Florence.PostProcessing import ErrorNorms
from scipy.io import savemat, loadmat
from Florence.Utils import RSWD
from Florence.PostProcessing import PostProcess
from math import sin, cos




def GetMeshes(p=2):

    # dd = loadmat("/home/roman/CurvedPatch_h"+str(532)+"P"+str(p)+".mat")
    # dd = loadmat("/home/roman/CurvedPatch_h"+str(26807)+"P"+str(p)+".mat")
    # dd = loadmat("/home/roman/CurvedPatch_h"+str(9220)+"P"+str(p)+".mat")
    # dd = loadmat("/home/roman/CurvedPatch_h"+str(6947)+"P"+str(p)+".mat")
    # mesh = Mesh()
    # mesh.elements = dd['elements']
    # mesh.points = dd['points']
    # mesh.faces = dd['faces']
    # mesh.element_type = "tet"
    # mesh.nelem = mesh.elements.shape[0]
    # TotalDisp = dd['TotalDisp']

    # post_process = PostProcess(3,3)
    # post_process.SetMesh(mesh)
    # post_process.SetSolution(TotalDisp)        
    # post_process.CurvilinearPlot()
    # # post_process.CurvilinearPlot(plot_edges=False)
    # exit()


    ndim=3

    material = LinearModel(ndim,youngs_modulus=10,poissons_ratio=0.45)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Patch.dat'
    # filename = ProblemPath + '/Patch_3476.dat'
    # filename = ProblemPath + '/Patch_6947.dat'
    # filename = ProblemPath + '/Patch_9220.dat'
    filename = ProblemPath + '/Patch_26807.dat'
    cad_file = ProblemPath +'/Patch.iges'


    mesh = Mesh()
    mesh.Reader(filename,"tet")
    mesh.GetHighOrderMesh(p=p)

    print 4*mesh.points.shape[0]
    exit()



    boundary_condition = BoundaryCondition()
    boundary_condition.SetCADProjectionParameters(cad_file,
        scale=1000.,condition=20000000.,project_on_curves=True,solve_for_planar_faces=True)
    # boundary_condition.SetCADProjectionParameters(cad_file,
        # scale=1000.,condition=20000000.,project_on_curves=False,solve_for_planar_faces=False)
    boundary_condition.GetProjectionCriteria(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False, compute_mesh_qualities=True)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # mesh.WriteHDF5("/home/roman/CurvedPatch_h"+str(mesh.nelem)+"P"+str(p)+".mat", {'TotalDisp':solution.sol})
    # solution.CurvilinearPlot(plot_edges=False)
    # solution.CurvilinearPlot()







##########################################################################################
##########################################################################################




def find_min_max_slope(xs, ys):
    """Finds maximum logarithmic slope of a loglog plot based on x,y arrays"""
    xs = np.array(xs)
    ys = np.array(ys)

    assert xs.shape == ys.shape

    max_slope = np.abs(np.log10(ys[0]/ys[1])/np.log10(xs[0]/xs[1]))
    min_slope = np.abs(np.log10(ys[0]/ys[1])/np.log10(xs[0]/xs[1]))

    for i in range(2,xs.shape[0]):
        slope = np.abs(np.log10(ys[0]/ys[i])/np.log10(xs[0]/xs[i]))
        if slope > max_slope:
            max_slope = slope
        if slope < min_slope:
            min_slope = slope

    return max_slope, min_slope


###

class ExactSolutions(object):

    def __init__(self):
        pass

    @staticmethod
    def Exact_Phi(points):
        phi = 10000.
        exact_sol = phi*np.sin(points[:,0])
        return exact_sol

    @staticmethod
    def Exact_E(points):

        phi = 10000.
        exact_sol = np.zeros_like(points)
        exact_sol[:,0] = phi*np.cos(points[:,0])
        # exact_sol[:,1] = 0.
        # exact_sol[:,2] = 0.

        return exact_sol

    @staticmethod
    def Exact_x(points):
        A,B,C = 0.1,0.2,0.3
        exact_sol = np.zeros_like(points)

        exact_sol[:,0] = A*np.sin(points[:,0])
        exact_sol[:,1] = B*np.cos(points[:,1])
        exact_sol[:,2] = C*(np.sin(points[:,2])+np.cos(points[:,2]))
        
        return exact_sol

    @staticmethod
    def Exact_F(points):
        F = np.zeros((points.shape[0],points.shape[1],points.shape[1]))
        A,B,C = 0.1,0.2,0.3
        # for i in range(points.shape[0]):
        #     F[i,:,:] = np.array([
        #         [1+A*np.cos(points[i,0]),        0.,                            0.                                           ],
        #         [0.,                             1-B*np.sin(points[i,1]),       0.                                           ],
        #         [0.,                             0.,                            1+C*(np.cos(points[i,2])-np.sin(points[i,2]))]
        #         ])

        F[:,0,0] = 1+A*np.cos(points[:,0])
        F[:,1,1] = 1-B*np.sin(points[:,1])
        F[:,2,2] = 1+C*(np.cos(points[:,2])-np.sin(points[:,2]))
        return F
 


##############################################################################################
##############################################################################################


def BenchmarkElectroMechanics(args):

    p = args.p
    mesh_chooser = args.mesh_chooser

    ndim=3

    # material = MooneyRivlin_2(ndim,mu1=1.0,mu2=1.0,lamb=2.0)
    material = IsotropicElectroMechanics_106(ndim, mu1=1.0, mu2=0.5, lamb=1.0, eps_1=4., eps_2=4.0)

    ProblemPath = PWD(__file__)

    if mesh_chooser==0:
            filename = ProblemPath + '/Patch.dat'
    elif mesh_chooser==1:
        filename = ProblemPath + '/Patch_3476.dat'
    elif mesh_chooser==2:
        filename = ProblemPath + '/Patch_6947.dat'
    elif mesh_chooser==3:
        filename = ProblemPath + '/Patch_9220.dat'
    else:
        filename = ProblemPath + '/Patch_26807.dat'

    # if mesh_chooser==0:
    #         filename = ProblemPath + '/Cube_12.dat'
    # elif mesh_chooser==1:
    #     filename = ProblemPath + '/Cube_208.dat'
    # elif mesh_chooser==2:
    #     filename = ProblemPath + '/Cube_1488.dat'
    # else:
    #     filename = ProblemPath + '/Cube_9335.dat'
    
    mesh = Mesh()
    mesh.Reader(filename,"tet")
    mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points, tol=1e-12)


    boundary_condition = BoundaryCondition()
    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    TotalDisp = np.zeros_like(mesh.points)
    TotalDisp = TotalDisp[:,:,None]
    # TotalDisp[:,:,-1] = ExactSol(mesh)
    solution = fem_solver.__makeoutput__(mesh, TotalDisp, 
        formulation=formulation, material=material, function_spaces=formulation.function_spaces)

    error_norms = ErrorNorms(solution,ExactSolutions)
    error_norms.SetMaterial(material)
    ex, eF, eH, eJ, ePhi, eD, ed, eSF, eSH, eSJ, eSD, eSd = error_norms.InterpolationBasedNormNonlinear(mesh,solution.sol)

    ##
    args.nelem.append(mesh.nelem)
    args.nnode.append(mesh.points.shape[0])
    args.vols.append(np.max(mesh.Volumes()))
    args.ex.append(ex)
    args.eF.append(eF)
    args.eH.append(eH)
    args.eJ.append(eJ)
    args.ePhi.append(ePhi)
    args.eD.append(eD)
    args.ed.append(ed)
    args.eSF.append(eSF)
    args.eSH.append(eSH)
    args.eSJ.append(eSJ)
    args.eSD.append(eSD)
    args.eSd.append(eSd)

    args.ndim = formulation.ndim
    args.ndim = formulation.nvar





def RunErrorNorms(p=1):

    class args(object):
        p = p
        mesh_chooser = 0
        vols = []
        ex   = []
        eF,  eH,  eJ, ePhi, eD, ed = [],[],[],[],[],[]
        eSF, eSH, eSJ, eSD, eSd = [],[],[],[],[]
        nelem = []
        nnode = []
        ndim = []
        nvar = []



    for mesh_chooser in range(5):
        args.mesh_chooser = mesh_chooser
        BenchmarkElectroMechanics(args) 


    print args.ex
    print args.eF
    print args.eH
    print args.eJ
    print args.ePhi
    print args.eD
    print args.ed
    print args.eSF
    print args.eSH
    print args.eSJ

    print 

    print find_min_max_slope(args.vols,args.ex)
    print find_min_max_slope(args.vols,args.eF)
    print find_min_max_slope(args.vols,args.eH)
    print find_min_max_slope(args.vols,args.eJ)
    print find_min_max_slope(args.vols,args.ePhi)
    print find_min_max_slope(args.vols,args.eD)
    print find_min_max_slope(args.vols,args.ed)
    print find_min_max_slope(args.vols,args.eSF)
    print find_min_max_slope(args.vols,args.eSH)
    print find_min_max_slope(args.vols,args.eSJ)
    print find_min_max_slope(args.vols,args.eSD)
    print find_min_max_slope(args.vols,args.eSd)


    sfilename = PWD(__file__) + "/Error_Norms_P" + str(p) + ".mat" 
    # print sfilename
    # savemat(sfilename,args.__dict__,do_compression=True)




################################################################################################
################################################################################################



def BenchmarkElectroMechanics_Objective(args):

    p = args.p
    mesh_chooser = args.mesh_chooser

    ndim=3

    material = IsotropicElectroMechanics_106(ndim, mu1=1.0, mu2=0.5, lamb=1.0, eps_1=4., eps_2=4.0)

    ProblemPath = PWD(__file__)

    if mesh_chooser==0:
            filename = ProblemPath + '/Patch.dat'
    elif mesh_chooser==1:
        filename = ProblemPath + '/Patch_3476.dat'
    elif mesh_chooser==2:
        filename = ProblemPath + '/Patch_6947.dat'
    elif mesh_chooser==3:
        filename = ProblemPath + '/Patch_9220.dat'
    else:
        filename = ProblemPath + '/Patch_26807.dat'

    mesh = Mesh()
    mesh.Reader(filename,"tet")
    mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points, tol=1e-12)


    boundary_condition = BoundaryCondition()
    quadrature_rule = QuadratureRule(norder=2*p, mesh_type="tet", optimal=3)
    function_space = FunctionSpace(mesh,quadrature=quadrature_rule,p=p)

    # formulation = DisplacementPotentialFormulation(mesh)
    formulation = DisplacementPotentialFormulation(mesh, quadrature_rules=(quadrature_rule,None), function_spaces=(function_space,None))

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear",parallelise=False, compute_mesh_qualities=False,
        newton_raphson_tolerance=1.0e-05)

    TotalDisp = np.zeros_like(mesh.points)
    TotalDisp = TotalDisp[:,:,None]
    # TotalDisp[:,:,-1] = ExactSol(mesh)
    # solution = fem_solver.__makeoutput__(mesh, TotalDisp, 
        # formulation=formulation, material=material, function_spaces=formulation.function_spaces)
    solution = fem_solver.__makeoutput__(mesh, TotalDisp, 
        formulation=formulation, material=material, function_spaces=(function_space,None))

    error_norms = ErrorNorms(solution,ExactSolutions)
    error_norms.SetMaterial(material)
    # print('Getting the error norms')
    ex, eC, eG, edetC, ePhi, eD0, eSC, eSG, eSdetC, eSD0 = error_norms.InterpolationBasedNormNonlinearObjective(mesh,solution.sol)
    # exit()

    ##
    args.nelem.append(mesh.nelem)
    args.nnode.append(mesh.points.shape[0])
    args.vols.append(np.max(mesh.Volumes()))
    args.ex.append(ex)
    args.eC.append(eC)
    args.eG.append(eG)
    args.edetC.append(edetC)
    args.ePhi.append(ePhi)
    args.eD0.append(eD0)
    args.eSC.append(eSC)
    args.eSG.append(eSG)
    args.eSdetC.append(eSdetC)
    args.eSD0.append(eSD0)

    args.ndim = formulation.ndim
    args.ndim = formulation.nvar





def RunErrorNorms_Objective(p=1):

    class args(object):
        p = p
        mesh_chooser = 0
        vols = []
        ex   = []
        eC,   eG,  edetC, ePhi, eD0 = [],[],[],[],[]
        eSC, eSG, eSdetC, eSD0 = [],[],[],[]
        nelem = []
        nnode = []
        ndim = []
        nvar = []


    for mesh_chooser in range(5):
        args.mesh_chooser = mesh_chooser
        BenchmarkElectroMechanics_Objective(args) 


    print args.ex
    print args.eC
    print args.eG
    print args.edetC
    print args.ePhi
    print args.eD0
    print args.eSC
    print args.eSG
    print args.eSdetC

    print 

    print 'ex', find_min_max_slope(args.vols,args.ex)
    print 'eC', find_min_max_slope(args.vols,args.eC)
    print 'eG', find_min_max_slope(args.vols,args.eG)
    print 'edetC', find_min_max_slope(args.vols,args.edetC)
    print 'ePhi', find_min_max_slope(args.vols,args.ePhi)
    print 'eD0', find_min_max_slope(args.vols,args.eD0)
    print 'eSC', find_min_max_slope(args.vols,args.eSC)
    print 'eSG', find_min_max_slope(args.vols,args.eSG)
    print 'eSdetC', find_min_max_slope(args.vols,args.eSdetC)
    print 'eSD0', find_min_max_slope(args.vols,args.eSD0)


    sfilename = PWD(__file__) + "/Error_Norms_Objective_P" + str(p) + ".mat" 
    print sfilename
    # savemat(sfilename,args.__dict__,do_compression=True)




################################################################################################
################################################################################################






def RunProblems(p=2):

    material = IsotropicElectroMechanics_105(ndim=3,mu1=10.0,mu2=10.0,lamb=20., eps_1=1.0, eps_2=1.0)

    ProblemPath = PWD(__file__)
    # filename = ProblemPath + '/Patch.dat'
    filename = ProblemPath + '/Patch_3476.dat'
    # filename = ProblemPath + '/Patch_6947.dat'
    # filename = ProblemPath + '/Patch_9220.dat'
    # filename = ProblemPath + '/Patch_26807.dat'


    mesh = Mesh()
    mesh.Reader(filename,"tet")
    # mesh.ReadHDF5("/home/roman/CurvedPatch_h"+str(6947)+"P"+str(p)+".mat")
    # mesh.GetHighOrderMesh(p=p)
    makezero(mesh.points, tol=1e-09)



    boundary_condition = BoundaryCondition()

    def DirichletFunc(mesh):

        boundary_data = np.zeros((mesh.points.shape[0],material.nvar))+np.NAN

        Y_0 = np.where(mesh.points[:,1] == 0)[0]
        boundary_data[Y_0,0] = 0.
        boundary_data[Y_0,1] = 0.
        boundary_data[Y_0,2] = 0.
        boundary_data[Y_0,3] = 0.

        Y_1 = np.where(mesh.points[:,1]==20.)[0]
        boundary_data[Y_1,0] = -30.
        # boundary_data[Y_1,1] = 0.
        boundary_data[Y_1,2] = 0.
        boundary_data[Y_1,3] = 10.

        return boundary_data


    # boundary_condition.dirichlet_flags = DirichletFunc(mesh)
    boundary_condition.SetDirichletCriteria(DirichletFunc,mesh)

    formulation = DisplacementPotentialFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="nonlinear", newton_raphson_tolerance=1.0e-02, 
        compute_mesh_qualities=False, parallelise=False)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # print solution.sol
    # solution.Animate(configuration="deformed")
    # solution.StressRecovery()
    # mesh.WriteHDF5("/home/roman/CurvedPatch_h"+str(mesh.nelem)+"P"+str(p)+".mat", {'TotalDisp':solution.sol})
    # solution.Plot(configuration="original", quantity=2)
    solution.Plot(configuration="deformed", quantity=1)
    # solution.PlotNewtonRaphsonConvergence()
    # solution.CurvilinearPlot(plot_edges=False)
    # solution.CurvilinearPlot()









if __name__ == "__main__":

    p=12
    # GetMeshes(p=p)

    # RunErrorNorms(p=p)
    RunErrorNorms_Objective(p=p)

    # RunProblems(p=p)

