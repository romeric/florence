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
    filename = ProblemPath + '/Patch_6947.dat'
    # filename = ProblemPath + '/Patch_9220.dat'
    # filename = ProblemPath + '/Patch_26807.dat'
    cad_file = ProblemPath +'/Patch.iges'


    mesh = Mesh()
    mesh.Reader(filename,"tet")
    mesh.GetHighOrderMesh(p=p)
    # makezero(mesh.points, tol=1e-12)

    # print mesh.points
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # exit()



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

    print solution.sol
    # solution.Animate(configuration="deformed")
    # solution.StressRecovery()
    mesh.WriteHDF5("/home/roman/CurvedPatch_h"+str(mesh.nelem)+"P"+str(p)+".mat", {'TotalDisp':solution.sol})
    # solution.Plot(configuration="original", quantity=2)
    # solution.Plot(configuration="deformed", quantity=1)
    # solution.PlotNewtonRaphsonConvergence()
    # solution.CurvilinearPlot(plot_edges=False)
    # solution.CurvilinearPlot()







##########################################################################################
##########################################################################################




def FindMaxSlop(xs, ys):
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
        exact_sol[:,1] = 0.
        exact_sol[:,2] = 0.

        return exact_sol

    @staticmethod
    def Exact_E(points):

        phi = 10000.
        exact_sol = np.zeros_like(points)
        exact_sol[:,0] = phi*np.cos(points[:,0])
        exact_sol[:,1] = 0.
        exact_sol[:,2] = 0.

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
        for i in range(points.shape[0]):
            F[i,:,:] = np.array([
                [1+A*np.cos(points[i,0]),        0.,                            0.                                           ],
                [0.,                             1-B*np.sin(points[i,1]),       0.                                           ],
                [0.,                             0.,                            1+C*(np.cos(points[i,2])-np.sin(points[i,2]))]
                ])
        return F




    # @staticmethod
    # def Exact_x(points):
    #     A = 0.1
    #     B = 0.2
    #     C = 0.3
    #     exact_sol = np.zeros_like(points)

    #     exact_sol[:,0] = A*points[:,0]**3
    #     exact_sol[:,1] = B*points[:,1]**3
    #     exact_sol[:,2] = C*points[:,2]**3
        
    #     return exact_sol

    # @staticmethod
    # def Exact_F(points):
    #     F = np.zeros((points.shape[0],points.shape[1],points.shape[1]))
    #     A = 0.1; B=0.2; C=0.3
    #     for i in range(points.shape[0]):
    #         F[i,:,:] = np.array([
    #             [1+3*A*points[i,0]**2,           0.,                            0.                        ],
    #             [0.,                             1+3*B*points[i,1]**2,          0.                        ],
    #             [0.,                             0.,                            1+3*C*points[i,2]**2      ]
    #             ])
    #     return F
 



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

    # print mesh.points
    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()
    # exit()



    boundary_condition = BoundaryCondition()
    # formulation = DisplacementFormulation(mesh)
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



    # BenchmarkMechanics(args)
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

    print FindMaxSlop(args.vols,args.ex)
    print FindMaxSlop(args.vols,args.eF)
    print FindMaxSlop(args.vols,args.eH)
    print FindMaxSlop(args.vols,args.eJ)
    print FindMaxSlop(args.vols,args.ePhi)
    print FindMaxSlop(args.vols,args.eD)
    print FindMaxSlop(args.vols,args.ed)
    print FindMaxSlop(args.vols,args.eSF)
    print FindMaxSlop(args.vols,args.eSH)
    print FindMaxSlop(args.vols,args.eSJ)
    print FindMaxSlop(args.vols,args.eSD)
    print FindMaxSlop(args.vols,args.eSd)


    sfilename = PWD(__file__) + "/Error_Norms_P" + str(p) + ".mat" 
    # print sfilename
    # savemat(sfilename,args.__dict__,do_compression=True)





if __name__ == "__main__":

    p=2
    # GetMeshes(p=p)

    RunErrorNorms(p=p)

