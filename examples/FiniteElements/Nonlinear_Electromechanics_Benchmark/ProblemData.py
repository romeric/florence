import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')

import numpy as np 
from Florence import *
from Florence.VariationalPrinciple import *
from Florence.Tensor import makezero
from Florence.PostProcessing import ErrorNorms
from scipy.io import savemat, loadmat
from Florence.Utils import RSWD


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



def GetMeshes(p=2):

    p = kwargs['p']
    ndim=2

    material = LinearModel(ndim,youngs_modulus=10,poissons_ratio=0.45)

    ProblemPath = PWD(__file__)
    filename = ProblemPath + '/Patch_3476.dat'
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
        scale=1,condition=2,project_on_curves=True,solve_for_planar_faces=True)
    boundary_condition.GetProjectionCriteria(mesh)
    formulation = DisplacementFormulation(mesh)

    fem_solver = FEMSolver(number_of_load_increments=1,analysis_type="static",
        analysis_nature="linear",parallelise=False, compute_mesh_qualities=True)

    solution = fem_solver.Solve(formulation=formulation, mesh=mesh, 
            material=material, boundary_condition=boundary_condition)

    # solution.Animate(configuration="deformed")
    # solution.StressRecovery()
    # solution.WriteVTK("/home/roman/zzchecker/HHH", quantity=1)
    # solution.Plot(configuration="original", quantity=2)
    # solution.Plot(configuration="deformed", quantity=1)
    # solution.PlotNewtonRaphsonConvergence()
    # solution.CurvilinearPlot(plot_edges=False)
    solution.CurvilinearPlot()

 



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
    for mesh_chooser in range(2):
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


    sfilename = RSWD() + "/Error_Norms_P" + str(p) + ".mat" 
    # savemat(sfilename,{'ex':np.array(args.ex),'eF':np.array(eF),'eH':np.array(eH),'eJ':np.array(eJ),
    #     'ePhi':np.array(ePhi),'eD':np.array(eD),'ed':np.array(ed),'eSF':np.array(eSF),'eSH':np.array(eSH),
    #     'eSJ':np.array(eSJ),'eSD':np.array(eSD),'eSd':np.array(eSd)})
    # print sfilename
    savemat(sfilename,args.__dict__,do_compression=True)





if __name__ == "__main__":

    p=2
    # GetMeshes(p=p)

    RunErrorNorms(p=p)




[0.0030902569872664713, 1.7631262616368935e-07, 9.9096676004553359e-10, 5.8928001482098283e-10, 1.1685562734356378e-11]
[0.0046934423917277838, 6.9761765277152682e-07, 1.1170075755259095e-08, 6.9031128353034029e-09, 3.0105794322159202e-10]
[0.0057815240422056536, 1.1631485227676551e-06, 2.0807302537257288e-08, 1.3068985241767194e-08, 5.1541253068953567e-10]
[0.0032725178306228842, 1.4248596249863167e-06, 2.8871799705000912e-08, 1.8385633191743525e-08, 6.2742787196987086e-10]
[0.00023478816267113735, 5.9119854771377098e-08, 2.9649144644834275e-10, 1.8706502933339191e-10, 3.3990218399843752e-12]
[0.00072103875385531619, 5.6842918268376727e-08, 3.4523912096474199e-10, 1.7879285523462958e-10, 3.4731864376694646e-12]
[0.0056372202742033615, 3.3612032167788282e-07, 2.9389188796877811e-09, 1.2439636961350113e-09, 1.6779618689421816e-10]
[0.0046934423917277838, 6.9761765277152682e-07, 1.1170075755259095e-08, 6.9031128353034029e-09, 3.0105794322159202e-10]
[0.0057815240422056536, 1.1631485227676551e-06, 2.0807302537257288e-08, 1.3068985241767194e-08, 5.1541253068953567e-10]
[0.014723510091821775, 1.357442540632285e-05, 3.7570978312009951e-07, 2.1574476506841903e-07, 5.5616445479189257e-09]

(5.2528263825201433, 4.8058233160989134)
(4.7381070076770548, 4.1712904711231111)
(4.5753785371930817, 4.0378036245390634)
(4.1603507431793556, 3.7550207661753241)
(4.6792608245830989, 4.3616997883420936)
(5.0790047399041605, 4.7242459029501029)
(5.2291338942983723, 4.4924088746294313)
(4.7381070076770548, 4.1712904711231111)
(4.5753785371930817, 4.0378036245390634)
(3.833749083745321, 3.4572652513510476)
(5.0790047399041605, 4.7242459029501029)
(5.2291338942983723, 4.4924088746294313)
