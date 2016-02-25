from __future__ import print_function
import sys, os, imp, time, gc
from sys import exit
from datetime import datetime
from warnings import warn
import numpy as np
import scipy as sp
import multiprocessing as MP
from scipy.io import loadmat, savemat

# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode
# SET NUMPY'S LINEWIDTH PRINT OPTION
np.set_printoptions(linewidth=300)

# IMPORT NECESSARY CLASSES FROM BASE
from Florence import Base as MainData
from Florence.FiniteElements.PreProcess import PreProcess
from Florence.FiniteElements.PostProcess import *
from Florence.FiniteElements.Solvers.Solver import *
import Florence.MaterialLibrary
from Florence import Mesh, BoundaryCondition



tick = u'\u2713'.encode('utf8')+' : ' 
cross = u'\u2717'.encode('utf8')+' : '


class run_tests(object):
    def __init__(self):
        pass


def entity_checker(x,y,tol=1e-10):
    """x,y both being ndarrays, x being new solution and y being pre-computed solution.
        tol parameter is a very important aspect of the comparison"""

    if x.shape != y.shape:
        raise TypeError("shape of entity does not match with pre-computed solution")

    # safe-guard against unsigned overflow
    if x.dtype != y.dtype:
        if x.dtype != float:
            if x.dtype == np.uint64 or x.dtype == np.uint32:
                x = x.astype(np.int64)
        if y.dtype != float:
            if y.dtype == np.uint64 or y.dtype == np.uint32:
                y = y.astype(np.int64)

    if np.isclose(x-y,0.,atol=tol).all():
        return True
    elif np.isclose(np.sum(x-y),0.,atol=tol):
        return True
    else:
        return False

def mesh_checker(mesh,Dict):
    """Give a mesh and a Dict loaded from HDF5 to compare"""

    print("Checking higher order mesh generators results")
    if entity_checker(mesh.elements,Dict['elements']):
        print(tick, "mesh elements match")
    else:
        print(cross, "mesh elements do not match")

    if entity_checker(mesh.points,Dict['points']):
        print(tick, "mesh points match")
    else:
        print(cross, "mesh points do not match")

    if entity_checker(mesh.edges,Dict['edges']):
        print(tick, "mesh edges match")
    else:
        print(cross, "mesh edges do not match")

    if mesh.element_type == "tet" or mesh.element_type == "hex":
        if entity_checker(mesh.faces,Dict['faces']):
            print(tick, "mesh faces match")
        else:
            print(cross, "mesh faces do not match")


def dirichlet_checker(ColumnsOut,AppliedDirichlet,Dict):

    if ColumnsOut.ndim > 1:
        ColumnsOut = ColumnsOut.flatten()
    if Dict['ColumnsOut'].ndim > 1:
        Dict['ColumnsOut'] = Dict['ColumnsOut'].flatten()

    if AppliedDirichlet.ndim > 1:
        AppliedDirichlet = AppliedDirichlet.flatten()
    if Dict['AppliedDirichlet'].ndim > 1:
        Dict['AppliedDirichlet'] = Dict['AppliedDirichlet'].flatten()

    print("Checking for projection data from OpenCascade wrapper")
    if entity_checker(ColumnsOut,Dict['ColumnsOut']):
        print(tick, "Dirichlet degrees of freedom match")
    else:
        print(cross, "Dirichlet degrees of freedom do not match")

    if entity_checker(AppliedDirichlet,Dict['AppliedDirichlet']):
        print(tick, "Dirichlet data for degrees of freedom match")
    else:
        print(cross, "Dirichlet data for degrees of freedom do not match")

def final_solution_checker(MainData,material,TotalDisp,Dict):

    print("Checking for final solution")
    if material.nu != Dict['PoissonRatio']:
        raise ValueError("Analysis with different material parameters are being compared")
    if material.E != Dict['YoungsModulus']:
        raise ValueError("Analysis with different material parameters are being compared")
    if material.is_transversely_isotropic:
        if material.E_A != Dict['E_A']:
            raise ValueError("Analysis with different material parameters are being compared")
        if material.G_A != Dict['G_A']:
            raise ValueError("Analysis with different material parameters are being compared")

    if MainData.solver.solver_type != Dict['SolverType']:
        raise ValueError("Results from different solvers are being compared")
    if MainData.solver.iterative_solver_tolerance != Dict['SolverTol']:
        raise ValueError("Solver results with different tolerances are being compared")


    tol = 1e-9
    print(np.linalg.norm(TotalDisp - Dict['TotalDisp']))
    if entity_checker(TotalDisp,Dict['TotalDisp'],tol):
        print(tick, "Final solution is correct")
    else:
        print(cross, "Final solution does not match with pre-computed solution")
        exit()

    Dict['ScaledJacobian'] = Dict['ScaledJacobian'].flatten()
    MainData.ScaledJacobian = MainData.ScaledJacobian.flatten()
    if entity_checker(MainData.ScaledJacobian,Dict['ScaledJacobian'],1e-8):
        print(tick,"Final mesh quality is correct")
    else:
        print(cross,"Final mesh quality does not match")
        exit()




def AlmondTestCases():

    print("\n=========================================================================")
    print("                       RUNNING FLORENCE TEST-SUITE                         ")
    print("=========================================================================\n")
    print("                RUNNING ALMOND 3D TETRAHEDRAL TEST CASES                   ")

    MainData.__NO_DEBUG__ = True
    MainData.__VECTORISATION__ = True
    MainData.__PARALLEL__ = True
    MainData.numCPU = MP.cpu_count()
    MainData.__PARALLEL__ = False
    MainData.__MEMORY__ = 'SHARED'
    # MainData.__MEMORY__ = 'DISTRIBUTED'
    
    MainData.C = 1
    MainData.norder = 2
    MainData.plot = (0, 3)
    nrplot = (0, 'last')
    MainData.write = 0

    import Tests.Almond3D.ProblemData as Pr
    from Florence import BoundaryCondition

    # GET THE CURRENT DIRECTORY PARTH
    pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
    # material_models = ["IncrementalLinearElastic","TranservselyIsotropicLinearElastic",
                        # "NeoHookean_2","MooneyRivlin","NearlyIncompressibleMooneyRivlin",
                        # "BonetTranservselyIsotropicHyperElastic"]

    material_models = ["IncrementalLinearElastic"]

    for p in [2,3,4,5,6]:
        MainData.C = p - 1

        # MainData.MaterialArgs.Type = "IncrementalLinearElastic"
        # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
        # MainData.MaterialArgs.Type = "NeoHookean_2"
        # MainData.MaterialArgs.Type = 'MooneyRivlin'
        # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
        # MainData.MaterialArgs.Type = 'BonetTranservselyIsotropicHyperElastic'

        # for Increment in [1,10]:
        for Increment in [1]:    
            MainData.LoadIncrement = Increment

            # if p > 3 and Increment == 10:
                # continue

            for material_model in material_models:

                MainData.MaterialArgs.Type = material_model

                # READ PROBLEM DATA FILE
                Pr.ProblemData(MainData)
                
                # PRE-PROCESS
                print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...')

                mesh = PreProcess(MainData,Pr,pwd)
                
                cdir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
                cfile = os.path.join(cdir,"Tests/Almond3D/almond_H1_P"+str(MainData.C+1)+".mat")
                Dict = loadmat(cfile)

                # Checking higher order mesh generators results
                mesh_checker(mesh,Dict)
                del Dict
                gc.collect()

                # Checking Dirichlet data from CAD
                ColumnsIn, ColumnsOut, AppliedDirichlet = GetDirichletBoundaryConditions(mesh,MainData)
                cfile = os.path.join(cdir,"Tests/Almond3D/almond_H1_DirichletData_P"+str(MainData.C+1)+".mat")
                Dict = loadmat(cfile)
                dirichlet_checker(ColumnsOut,AppliedDirichlet,Dict)
                del Dict
                gc.collect()

                print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar)
                if MainData.ndim==2:
                    print('Number of elements is', mesh.elements.shape[0], \
                         'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
                elif MainData.ndim==3:
                    print('Number of elements is', mesh.elements.shape[0], \
                         'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

                # CALL THE MAIN ROUTINE
                TotalDisp = MainSolver(MainData,mesh)

                # if MainData.AssemblyParameters.LoadIncrements != 10:
                    # raise ValueError("Results with different load increments are being compared")

                # cfile = MainData.MeshInfo.FileName.split(".")[0]+"_Solution_"+\
                #     MainData.MaterialArgs.Type+"_Increments_"+\
                #     str(MainData.AssemblyParameters.LoadIncrements)+"_P"+str(MainData.C+1)+".mat"

                # Dict = loadmat(cfile)
                # # Checking the final solution 
                # final_solution_checker(MainData,TotalDisp,Dict)
                # del Dict
                # gc.collect()

                Dict = {'TotalDisp':TotalDisp, 'ScaledJacobian':MainData.ScaledJacobian,'LoadIncrements':MainData.AssemblyParameters.LoadIncrements,
                    'YoungsModulus': MainData.MaterialArgs.E, 'PoissonRatio':MainData.MaterialArgs.nu,'AnalysisType':MainData.AnalysisType,
                    'MaterialArgsType':MainData.MaterialArgs.Type,'SolverType':MainData.solve.type,'SolverSubType':MainData.solve.sub_type,
                    'SolverTol':MainData.solve.tol,'E_A':MainData.MaterialArgs.E_A,'G_A':MainData.MaterialArgs.G_A,
                    'AnisotropicOrientations':MainData.MaterialArgs.AnisotropicOrientations}

                spath = MainData.MeshInfo.FileName.split(".")[0]+"_Solution_"+\
                    MainData.MaterialArgs.Type+"_Increments_"+str(MainData.AssemblyParameters.LoadIncrements)+"_P"+str(MainData.C+1)+".mat"
                savemat(spath,Dict)


def RunTestCases():

    print("\n=========================================================================")
    print("                       RUNNING FLORENCE TEST-SUITE                         ")
    print("=========================================================================\n")
    print("                RUNNING MechanicalComponent2D TEST CASES                   ")

    MainData.__NO_DEBUG__ = True
    MainData.__VECTORISATION__ = True
    MainData.__PARALLEL__ = True
    MainData.numCPU = MP.cpu_count()
    MainData.__PARALLEL__ = False
    MainData.__MEMORY__ = 'SHARED'
    # MainData.__MEMORY__ = 'DISTRIBUTED'
    
    MainData.norder = 2
    MainData.plot = (0, 3)
    nrplot = (0, 'last')
    MainData.write = 0

    import Tests.Almond3D.ProblemData as Pr
    # import Tests.MechanicalComponent2D.ProblemData as Pr
    # import Tests.Annular_Circle.ProblemData as Pr
    # import Tests.Leaf.ProblemData as Pr
    # import Tests.Sphere.ProblemData as Pr # MeshPy with 10 points
    # import Tests.MechanicalComponent3D.ProblemData as Pr
    # import Tests.F6.ProblemData as Pr
    # import Tests.Falcon3D.ProblemData as Pr
    # from Core.FiniteElements.ApplyDirichletBoundaryConditions import GetDirichletBoundaryConditions

    # GET THE CURRENT DIRECTORY PATH
    pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
    # material_models = ["IncrementalLinearElastic","TranservselyIsotropicLinearElastic",
                        # "NeoHookean_2","MooneyRivlin","NearlyIncompressibleMooneyRivlin",
                        # "BonetTranservselyIsotropicHyperElastic"]

    material_models = ["NeoHookean_2","MooneyRivlin","NearlyIncompressibleMooneyRivlin",
                        "BonetTranservselyIsotropicHyperElastic"]

    # material_models = ["IncrementalLinearElastic"]

    # for p in [2,3,4,5,6]:
    for p in [2,3,4,5]:    
        MainData.C = p - 1

        # MainData.MaterialArgs.Type = "IncrementalLinearElastic"
        # MainData.MaterialArgs.Type = 'TranservselyIsotropicLinearElastic'
        # MainData.MaterialArgs.Type = "NeoHookean_2"
        # MainData.MaterialArgs.Type = 'MooneyRivlin'
        # MainData.MaterialArgs.Type = 'NearlyIncompressibleMooneyRivlin'
        # MainData.MaterialArgs.Type = 'BonetTranservselyIsotropicHyperElastic'

        # for Increment in [1,10]:
        # for Increment in [5]:
        for Increment in [10]:    
            MainData.LoadIncrement = Increment

            # if p > 3 and Increment == 10:
                # continue

            for material_model in material_models:

                material_func = getattr(Florence.MaterialLibrary,material_model,None)

                material = material_func(3,youngs_modulus=1.0e05,poissons_ratio=0.485,
                    E_A=2.5e05,G_A=5.0e04)
                # material = material_func(2,youngs_modulus=1.0e05,poissons_ratio=0.4,
                    # E_A=2.5e05,G_A=5.0e04)

                # READ PROBLEM DATA FILE
                mesh, boundary_condition = Pr.ProblemData(MainData)
                MainData.nvar = material.nvar
                MainData.ndim = material.ndim


                
                # PRE-PROCESS
                print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...')

                # mesh = PreProcess(MainData,material,Pr,pwd)
                PreProcess(MainData,mesh,material,Pr,pwd)
                # mesh.face_to_surface = np.loadtxt(MainData.Path.Problem+"/face_to_surface_mapped.dat").astype(np.int64)
                # mesh.face_to_surface = np.loadtxt(MainData.Path.Problem+"/f6_iso_face_to_surface_mapped.dat").astype(np.int64)

                # from Florence import Mesh
                # lmesh = Mesh()
                # lmesh.ReadGIDMesh(MainData.MeshInfo.FileName,"tet")
                # mesh.face_to_surface = lmesh.face_to_surface

                if material.is_transversely_isotropic:
                    material.GetFibresOrientation(mesh)
                
                cdir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
                cfile = os.path.join(cdir,"Tests/Almond3D/almond_H1_P"+str(MainData.C+1)+".mat")
                Dict = loadmat(cfile)

                # Checking higher order mesh generators results
                mesh_checker(mesh,Dict)
                del Dict
                gc.collect()

                # Checking Dirichlet data from CAD
                # ColumnsIn, ColumnsOut, AppliedDirichlet = GetDirichletBoundaryConditions(mesh,MainData)
                boundary_condition.GetDirichletBoundaryConditions(MainData,mesh,material)
                cfile = os.path.join(cdir,"Tests/Almond3D/almond_H1_DirichletData_P"+str(MainData.C+1)+".mat")
                Dict = loadmat(cfile)
                dirichlet_checker(boundary_condition.columns_out,boundary_condition.applied_dirichlet,Dict)
                del Dict
                gc.collect()

                print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar)
                if MainData.ndim==2:
                    print('Number of elements is', mesh.elements.shape[0], \
                         'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
                elif MainData.ndim==3:
                    print('Number of elements is', mesh.elements.shape[0], \
                         'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

                # CALL THE MAIN ROUTINE
                TotalDisp = MainSolver(MainData,mesh,material,boundary_condition)

                if MainData.AssemblyParameters.LoadIncrements != 10:
                    raise ValueError("Results with different load increments are being compared")

                cfile = mesh.filename.split(".")[0]+"_Solution_"+\
                    material.mtype+"_Increments_"+\
                    str(MainData.AssemblyParameters.LoadIncrements)+"_P"+str(MainData.C+1)+".mat"

                Dict = loadmat(cfile)
                # Checking the final solution 
                final_solution_checker(MainData,material,TotalDisp,Dict)
                del Dict
                gc.collect()

                if material.anisotropic_orientations is None:
                    material.anisotropic_orientations = np.array([np.NAN])

                post_process = PostProcess(MainData.ndim,MainData.nvar)
                if material.is_transversely_isotropic:
                    post_process.is_material_anisotropic = True
                    post_process.SetAnisotropicOrientations(material.anisotropic_orientations)

                if MainData.AnalysisType == 'Nonlinear':
                    post_process.SetBases(postdomain=MainData.PostDomain)
                    qualities = post_process.MeshQualityMeasures(mesh,TotalDisp,plot=False,show_plot=False)
                    MainData.isScaledJacobianComputed = qualities[0]
                    MainData.ScaledJacobian = qualities[3]

                Dict = {'TotalDisp':TotalDisp, 'ScaledJacobian':MainData.ScaledJacobian,'LoadIncrements':MainData.AssemblyParameters.LoadIncrements,
                    'YoungsModulus': material.E, 'PoissonRatio':material.nu,'AnalysisType':MainData.AnalysisType,
                    'MaterialArgsType':material.mtype,'SolverType':MainData.solver.solver_type,'SolverSubType':MainData.solver.solver_subtype,
                    'SolverTol':MainData.solver.iterative_solver_tolerance,'E_A':material.E_A,'G_A':material.G_A,
                    'AnisotropicOrientations':material.anisotropic_orientations}

                spath = mesh.filename.split(".")[0]+"_Solution_"+\
                    material_model+"_Increments_"+str(MainData.AssemblyParameters.LoadIncrements)+"_P"+str(MainData.C+1)+".mat"
                # spath = mesh.filename.split(".")[0]+"_Solution_"+\
                    # material_model+"_Nonlinear_P"+str(MainData.C+1)+".mat"

                # savemat(spath,Dict)





# RUN TEST-CASES
# if __name__ == "__main__":
RunTestCases()
