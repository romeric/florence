from __future__ import print_function
import os
from time import time
from copy import deepcopy

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Florence.FiniteElements.PreProcess import PreProcess
from Florence.FiniteElements.PostProcess import *
from Florence.FiniteElements.ComputeErrorNorms import *


###########################################################################################################
# PROBLEM FILE DIRECTORIES
# import examples.FiniteElements.Nonlinear_3D_Cube.ProblemData as Pr
# import examples.FiniteElements.MultiPhysics_3D_Cube.ProblemData as Pr
# import examples.FiniteElements.MultiPhysics_Fibre_3D.ProblemData as Pr
# import examples.FiniteElements.Nonlinear_Electromechanics_3D_Ellipse_Cylinder.ProblemData as Pr
# import examples.FiniteElements.Nonlinear_Electromechanics_3D_Cube.ProblemData as Pr
# import examples.FiniteElements.Nonlinear_Electromechanics_3D_Beam.ProblemData as Pr

# import examples.FiniteElements.Hollow_Arc_Tri.ProblemData as Pr
# import examples.FiniteElements.Annular_Circle_Electromechanics.ProblemData as Pr
import examples.FiniteElements.Annular_Circle.ProblemData as Pr
# import examples.FiniteElements.Annular_Circle_Nurbs.ProblemData as Pr
# import examples.FiniteElements.AnnularCircle_MVP.ProblemData as Pr
# import examples.FiniteElements.MechanicalComponent2D.ProblemData as Pr
# import examples.FiniteElements.Wing2D.ProblemData as Pr
# import examples.FiniteElements.Naca_Isotropic.ProblemData as Pr
# import examples.FiniteElements.RAE2822.ProblemData as Pr
# import examples.FiniteElements.Misc.ProblemData as Pr
# import examples.FiniteElements.Leaf.ProblemData as Pr
# import examples.FiniteElements.Misc3D.ProblemData as Pr
# import examples.FiniteElements.Sphere.ProblemData as Pr
# import examples.FiniteElements.Almond3D.ProblemData as Pr
# import examples.FiniteElements.Falcon3D.ProblemData as Pr
# import examples.FiniteElements.F6.ProblemData as Pr
# import examples.FiniteElements.Drill.ProblemData as Pr
# import examples.FiniteElements.Valve.ProblemData as Pr
# import examples.FiniteElements.MechanicalComponent3D.ProblemData as Pr

###########################################################################################################


def main(MainData, DictOutput=None, nStep=0):

    # # GET THE CURRENT DIRECTORY PATH
    # pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

    # READ PROBLEM DATA FILE
    # formulation, mesh, material, boundary_condition, solver, fem_solver = Pr.ProblemData(MainData)
    TotalDisp = Pr.ProblemData(MainData)

    # # # PRE-PROCESS
    # # print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...')

    # # # quadrature_rules, function_spaces = PreProcess(MainData, formulation, mesh, material, fem_solver, Pr, pwd)

    # # print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar)
    # # if MainData.ndim==2:
    # #     print('Number of elements is', mesh.elements.shape[0], \
    # #          'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
    # # elif MainData.ndim==3:
    # #     print('Number of elements is', mesh.elements.shape[0], \
    # #          'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

    # # CALL THE MAIN ROUTINE
    # # MainData.Timer = time()
    # # TotalDisp = MainSolver(function_spaces, formulation, mesh, material, boundary_condition, solver, fem_solver)
    # # TotalDisp = fem_solver.Solve(function_spaces, formulation, mesh, material, boundary_condition, solver)
    # # MainData.Timer = time() - MainData.Timer
    # exit()




    # # POST-PROCESS
    # # print ('Post-Processing the information...')
    # # PostProcess().StressRecovery(MainData,mesh,TotalDisp) 



    # # ####################
    # # from scipy.io import savemat
    # # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    # #     'element_type':mesh.element_type, 'faces':mesh.faces,
    # #     'TotalDisp':TotalDisp,
    # #     'ScaledJacobian':MainData.ScaledJacobian, 
    # #     'C':MainData.C, 'ProjFlags':MainData.BoundaryData().ProjectionCriteria(mesh)}
    # # # savemat('/home/roman/Sphere_P'+str(MainData.C+1)+'.mat',Dict)
    # # savemat('/home/roman/Dropbox/Almond3D_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/Falcon3DIso_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/Falcon3DBig_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/Drill_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/Valve_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'_New_200.mat',Dict,do_compression=True)
    # # # savemat('/home/roman/Dropbox/F6_P'+str(MainData.C+1)+'.mat',Dict)
    # # # savemat('/home/roman/Dropbox/F6Iso_P'+str(MainData.C+1)+'.mat',Dict)
    # # # savemat(MainData.SolName,Dict,do_compression=True)
    # # exit()
    # # # # return
    # #####################


    # # if nStep == 1:
    # #     MainData.mesh = mesh
    # #     MainData.mesh.points = mesh.points + TotalDisp[:,:MainData.ndim,-1]
    
    # #------------------------------------------------------------------------

    # # if MainData.AssemblyParameters.FailedToConverge==False:
    # if not fem_solver.newton_raphson_failed_to_converge:

    #     post_process = PostProcess(formulation.ndim,formulation.nvar)
    #     if material.is_transversely_isotropic:
    #         post_process.is_material_anisotropic = True
    #         post_process.SetAnisotropicOrientations(material.anisotropic_orientations)

    #     if fem_solver.analysis_nature == 'nonlinear':
    #         post_process.SetBases(postdomain=function_spaces[1])
    #         qualities = post_process.MeshQualityMeasures(mesh,TotalDisp,plot=False,show_plot=False)
    #         fem_solver.isScaledJacobianComputed = qualities[0]
    #         fem_solver.ScaledJacobian = qualities[3]

    #     # if fem_solver.analysis_nature == "linear":
    #     #     vmesh = deepcopy(mesh)
    #     #     # vmesh.points = vmesh.points + TotalDisp[:,:,MainData.AssemblyParameters.LoadIncrements-1]
    #     #     vmesh.points = vmesh.points + TotalDisp[:,:,fem_solver.number_of_load_increments-1]
    #     #     # TotalDisp = np.sum(TotalDisp,axis=2)[:,:,None]
    #     # else:
    #     #     vmesh = mesh

    #     if boundary_condition.projection_flags is None:
    #         # ProjFlags = np.ones(mesh.faces.shape[0],dtype=np.int64)
    #         if formulation.ndim == 1:
    #             boundary_condition.projection_flags = np.ones(mesh.faces.shape[0],dtype=np.int64)
    #         elif formulation.ndim == 2:
    #             boundary_condition.projection_flags = np.ones(mesh.edges.shape[0],dtype=np.int64)
    #     # else:
    #         # ProjFlags = MainData.BoundaryData().ProjectionCriteria(mesh)

    #     # TotalDisp = np.zeros_like(TotalDisp)
    #     # TotalDisp = TotalDisp/3.
    #     # MainData.ScaledJacobian = np.zeros_like(MainData.ScaledJacobian)
    #     # PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)

    #     post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=fem_solver.ScaledJacobian,
    #         ProjectionFlags=boundary_condition.projection_flags,InterpolationDegree=40)

    #     # post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp, ProjectionFlags=boundary_condition.projection_flags,
    #     #     InterpolationDegree=0,plot_points=False)
    #     import matplotlib.pyplot as plt
    #     plt.show()
    # else:
    #     MainData.ScaledJacobian = np.zeros(mesh.nelem)+np.NAN
    #     MainData.ScaledFF = np.zeros(mesh.nelem)+np.NAN
    #     MainData.ScaledHH = np.zeros(mesh.nelem)+np.NAN
    #     MainData.ScaledFNFN = np.zeros(mesh.nelem)+np.NAN
    #     MainData.ScaledCNCN = np.zeros(mesh.nelem)+np.NAN

    # # if DictOutput is not None:
    # #     DictOutput['MeshPoints_P'+str(MainData.C+1)] = mesh.points
    # #     DictOutput['MeshElements_P'+str(MainData.C+1)] = mesh.elements+1
    # #     DictOutput['MeshEdges_P'+str(MainData.C+1)] = mesh.edges+1
    # #     if MainData.ndim==3:
    # #         DictOutput['MeshFaces_P'+str(MainData.C+1)] = mesh.faces+1
    # #     DictOutput['TotalDisplacement_P'+str(MainData.C+1)] = TotalDisp
    # #     DictOutput['nSteps'] = MainData.AssemblyParameters.LoadIncrements

    # #---------------------------------------------------------------------------------

    # # Compute Error Norms
    # # L2Norm=0; EnergyNorm=0
    # # L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

    # # CheapNorm(MainData,mesh,TotalDisp)

    # #----------------------------------------------------------------------------------

    # # DEGUGGING 
    # if MainData.__NO_DEBUG__ is False:
    #     # NOTE THAT PYTHON'S BUILT-IN DEBUGGER IS ALWAYS TRUE __debug__ WITHOUT -0 FLAG
    #     from Core import debug 
    #     debug(MainData,mesh,TotalDisp)
