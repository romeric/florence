from __future__ import print_function
import os
from time import time
from copy import deepcopy

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Florence.FiniteElements.PreProcess import PreProcess
from Florence.FiniteElements.PostProcess import *
from Florence.FiniteElements.Solvers.Solver import *
from Florence.FiniteElements.ComputeErrorNorms import *


###########################################################################################################
# PROBLEM FILE DIRECTORIES
# import Examples.FiniteElements.Nonlinear_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.MultiPhysics_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.MultiPhysics_Fibre_3D.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Ellipse_Cylinder.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Beam.ProblemData as Pr

# import Examples.FiniteElements.Hollow_Arc_Tri.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle_Electromechanics.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle_Nurbs.ProblemData as Pr
# import Examples.FiniteElements.MechanicalComponent2D.ProblemData as Pr
# import Examples.FiniteElements.Wing2D.ProblemData as Pr
# import Examples.FiniteElements.Naca_Isotropic.ProblemData as Pr
# import Examples.FiniteElements.RAE2822.ProblemData as Pr
# import Examples.FiniteElements.Misc.ProblemData as Pr
# import Examples.FiniteElements.Leaf.ProblemData as Pr
# import Examples.FiniteElements.Misc3D.ProblemData as Pr
# import Examples.FiniteElements.Sphere.ProblemData as Pr
import Examples.FiniteElements.Almond3D.ProblemData as Pr
# import Examples.FiniteElements.Falcon3D.ProblemData as Pr
# import Examples.FiniteElements.F6.ProblemData as Pr
# import Examples.FiniteElements.Drill.ProblemData as Pr
# import Examples.FiniteElements.Valve.ProblemData as Pr
# import Examples.FiniteElements.MechanicalComponent3D.ProblemData as Pr

###########################################################################################################


def main(MainData, DictOutput=None, nStep=0):

    # GET THE CURRENT DIRECTORY PATH
    pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

    # READ PROBLEM DATA FILE
    mesh, material, boundary_condition = Pr.ProblemData(MainData)
    
    # PRE-PROCESS
    print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...')

    PreProcess(MainData,mesh,material,Pr,pwd)

    print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar)
    if MainData.ndim==2:
        print('Number of elements is', mesh.elements.shape[0], \
             'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
    elif MainData.ndim==3:
        print('Number of elements is', mesh.elements.shape[0], \
             'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

    # CALL THE MAIN ROUTINE
    MainData.Timer = time()
    TotalDisp = MainSolver(MainData,mesh,material,boundary_condition)
    MainData.Timer = time() - MainData.Timer

    # POST-PROCESS
    # print ('Post-Processing the information...')
    # PostProcess().StressRecovery(MainData,mesh,TotalDisp) 


    # # CHECK IF ALL THE FACE POINTS COORDINATES ARE ON THE SPHERE
    # vpoints = mesh.points + TotalDisp[:,:,-1]
    # un_faces = np.unique(mesh.faces)
    # print(np.linalg.norm(vpoints[un_faces,:],axis=1))
    # print(np.allclose(np.linalg.norm(vpoints[un_faces,:],axis=1),1))

    # vpoints = mesh.points + TotalDisp[:,:,-1]
    # un_faces = np.unique(mesh.faces)
    # print(np.linalg.norm(vpoints[un_faces,:],axis=1))


    # ####################
    # from scipy.io import savemat
    # if TotalDisp.ndim == 3:
    #     mesh.points = mesh.points + TotalDisp[:,:,-1]
    # else:
    #     mesh.points = mesh.points + TotalDisp

    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 'faces':mesh.faces, 'edges':mesh.edges}
    # savemat('/home/roman/Dropbox/Falcon3DBig_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/F6ISO_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # return
    #####################



    # ####################
    # from scipy.io import savemat
    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 'faces':mesh.faces,
    #     'TotalDisp':TotalDisp,
    #     'ScaledJacobian':MainData.ScaledJacobian, 
    #     'C':MainData.C, 'ProjFlags':MainData.BoundaryData().ProjectionCriteria(mesh)}
    # # savemat('/home/roman/Sphere_P'+str(MainData.C+1)+'.mat',Dict)
    # savemat('/home/roman/Dropbox/Almond3D_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/Falcon3DIso_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/Falcon3DBig_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/Drill_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/Valve_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/MechanicalComponent3D_P'+str(MainData.C+1)+'_New_200.mat',Dict,do_compression=True)
    # # savemat('/home/roman/Dropbox/F6_P'+str(MainData.C+1)+'.mat',Dict)
    # # savemat('/home/roman/Dropbox/F6Iso_P'+str(MainData.C+1)+'.mat',Dict)
    # # savemat('/home/roman/LayerSolution/Layer_dd/f6BL_Layer_dd_Sol_P'+str(MainData.C+1)+'.mat',Dict)
    # # savemat(MainData.SolName,Dict,do_compression=True)

    exit()
    # # # return
    #####################


    # if nStep == 1:
    #     MainData.mesh = mesh
    #     MainData.mesh.points = mesh.points + TotalDisp[:,:MainData.ndim,-1]
    
    #------------------------------------------------------------------------

    if MainData.AssemblyParameters.FailedToConverge==False:

        post_process = PostProcess(MainData.ndim,MainData.nvar)
        if material.is_transversely_isotropic:
            post_process.is_material_anisotropic = True
            post_process.SetAnisotropicOrientations(material.anisotropic_orientations)

        if MainData.AnalysisType == 'Nonlinear':
            post_process.SetBases(postdomain=MainData.PostDomain)
            qualities = post_process.MeshQualityMeasures(mesh,TotalDisp,plot=False,show_plot=False)
            MainData.isScaledJacobianComputed = qualities[0]
            MainData.ScaledJacobian = qualities[3]

        if MainData.AnalysisType == "Linear":
            vmesh = deepcopy(mesh)
            vmesh.points = vmesh.points + TotalDisp[:,:,MainData.AssemblyParameters.LoadIncrements-1]
            # TotalDisp = np.sum(TotalDisp,axis=2)[:,:,None]
        else:
            vmesh = mesh

        if boundary_condition.projection_flags is None:
            # ProjFlags = np.ones(mesh.faces.shape[0],dtype=np.int64)
            if MainData.ndim == 1:
                boundary_condition.projection_flags = np.ones(mesh.faces.shape[0],dtype=np.int64)
            elif MainData.ndim == 2:
                boundary_condition.projection_flags = np.ones(mesh.edges.shape[0],dtype=np.int64)
        # else:
            # ProjFlags = MainData.BoundaryData().ProjectionCriteria(mesh)

        # TotalDisp = np.zeros_like(TotalDisp)
        # TotalDisp = TotalDisp/3.
        # MainData.ScaledJacobian = np.zeros_like(MainData.ScaledJacobian)
        # PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)

        post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=MainData.ScaledJacobian,
            ProjectionFlags=boundary_condition.projection_flags,InterpolationDegree=40)
        import matplotlib.pyplot as plt
        plt.show()
    else:
        MainData.ScaledJacobian = np.zeros(mesh.nelem)+np.NAN
        MainData.ScaledFF = np.zeros(mesh.nelem)+np.NAN
        MainData.ScaledHH = np.zeros(mesh.nelem)+np.NAN
        MainData.ScaledFNFN = np.zeros(mesh.nelem)+np.NAN
        MainData.ScaledCNCN = np.zeros(mesh.nelem)+np.NAN

    # if DictOutput is not None:
    #     DictOutput['MeshPoints_P'+str(MainData.C+1)] = mesh.points
    #     DictOutput['MeshElements_P'+str(MainData.C+1)] = mesh.elements+1
    #     DictOutput['MeshEdges_P'+str(MainData.C+1)] = mesh.edges+1
    #     if MainData.ndim==3:
    #         DictOutput['MeshFaces_P'+str(MainData.C+1)] = mesh.faces+1
    #     DictOutput['TotalDisplacement_P'+str(MainData.C+1)] = TotalDisp
    #     DictOutput['nSteps'] = MainData.AssemblyParameters.LoadIncrements

    #---------------------------------------------------------------------------------

    # Compute Error Norms
    # L2Norm=0; EnergyNorm=0
    # L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

    # CheapNorm(MainData,mesh,TotalDisp)

    #----------------------------------------------------------------------------------

    # DEGUGGING 
    if MainData.__NO_DEBUG__ is False:
        # NOTE THAT PYTHON'S BUILT-IN DEBUGGER IS ALWAYS TRUE __debug__ WITHOUT -0 FLAG
        from Core import debug 
        debug(MainData,mesh,TotalDisp)

