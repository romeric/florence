from __future__ import print_function
import os
from time import time
from copy import deepcopy

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Core.FiniteElements.PreProcess import PreProcess
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Solvers.Solver import *
from Core.FiniteElements.ComputeErrorNorms import *


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
import Examples.FiniteElements.MechanicalComponent2D.ProblemData as Pr
# import Examples.FiniteElements.Wing2D.ProblemData as Pr
# import Examples.FiniteElements.Naca_Isotropic.ProblemData as Pr
# import Examples.FiniteElements.RAE2822.ProblemData as Pr
# import Examples.FiniteElements.Misc.ProblemData as Pr
# import Examples.FiniteElements.Tests.ProblemData as Pr
# import Examples.FiniteElements.Sphere.ProblemData as Pr
# import Examples.FiniteElements.Almond3D.ProblemData as Pr
# import Examples.FiniteElements.Falcon3D.ProblemData as Pr

###########################################################################################################


def main(MainData, DictOutput=None, nStep=0):

    # GET THE CURRENT DIRECTORY PARTH
    pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

    # READ PROBLEM DATA FILE
    Pr.ProblemData(MainData)
    
    # PRE-PROCESS
    print('Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...')

    mesh = PreProcess(MainData,Pr,pwd)

    print('Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar)
    if MainData.ndim==2:
        print('Number of elements is', mesh.elements.shape[0], \
             'and number of boundary nodes is', np.unique(mesh.edges).shape[0])
    elif MainData.ndim==3:
        print('Number of elements is', mesh.elements.shape[0], \
             'and number of boundary nodes is', np.unique(mesh.faces).shape[0])

    # CALL THE MAIN ROUTINE
    MainData.Timer = time()
    TotalDisp = MainSolver(MainData,mesh)
    MainData.Timer = time() - MainData.Timer

    # POST-PROCESS
    # print ('Post-Processing the information...')
    # PostProcess().StressRecovery(MainData,mesh,TotalDisp) 


    # # CHECK IF ALL THE FACE POINTS COORDINATES ARE ON THE SPHERE
    # vpoints = mesh.points + TotalDisp[:,:,-1]
    # # print(vpoints[mesh.elements[1502,:],:])
    # un_faces = np.unique(mesh.faces)
    # print(np.linalg.norm(vpoints[un_faces,:],axis=1))
    # print(np.allclose(np.linalg.norm(vpoints[un_faces,:],axis=1),1))

    # vpoints = mesh.points + TotalDisp[:,:,-1]
    # un_faces = np.unique(mesh.faces)
    # print np.linalg.norm(vpoints[un_faces,:],axis=1)


    #####################
    # from scipy.io import savemat
    # Dict = {'InitialX':mesh.points, 'T':mesh.elements, 
    #     'TotalDisplacement':TotalDisp[:,:,MainData.AssemblyParameters.LoadIncrements-1],
    #     'FinalX':mesh.points+TotalDisp[:,:,MainData.AssemblyParameters.LoadIncrements-1],
    #     'ScaledJacobian':MainData.ScaledJacobian,
    #     'nIncrements':MainData.AssemblyParameters.LoadIncrements}
    # savemat('/home/roman/Dropbox/Wing2D_Results_P'+str(MainData.C+1)+'.mat',Dict)
    #####################


    #####################
    # from scipy.io import savemat
    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 'faces':mesh.faces,
    #     'TotalDisp':TotalDisp,
    #     'ScaledJacobian':MainData.ScaledJacobian, 
    #     'C':MainData.C, 'ProjFlags':MainData.BoundaryData().ProjectionCriteria(mesh)}
    # # savemat('/home/roman/Dropbox/Almond3D_P'+str(MainData.C+1)+'.mat',Dict)
    # savemat('/home/roman/Dropbox/Almond3D_H2_P'+str(MainData.C+1)+'.mat',Dict)
    # # savemat('/home/roman/Sphere_P'+str(MainData.C+1)+'.mat',Dict)
    # exit()
    #####################


    # np.savetxt("/home/roman/Dropbox/PostMesh/post_mesh/examples/sphere/sphere_points.dat",mesh.points,fmt="%10.9f",delimiter=',')
    # np.savetxt("/home/roman/Dropbox/PostMesh/post_mesh/examples/sphere/sphere_elements.dat",mesh.elements,delimiter=',')
    # np.savetxt("/home/roman/Dropbox/PostMesh/post_mesh/examples/sphere/sphere_faces.dat",mesh.faces,delimiter=',')
    # np.savetxt("/home/roman/Dropbox/PostMesh/post_mesh/examples/sphere/sphere_edges.dat",mesh.edges,delimiter=',')
    # from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
    # fekete = FeketePointsTri(MainData.C)
    # np.savetxt("/home/roman/Dropbox/PostMesh/post_mesh/examples/sphere/nodal_spacing_p4.dat",fekete,fmt="%10.9f",delimiter=',')
    # exit()

    # np.savetxt("/home/roman/Dropbox/PostMesh/examples/leaf/leaf_points.dat",mesh.points,fmt="%10.9f",delimiter=',')
    # np.savetxt("/home/roman/Dropbox/PostMesh/examples/leaf/leaf_elements.dat",mesh.elements,delimiter=',')
    # # np.savetxt("/home/roman/Dropbox/PostMesh/examples/leaf/leaf_faces.dat",mesh.faces,delimiter=',')
    # np.savetxt("/home/roman/Dropbox/PostMesh/examples/leaf/leaf_edges.dat",mesh.edges,delimiter=',')
    # from Core.QuadratureRules import GaussLobattoQuadrature
    # fekete = GaussLobattoQuadrature(MainData.C)
    # np.savetxt("/home/roman/Dropbox/PostMesh/examples/leaf/nodal_spacing_p8.dat",fekete,fmt="%10.9f",delimiter=',')
    # exit()
    


    if nStep == 1:
        MainData.mesh = mesh
        MainData.mesh.points = mesh.points + TotalDisp[:,:MainData.ndim,-1]
    
    #------------------------------------------------------------------------

    if MainData.AssemblyParameters.FailedToConverge==False:
        if MainData.AnalysisType == 'Nonlinear':
            PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp,show_plot=False)
            pass
        if MainData.AnalysisType == "Linear":
            vmesh = deepcopy(mesh)
            vmesh.points = vmesh.points + TotalDisp[:,:,MainData.AssemblyParameters.LoadIncrements-1]
        else:
            vmesh = mesh

        ProjFunc = getattr(MainData.BoundaryData,'ProjectionCriteria',None)
        if ProjFunc is None:
            ProjFlags = np.ones(mesh.faces.shape[0],dtype=np.int64)
        else:
            ProjFlags = MainData.BoundaryData().ProjectionCriteria(mesh)

        # print(np.max(mesh.points[:,0]), np.min(mesh.points[:,0]))
        # ProjFlags = MainData.BoundaryData().PlottingCriteria(mesh)
        # # exit()

        # TotalDisp = np.zeros_like(TotalDisp)
        # MainData.ScaledJacobian = np.zeros_like(MainData.ScaledJacobian)
        # PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
        # PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=MainData.ScaledJacobian,
        #     ProjectionFlags=ProjFlags,InterpolationDegree=40)
        # import matplotlib.pyplot as plt
        # plt.show()
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

    #########################################################
    # # For WriteCurvedMeshFiles
    # if DictOutput is not None:
    #     DictOutput['MeshPoints_P'+str(MainData.C+1)] = mesh.points
    #     DictOutput['MeshElements_P'+str(MainData.C+1)] = mesh.elements+1
    #     # DictOutput['MeshEdges_P'+str(MainData.C+1)] = mesh.edges+1
    #     if MainData.ndim==3:
    #         DictOutput['MeshFaces_P'+str(MainData.C+1)] = mesh.faces+1
    #     DictOutput['TotalDisplacement_P'+str(MainData.C+1)+"_"+MainData.AnalysisType+"_"+MainData.MaterialArgs.Type] = TotalDisp
    #     DictOutput['ScaledJacobian_P'+str(MainData.C+1)+"_"+MainData.AnalysisType+"_"+MainData.MaterialArgs.Type] = MainData.ScaledJacobian
    #     DictOutput['nSteps'] = MainData.AssemblyParameters.LoadIncrements

    # For WriteCurvedMeshFiles for the case where nonlinear struggles for Wing2D
    # Results = {'PolynomialDegree':MainData.C+1,
    #     'PoissonsRatio':MainData.MaterialArgs.nu,'Youngs_Modulus':MainData.MaterialArgs.E,
    #     'MeshPoints':mesh.points,'MeshElements':mesh.elements+1,'TotalDisplacement':TotalDisp,
    #     'ScaledJacobian':MainData.ScaledJacobian}
    # spath = "/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles/Wing2D_Nonlinear.mat"
    # from scipy.io import savemat
    # savemat(spath,Results)
    #########################################################

    # vpoints = mesh.points + TotalDisp[:,:,-1]
    # mpath = "/home/roman/Dropbox/Matlab_Files/tetplots/"
    # # mname = "sphere"
    # # mname = "torus"
    # mname = "almond"

    # np.savetxt(mpath+'elements_'+mname+'_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
    # np.savetxt(mpath+'points_'+mname+'_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
    # np.savetxt(mpath+'vpoints_'+mname+'_p'+str(MainData.C+1)+'.dat', vpoints,fmt='%10.9f',delimiter=',')
    # np.savetxt(mpath+'faces_'+mname+'_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')
    # np.savetxt(mpath+'sjacobian_'+mname+'_p'+str(MainData.C+1)+'.dat',MainData.ScaledJacobian,fmt='%8.9f',delimiter=',')

    # np.savetxt(mpath+'pfaces_'+mname+'_p'+str(MainData.C+1)+'.dat',MainData.BoundaryData().ProjectionCriteria(mesh))


    # mesh.WriteVTK(fname="/home/roman/Dropbox/dd.vtu",pdata=TotalDisp[:,:,-1])
    # mesh.WriteVTK(pdata=TotalDisp[:,:,-1])



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

