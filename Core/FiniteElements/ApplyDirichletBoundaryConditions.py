import numpy as np 
import scipy as sp
from DirichletBoundaryDataFromCAD import IGAKitWrapper, PostMeshWrapper
from time import time

# ROUTINE FOR APPLYING DIRICHLET BOUNDARY CONDITIONS
def GetDirichletBoundaryConditions(mesh,MainData):

    #######################################################
    nvar = MainData.nvar
    ndim = MainData.ndim

    ColumnsOut = []; AppliedDirichlet = []


    #----------------------------------------------------------------------------------------------------#
    #-------------------------------------- NURBS BASED SOLUTION ----------------------------------------#
    #----------------------------------------------------------------------------------------------------#
    if MainData.BoundaryData.Type == 'nurbs':

        tCAD = time()

        IsHighOrder = getattr(MainData.MeshInfo,"IsHighOrder",False)
        IsDirichletComputed = getattr(MainData.BoundaryData,"IsDirichletComputed",None)
            
        IsHighOrder = False

        if IsHighOrder is False:

            if IsDirichletComputed is None:

                # GET DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY FROM CAD
                if MainData.BoundaryData.RequiresCAD:
                    # CALL POSTMESH WRAPPER
                    nodesDBC, Dirichlet = PostMeshWrapper(MainData,mesh)
                else:
                    # CALL IGAKIT WRAPPER
                    nodesDBC, Dirichlet = IGAKitWrapper(MainData,mesh)

            else:
                nodesDBC, Dirichlet = MainData.BoundaryData.nodesDBC, MainData.BoundaryData.Dirichlet                


            nOfDBCnodes = nodesDBC.shape[0]
            for inode in range(nOfDBCnodes):
                for i in range(nvar):
                    ColumnsOut = np.append(ColumnsOut,nvar*nodesDBC[inode]+i)
                    AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])

            # FIX THE DOF IN THE REST OF THE BOUNDARY 
            if ndim==2:
                Rest_DOFs = np.setdiff1d(np.unique(mesh.edges),nodesDBC)
            elif ndim==3:
                Rest_DOFs = np.setdiff1d(np.unique(mesh.faces),nodesDBC)
            for inode in range(Rest_DOFs.shape[0]):
              for i in range(nvar):
                  ColumnsOut = np.append(ColumnsOut,nvar*Rest_DOFs[inode]+i)
                  AppliedDirichlet = np.append(AppliedDirichlet,0.0)

            print 'Finished identifying Dirichlet boundary conditions from CAD geometry. Time taken ', time()-tCAD, 'seconds'

            # end = -3
            # np.savetxt(MainData.MeshInfo.FileName.split(".")[0][:end]+"_Dirichlet_"+"P"+str(MainData.C+1)+".dat",AppliedDirichlet,fmt="%9.16f")
            # np.savetxt(MainData.MeshInfo.FileName.split(".")[0][:end]+"_ColumnsOut_"+"P"+str(MainData.C+1)+".dat",ColumnsOut)
            # # np.savetxt(MainData.MeshInfo.FileName.split(".")[0][:end]+"_PlanarMeshFaces_"+"P"+str(MainData.C+1)+".dat",MainData.planar_mesh_faces)

        else:
            
            end = -3
            AppliedDirichlet = np.loadtxt(MainData.MeshInfo.FileName.split(".")[0][:end]+"_Dirichlet_"+"P"+str(MainData.C+1)+".dat",dtype=np.float64)
            ColumnsOut = np.loadtxt(MainData.MeshInfo.FileName.split(".")[0][:end]+"_ColumnsOut_"+"P"+str(MainData.C+1)+".dat")

            # AppliedDirichlet = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_big_Dirichlet_P3.dat")
            # ColumnsOut = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_big_ColumnsOut_P3.dat")

            # AppliedDirichlet = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/mechanicalComplex_Dirichlet_P4.dat")
            # ColumnsOut = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/mechanicalComplex_ColumnsOut_P4.dat")

            # AppliedDirichlet = np.loadtxt("/home/roman/LayerSolution/Layer_dd/Layer_dd_AppliedDirichlet.dat",dtype=np.float64)
            # ColumnsOut = np.loadtxt("/home/roman/LayerSolution/Layer_dd/Layer_dd_ColumnsOut.dat")

            # AppliedDirichlet = np.loadtxt(MainData.DirichletName,dtype=np.float64)
            # ColumnsOut = np.loadtxt(MainData.ColumnsOutName)

            # AppliedDirichlet = AppliedDirichlet*MainData.CurrentIncr/MainData.nStep
            # AppliedDirichlet = AppliedDirichlet*1.0/MainData.nStep

            print 'Finished identifying Dirichlet boundary conditions from CAD geometry. Time taken ', time()-tCAD, 'seconds'



        ############################
        # print np.max(AppliedDirichlet), mesh.Bounds
        # exit()
        ############################

    #----------------------------------------------------------------------------------------------------#
    #------------------------------------- NON-NURBS BASED SOLUTION -------------------------------------#
    #----------------------------------------------------------------------------------------------------#

    elif MainData.BoundaryData.Type == 'straight' or MainData.BoundaryData.Type == 'mixed':
        # IF DIRICHLET BOUNDARY CONDITIONS ARE APPLIED DIRECTLY AT NODES
        if MainData.BoundaryData().DirichArgs.Applied_at == 'node':
            # GET UNIQUE NODES AT THE BOUNDARY
            unique_edge_nodes = []
            if ndim==2:
                unique_edge_nodes = np.unique(mesh.edges)
            elif ndim==3:
                unique_edge_nodes = np.unique(mesh.faces)
            # ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
            # unique_edge_nodes = np.unique(mesh.elements) 


            MainData.BoundaryData().DirichArgs.points = mesh.points
            MainData.BoundaryData().DirichArgs.edges = mesh.edges
            for inode in range(0,unique_edge_nodes.shape[0]):
                coord_node = mesh.points[unique_edge_nodes[inode]]
                MainData.BoundaryData().DirichArgs.node = coord_node
                MainData.BoundaryData().DirichArgs.inode = unique_edge_nodes[inode]

                Dirichlet = MainData.BoundaryData().DirichletCriterion(MainData.BoundaryData().DirichArgs)

                # COMMENTED RECENTLY IN FAVOR OF WHAT APPEARS BELOW
                # if type(Dirichlet) is None:
                #   pass
                # else:
                #   for i in range(nvar):
                #       # if type(Dirichlet[i]) is list:
                #       if Dirichlet[i] is None:
                #           pass
                #       else:
                #           # ColumnsOut = np.append(ColumnsOut,nvar*inode+i) # THIS IS INVALID
                #           # ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
                #           ColumnsOut = np.append(ColumnsOut,nvar*unique_edge_nodes[inode]+i)
                #           AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[i])

                if type(Dirichlet) is not None:
                    for i in range(nvar):
                        if Dirichlet[i] is not None:
                            # ColumnsOut = np.append(ColumnsOut,nvar*inode+i) # THIS IS INVALID
                            # ACTIVATE THIS FOR DEBUGGING ELECTROMECHANICAL PROBLEMS
                            ColumnsOut = np.append(ColumnsOut,nvar*unique_edge_nodes[inode]+i)
                            AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[i])


    # GENERAL PROCEDURE - GET REDUCED MATRICES FOR FINAL SOLUTION
    ColumnsOut = ColumnsOut.astype(np.int64)
    ColumnsIn = np.delete(np.arange(0,nvar*mesh.points.shape[0]),ColumnsOut)


    return ColumnsIn, ColumnsOut, AppliedDirichlet





def GetReducedMatrices(stiffness,F,ColumnsIn,Analysis=None,mass=None):

    # GET REDUCED FORCE VECTOR
    F_b = F[ColumnsIn,0]

    # GET REDUCED STIFFNESS MATRIX
    stiffness_b = stiffness[ColumnsIn,:][:,ColumnsIn]

    # GET REDUCED MASS MATRIX
    mass_b = np.array([])
    if Analysis != 'Static':
        mass_b = mass[ColumnsIn,:][:,ColumnsIn]

    return stiffness_b, F_b, mass_b



def ApplyDirichletGetReducedMatrices(stiffness,F,ColumnsIn,ColumnsOut,AppliedDirichlet,Analysis=None,mass=None):

    # APPLY DIRICHLET BOUNDARY CONDITIONS
    for i in range(0,ColumnsOut.shape[0]):
        F = F - AppliedDirichlet[i]*stiffness.getcol(ColumnsOut[i])

    # GET REDUCED FORCE VECTOR
    F_b = F[ColumnsIn,0]

    # print int(sp.__version__.split('.')[1] )
    # FOR UMFPACK SOLVER TAKE SPECIAL CARE
    if int(sp.__version__.split('.')[1]) < 15:
        F_b_umf = np.zeros(F_b.shape[0])
        # F_b_umf[:] = F_b[:,0] # DOESN'T WORK
        for i in range(F_b_umf.shape[0]):
            F_b_umf[i] = F_b[i,0]
        F_b = np.copy(F_b_umf)

    # GET REDUCED STIFFNESS
    stiffness_b = stiffness[ColumnsIn,:][:,ColumnsIn]

    # GET REDUCED MASS MATRIX
    mass_b = np.array([])
    if Analysis != 'Static':
        mass = mass[ColumnsIn,:][:,ColumnsIn]

    return stiffness_b, F_b, F, mass_b
