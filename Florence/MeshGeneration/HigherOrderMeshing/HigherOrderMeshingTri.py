from time import time
import numpy as np 
import multiprocessing as MP
import imp

import GetInteriorCoordinates as Gett
import Florence.ParallelProcessing.parmap as parmap
from Florence.QuadratureRules.FeketePointsTri import *
from Florence import FunctionSpace
# import Florence.FunctionSpace.TwoDimensional.Tri.hpNodal
from Florence.FunctionSpace import Tri
# import Florence.FunctionSpace.TwoDimensional.Tri.hpNodal as Tri 
from Florence.Tensor import itemfreq, makezero, unique2d
from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri

#--------------------------------------------------------------------------------------------------------------------------#
# SUPPLEMENTARY FUNCTIONS 
def ElementLoopTri(elem,elements,points,MeshType,eps,Neval):
    xycoord_higher = Gett.GetInteriorNodesCoordinates(points[elements[elem,:],:],MeshType,elem,eps,Neval)
    return xycoord_higher

def HighOrderMeshTri_SEMISTABLE(C,mesh,Decimals=10,Parallel=False,nCPU=1,ComputeAll=False):


    # SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
    if (mesh.elements.shape[0] < 1000) and (C < 8):
        Parallel = False
        nCPU = 1

    if mesh.points.shape[1]!=2:
        raise ValueError('Incompatible mesh coordinates size. mesh.point.shape[1] must be 2')

    eps =  FeketePointsTri(C)
    # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
    hpBases = Tri.hpNodal.hpBases
    Neval = np.zeros((3,eps.shape[0]),dtype=np.float64)
    for i in range(3,eps.shape[0]):
        Neval[:,i]  = hpBases(0,eps[i,0],eps[i,1],1)[0]

    # from Core.Supplementary.Tensors import makezero
    # np.savetxt("/home/roman/Dropbox/neval.dat",makezero(Neval),delimiter=',')

    nodeperelem = mesh.elements.shape[1]
    renodeperelem = int((C+2.)*(C+3.)/2.)
    left_over_nodes = renodeperelem - nodeperelem

    reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
    reelements[:,:3] = mesh.elements
    iesize = int( C*(C+5.)/2. )
    repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],2),dtype=np.float64)
    repoints[:mesh.points.shape[0],:]=mesh.points[:,:2]

    pshape0, pshape1 = mesh.points.shape[0], mesh.points.shape[1]
    repshape0, repshape1 = repoints.shape[0], repoints.shape[1]

    telements = time()

    xycoord_higher=[]; ParallelTuple1=[]
    if Parallel:
        # GET HIGHER ORDER COORDINATES - PARALLEL
        ParallelTuple1 = parmap.map(ElementLoopTri,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'tri',eps,
            Neval,pool=MP.Pool(processes=nCPU))

    # LOOP OVER ELEMENTS
    maxNode = np.max(reelements)
    for elem in range(0,mesh.elements.shape[0]):

        # GET HIGHER ORDER COORDINATES
        if Parallel:
            xycoord_higher = ParallelTuple1[elem]
        else:
            # xycoord =  mesh.points[mesh.elements[elem,:],:]
            xycoord_higher = Gett.GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],'tri',elem,eps,Neval)
    
        # EXPAND THE ELEMENT CONNECTIVITY
        newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
        # reelements[elem,3:] = np.arange(maxNode+1,maxNode+1+left_over_nodes)  
        reelements[elem,3:] = newElements
        maxNode = newElements[-1]
        
        repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[3:,:]

    telements = time()-telements

    #--------------------------------------------------------------------------------------
    # NOW REMOVE DUPLICATED POINTS
    tnodes = time()
    nnode_linear = mesh.points.shape[0]
    # KEEP ZEROFY ON, OTHERWISE YOU GET STRANGE BEHVAIOUR
    rounded_repoints = repoints[nnode_linear:,:].copy()
    makezero(rounded_repoints) 
    rounded_repoints = np.round(rounded_repoints,decimals=Decimals)
    # flattened_repoints = np.ascontiguousarray(rounded_repoints).view(np.dtype((np.void, 
        # rounded_repoints.dtype.itemsize * rounded_repoints.shape[1])))
    # _, idx_repoints, inv_repoints = np.unique(flattened_repoints,return_index=True,return_inverse=True)
    _, idx_repoints, inv_repoints = unique2d(rounded_repoints,order=False,
        consider_sort=False,return_index=True,return_inverse=True)
    del rounded_repoints

    idx_repoints = np.concatenate((np.arange(nnode_linear),idx_repoints+nnode_linear))
    repoints = repoints[idx_repoints,:]

    unique_reelements, inv_reelements = np.unique(reelements[:,3:],return_inverse=True)
    unique_reelements = unique_reelements[inv_repoints]
    reelements = unique_reelements[inv_reelements]
    reelements = reelements.reshape(mesh.elements.shape[0],renodeperelem-3) 
    reelements = np.concatenate((mesh.elements,reelements),axis=1)

    tnodes = time() - tnodes
    #------------------------------------------------------------------------------------------


    #------------------------------------------------------------------------------------------
    # BUILD EDGES NOW
    tedges = time()
    
    edge_to_elements = mesh.GetElementsWithBoundaryEdgesTri()
    node_arranger = NodeArrangementTri(C)[0]
    reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
    # for i in range(mesh.edges.shape[0]):
        # reedges[i,:] = reelements[edge_to_elements[i,0],node_arranger[edge_to_elements[i,1],:]]

    reedges = reelements[edge_to_elements[:,0][:,None],node_arranger[edge_to_elements[:,1],:]]
    tedges = time()-tedges
    # exit()
    #------------------------------------------------------------------------------------------



    # #------------------------------------------------------------------------------------------
    # # BUILD EDGES NOW
    # tedges = time()

    # reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
    # reedges[:,:2]=mesh.edges

    # for iedge in range(mesh.edges.shape[0]):
    #     # TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
    #     node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repshape1),repshape0-pshape0,axis=0)
    #     node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repshape1),repshape0-pshape0,axis=0)

    #     # FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
    #     L1 = np.linalg.norm(node1-repoints[pshape0:],axis=1)
    #     L2 = np.linalg.norm(node2-repoints[pshape0:],axis=1)
    #     L = np.linalg.norm(node1-node2,axis=1)

    #     j = np.where(np.abs((L1+L2)-L) < tol)[0]
    #     # if j.shape[0]!=0:
    #     if j.shape[0]==reedges.shape[1]-2:    
    #         reedges[iedge,2:] = j+mesh.points.shape[0]
    # reedges = reedges.astype(np.int64)

    # tedges = time()-tedges
    # # ------------------------------------------------------------------------------------------
    
    class nmesh(object):
        # """Construct pMesh"""
        points = repoints
        elements = reelements
        edges = reedges
        faces = np.array([[],[]])
        nnode = repoints.shape[0]
        nelem = reelements.shape[0]
        info = 'tri'

    # print '\npMeshing timing:\n\t\tElement loop 1:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
    #  ' seconds'+'\n\t\tElement loop 2:\t '+str(telements_2)+' seconds\n\t\tEdge loop:\t\t '+str(tedges)+' seconds\n'


    return nmesh
