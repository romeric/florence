import numpy as np
from copy import deepcopy
from warnings import warn
import gc
from time import time
import multiprocessing as MP

from Florence.Tensor import itemfreq, makezero, unique2d, remove_duplicates_2D
import Florence.ParallelProcessing.parmap as parmap
from .GetInteriorCoordinates import GetInteriorNodesCoordinates

#--------------------------------------------------------------------------------------------------------------------------#
# SUPPLEMENTARY FUNCTIONS
def ElementLoopHex(elem,elements,points,MeshType,eps,Neval):
    xycoord_higher = GetInteriorNodesCoordinates(points[elements[elem,:],:],MeshType,elem,eps,Neval)
    return xycoord_higher


def HighOrderMeshHex(C, mesh, Decimals=10, equally_spaced=False, check_duplicates=True,
    Zerofy=True, Parallel=False, nCPU=1, ComputeAll=True):

    from Florence.FunctionSpace import Hex, HexES
    from Florence.QuadratureRules import GaussLobattoPointsHex
    from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementHex


    # SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
    if (mesh.elements.shape[0] < 500) and (C < 5):
        Parallel = False
        nCPU = 1

    if not equally_spaced:
        eps = GaussLobattoPointsHex(C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((8,eps.shape[0]),dtype=np.float64)
        hpBases = Hex.LagrangeGaussLobatto
        for i in range(8,eps.shape[0]):
            Neval[:,i] = hpBases(0,eps[i,0],eps[i,1],eps[i,2])[:,0]
    else:
        eps = EquallySpacedPoints(4,C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((8,eps.shape[0]),dtype=np.float64)
        hpBases = HexES.Lagrange
        for i in range(8,eps.shape[0]):
            Neval[:,i] = hpBases(0,eps[i,0],eps[i,1],eps[i,2])[:,0]

    # THIS IS NECESSARY FOR REMOVING DUPLICATES
    makezero(Neval, tol=1e-12)

    nodeperelem = mesh.elements.shape[1]
    renodeperelem = int((C+2)**3)
    left_over_nodes = renodeperelem - nodeperelem

    reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
    reelements[:,:8] = mesh.elements
    # TOTAL NUMBER OF (INTERIOR+EDGE+FACE) NODES
    iesize = renodeperelem - 8
    repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],3),dtype=np.float64)
    repoints[:mesh.points.shape[0],:]=mesh.points


    telements = time()

    xycoord_higher=[]; ParallelTuple1=[]
    if Parallel:
        # GET HIGHER ORDER COORDINATES - PARALLEL
        ParallelTuple1 = parmap.map(ElementLoopTet,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'hex',eps,
            Neval,pool=MP.Pool(processes=nCPU))

    maxNode = np.max(reelements)
    for elem in range(0,mesh.elements.shape[0]):
        # maxNode = np.max(reelements) # BIG BOTTLENECK
        if Parallel:
            xycoord_higher = ParallelTuple1[elem]
        else:
            xycoord =  mesh.points[mesh.elements[elem,:],:]
            # GET HIGHER ORDER COORDINATES
            xycoord_higher = GetInteriorNodesCoordinates(xycoord,'hex',elem,eps,Neval)

        # EXPAND THE ELEMENT CONNECTIVITY
        newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes)
        reelements[elem,8:] = newElements
        # INSTEAD COMPUTE maxNode BY INDEXING
        maxNode = newElements[-1]

        repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[8:,:]

    if Parallel:
        del ParallelTuple1

    telements = time()-telements
    #--------------------------------------------------------------------------------------
    # NOW REMOVE DUPLICATED POINTS
    tnodes = time()

    nnode_linear = mesh.points.shape[0]
    # KEEP ZEROFY ON, OTHERWISE YOU GET STRANGE BEHVAIOUR
    # rounded_repoints = makezero(repoints[nnode_linear:,:].copy())
    rounded_repoints = repoints[nnode_linear:,:].copy()
    makezero(rounded_repoints)
    rounded_repoints = np.round(rounded_repoints,decimals=Decimals)

    _, idx_repoints, inv_repoints = unique2d(rounded_repoints,order=False,
        consider_sort=False,return_index=True,return_inverse=True)
    # idx_repoints.sort()
    del rounded_repoints#, flattened_repoints

    idx_repoints = np.concatenate((np.arange(nnode_linear),idx_repoints+nnode_linear))
    repoints = repoints[idx_repoints,:]

    unique_reelements, inv_reelements = np.unique(reelements[:,8:],return_inverse=True)
    unique_reelements = unique_reelements[inv_repoints]
    reelements = unique_reelements[inv_reelements]
    reelements = reelements.reshape(mesh.elements.shape[0],renodeperelem-8)
    reelements = np.concatenate((mesh.elements,reelements),axis=1)

    # SANITY CHECK fOR DUPLICATES
    #---------------------------------------------------------------------#
    # NOTE THAT THIS REMAPS THE ELEMENT CONNECTIVITY FOR THE WHOLE MESH
    # AND AS A RESULT THE FIRST FEW COLUMNS WOULD NO LONGER CORRESPOND TO
    # LINEAR CONNECTIVITY
    if check_duplicates:
        last_shape = repoints.shape[0]
        deci = int(Decimals)-2
        if Decimals < 6:
            deci = Decimals
        repoints, idx_repoints, inv_repoints = remove_duplicates_2D(repoints, decimals=deci)
        unique_reelements, inv_reelements = np.unique(reelements,return_inverse=True)
        unique_reelements = unique_reelements[inv_repoints]
        reelements = unique_reelements[inv_reelements]
        reelements = reelements.reshape(mesh.elements.shape[0],renodeperelem)
        if last_shape != repoints.shape[0]:
            warn('Duplicated points generated in high order mesh. Lower the "Decimals". I have fixed it for now')
    #---------------------------------------------------------------------#

    tnodes = time() - tnodes
    #------------------------------------------------------------------------------------------


    # USE ALTERNATIVE APPROACH TO GET MESH EDGES AND FACES
    reedges = np.zeros((mesh.edges.shape[0],C+2))
    fsize = int((C+2.)*(C+3.)/2.)
    refaces = np.zeros((mesh.faces.shape[0],fsize),dtype=mesh.faces.dtype)


    if ComputeAll == True:
        #------------------------------------------------------------------------------------------
        # BUILD FACES NOW
        tfaces = time()

        refaces = np.zeros((mesh.faces.shape[0],fsize))
        # DO NOT CHANGE THE FACES, BY RECOMPUTING THEM, AS THE LINEAR FACES CAN COME FROM
        # AN EXTERNAL MESH GENERATOR, WHOSE ORDERING MAY NOT BE THE SAME, SO JUST FIND WHICH
        # ELEMENTS CONTAIN THESE FACES
        face_to_elements = mesh.GetElementsWithBoundaryFacesHex()
        node_arranger = NodeArrangementHex(C)[0]

        refaces = reelements[face_to_elements[:,0][:,None],node_arranger[face_to_elements[:,1],:]].astype(mesh.faces.dtype)

        tfaces = time()-tfaces
        #------------------------------------------------------------------------------------------

        #------------------------------------------------------------------------------------------
        # BUILD EDGES NOW
        tedges = time()

        # BUILD A 2D MESH
        from Florence import Mesh
        tmesh = Mesh()
        tmesh.element_type = "quad"
        tmesh.elements = refaces
        tmesh.nelem = tmesh.elements.shape[0]
        # GET BOUNDARY EDGES
        reedges = tmesh.GetEdgesQuad()
        del tmesh

        tedges = time()-tedges
        #------------------------------------------------------------------------------------------

    class nmesh(object):
        # """Construct pMesh"""
        points = repoints
        elements = reelements
        edges = np.array([[],[]])
        faces = np.array([[],[]])
        nnode = repoints.shape[0]
        nelem = reelements.shape[0]
        info = 'hex'
    if ComputeAll is True:
        nmesh.edges = reedges
        nmesh.faces = refaces

    gc.collect()


    # print '\nHigh order meshing timing:\n\t\tElement loop:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
    #  ' seconds'+'\n\t\tEdge loop:\t\t '+str(tedges)+' seconds'+\
    #  '\n\t\tFace loop:\t\t '+str(tfaces)+' seconds\n'

    return nmesh