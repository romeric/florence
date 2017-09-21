from time import time
import numpy as np
from warnings import warn
import multiprocessing as MP
import imp

from .GetInteriorCoordinates import GetInteriorNodesCoordinates
import Florence.ParallelProcessing.parmap as parmap
from Florence.Tensor import itemfreq, makezero, unique2d, remove_duplicates_2D

#--------------------------------------------------------------------------------------------------------------------------#
# SUPPLEMENTARY FUNCTIONS
def ElementLoopTri(elem,elements,points,MeshType,eps,Neval):
    xycoord_higher = GetInteriorNodesCoordinates(points[elements[elem,:],:],MeshType,elem,eps,Neval)
    return xycoord_higher

def HighOrderMeshTri_SEMISTABLE(C, mesh, Decimals=10, equally_spaced=False, check_duplicates=True,
    Parallel=False, nCPU=1, ComputeAll=False):

    from Florence.FunctionSpace import Tri
    from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementTri

    # SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
    if (mesh.elements.shape[0] < 1000) and (C < 8):
        Parallel = False
        nCPU = 1

    if not equally_spaced:
        eps =  FeketePointsTri(C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        hpBases = Tri.hpNodal.hpBases
        Neval = np.zeros((3,eps.shape[0]),dtype=np.float64)
        for i in range(3,eps.shape[0]):
            Neval[:,i]  = hpBases(0,eps[i,0],eps[i,1],1)[0]
    else:
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        eps =  EquallySpacedPointsTri(C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        hpBases = Tri.hpNodal.hpBases
        Neval = np.zeros((3,eps.shape[0]),dtype=np.float64)
        for i in range(3,eps.shape[0]):
            Neval[:,i]  = hpBases(0,eps[i,0],eps[i,1],Transform=1,EvalOpt=1,equally_spaced=True)[0]

    # THIS IS NECESSARY FOR REMOVING DUPLICATES
    makezero(Neval, tol=1e-12)

    nodeperelem = mesh.elements.shape[1]
    renodeperelem = int((C+2.)*(C+3.)/2.)
    left_over_nodes = renodeperelem - nodeperelem

    reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
    reelements[:,:3] = mesh.elements
    iesize = int( C*(C+5.)/2. )
    repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],mesh.points.shape[1]),dtype=np.float64)
    # repoints[:mesh.points.shape[0],:]=mesh.points[:,:2]
    repoints[:mesh.points.shape[0],:]=mesh.points

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
            xycoord_higher = GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],'tri',elem,eps,Neval)

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

    # SANITY CHECK FOR DUPLICATES
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


    # BUILD EDGES NOW
    #------------------------------------------------------------------------------------------
    tedges = time()

    edge_to_elements = mesh.GetElementsWithBoundaryEdgesTri()
    node_arranger = NodeArrangementTri(C)[0]
    reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
    # for i in range(mesh.edges.shape[0]):
        # reedges[i,:] = reelements[edge_to_elements[i,0],node_arranger[edge_to_elements[i,1],:]]
    reedges = reelements[edge_to_elements[:,0][:,None],node_arranger[edge_to_elements[:,1],:]]

    tedges = time()-tedges
    #------------------------------------------------------------------------------------------

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
