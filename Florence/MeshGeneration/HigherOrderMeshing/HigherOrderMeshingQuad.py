import numpy as np
from time import time
from warnings import warn
from .GetInteriorCoordinates import GetInteriorNodesCoordinates
from Florence.Tensor import itemfreq, makezero, unique2d, remove_duplicates_2D


#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

def HighOrderMeshQuad(C, mesh, Decimals=10, equally_spaced=False, check_duplicates=True,
    Parallel=False, nCPU=1):

    from Florence.FunctionSpace import Quad, QuadES
    from Florence.QuadratureRules import GaussLobattoPointsQuad
    from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad

    if not equally_spaced:
        eps = GaussLobattoPointsQuad(C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((4,eps.shape[0]),dtype=np.float64)
        for i in range(0,eps.shape[0]):
            Neval[:,i] = Quad.LagrangeGaussLobatto(0,eps[i,0],eps[i,1],arrange=1)[:,0]
    else:
        eps = EquallySpacedPoints(3,C)
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((4,eps.shape[0]),dtype=np.float64)
        for i in range(0,eps.shape[0]):
            Neval[:,i] = QuadES.Lagrange(0,eps[i,0],eps[i,1],arrange=1)[:,0]
    makezero(Neval)

    nodeperelem = mesh.elements.shape[1]
    renodeperelem = int((C+2)**2)
    left_over_nodes = renodeperelem - nodeperelem

    reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
    reelements[:,:4] = mesh.elements
    iesize = int(4*C + C**2)
    repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],mesh.points.shape[1]),dtype=np.float64)
    repoints[:mesh.points.shape[0],:]=mesh.points


    #--------------------------------------------------------------------------------------
    telements = time()

    xycoord_higher=[]; ParallelTuple1=[]
    if Parallel:
        # GET HIGHER ORDER COORDINATES - PARALLEL
        ParallelTuple1 = parmap.map(ElementLoopTri,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'quad',eps,
            Neval,pool=MP.Pool(processes=nCPU))

    # LOOP OVER ELEMENTS
    maxNode = np.max(reelements)
    for elem in range(0,mesh.elements.shape[0]):

        # GET HIGHER ORDER COORDINATES
        if Parallel:
            xycoord_higher = ParallelTuple1[elem]
        else:
            xycoord_higher = GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],'quad',elem,eps,Neval)

        # EXPAND THE ELEMENT CONNECTIVITY
        newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes)
        # reelements[elem,3:] = np.arange(maxNode+1,maxNode+1+left_over_nodes)
        reelements[elem,4:] = newElements
        maxNode = newElements[-1]

        repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[4:,:]


    telements = time()-telements

    #--------------------------------------------------------------------------------------
    # NOW REMOVE DUPLICATED POINTS
    tnodes = time()
    nnode_linear = mesh.points.shape[0]
    # KEEP ZEROFY ON, OTHERWISE YOU GET STRANGE BEHVAIOUR
    rounded_repoints = repoints[nnode_linear:,:].copy()
    makezero(rounded_repoints)
    rounded_repoints = np.round(rounded_repoints,decimals=Decimals)
    _, idx_repoints, inv_repoints = unique2d(rounded_repoints,order=False,
        consider_sort=False,return_index=True,return_inverse=True)
    del rounded_repoints

    idx_repoints = np.concatenate((np.arange(nnode_linear),idx_repoints+nnode_linear))
    repoints = repoints[idx_repoints,:]

    unique_reelements, inv_reelements = np.unique(reelements[:,4:],return_inverse=True)
    unique_reelements = unique_reelements[inv_repoints]
    reelements = unique_reelements[inv_reelements]
    reelements = reelements.reshape(mesh.elements.shape[0],renodeperelem-4)
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

    #------------------------------------------------------------------------------------------
    # BUILD EDGES NOW
    tedges = time()

    edge_to_elements = mesh.GetElementsWithBoundaryEdgesQuad()
    node_arranger = NodeArrangementQuad(C)[0]
    reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
    reedges = reelements[edge_to_elements[:,0][:,None],node_arranger[edge_to_elements[:,1],:]]

    tedges = time()-tedges
    #------------------------------------------------------------------------------------------



    class nmesh(object):
        points = repoints
        elements = reelements
        edges = reedges
        faces = []
        nnode = repoints.shape[0]
        nelem = reelements.shape[0]
        info = 'quad'

    return nmesh