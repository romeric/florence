import numpy as np
from time import time
from warnings import warn
from .GetInteriorCoordinates import GetInteriorNodesCoordinates
from Florence.Tensor import itemfreq, makezero, unique2d, remove_duplicates_2D


#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

def HighOrderMeshLine(C, mesh, Decimals=10, equally_spaced=False, check_duplicates=True,
    Parallel=False, nCPU=1):

    from Florence.FunctionSpace import Line
    from Florence.QuadratureRules import GaussLobattoPoints1D
    from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementLine

    # ARRANGE NODES FOR LINES HERE (DONE ONLY FOR LINES) - IMPORTANT
    node_aranger = NodeArrangementLine(C)

    if not equally_spaced:
        eps = GaussLobattoPoints1D(C).ravel()
        eps = eps[node_aranger]
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((2,eps.shape[0]),dtype=np.float64)
        for i in range(0,eps.shape[0]):
            Neval[:,i] = Line.LagrangeGaussLobatto(0,eps[i])[0]
    else:
        eps = EquallySpacedPoints(2,C).ravel()
        eps = eps[node_aranger]
        # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
        Neval = np.zeros((2,eps.shape[0]),dtype=np.float64)
        for i in range(0,eps.shape[0]):
            Neval[:,i] = Line.Lagrange(0,eps[i])[0]
    makezero(Neval)


    nodeperelem = mesh.elements.shape[1]
    renodeperelem = int(C+2)
    left_over_nodes = renodeperelem - nodeperelem

    reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
    reelements[:,:2] = mesh.elements
    iesize = int(C)
    repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],mesh.points.shape[1]),dtype=np.float64)
    repoints[:mesh.points.shape[0],:]=mesh.points

    #--------------------------------------------------------------------------------------
    telements = time()

    xycoord_higher=[]; ParallelTuple1=[]
    if Parallel:
        # GET HIGHER ORDER COORDINATES - PARALLEL
        ParallelTuple1 = parmap.map(ElementLoopTri,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'line',eps,
            Neval,pool=MP.Pool(processes=nCPU))

    # LOOP OVER ELEMENTS
    maxNode = np.max(reelements)
    for elem in range(0,mesh.elements.shape[0]):

        # GET HIGHER ORDER COORDINATES
        if Parallel:
            xycoord_higher = ParallelTuple1[elem]
        else:
            xycoord_higher = GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],'line',elem,eps,Neval)

        # EXPAND THE ELEMENT CONNECTIVITY
        newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes)
        # reelements[elem,3:] = np.arange(maxNode+1,maxNode+1+left_over_nodes)
        reelements[elem,2:] = newElements
        maxNode = newElements[-1]

        repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[2:,:]


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

    unique_reelements, inv_reelements = np.unique(reelements[:,2:],return_inverse=True)
    unique_reelements = unique_reelements[inv_repoints]
    reelements = unique_reelements[inv_reelements]
    reelements = reelements.reshape(mesh.elements.shape[0],renodeperelem-2)
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


    class nmesh(object):
        points = repoints
        elements = reelements
        edges = []
        faces = []
        nnode = repoints.shape[0]
        nelem = reelements.shape[0]
        info = 'line'

    # MESH CORNERS REMAIN THE SAME FOR ALL POLYNOMIAL DEGREES
    if isinstance(mesh.corners,np.ndarray):
        nmesh.corners = mesh.corners

    return nmesh