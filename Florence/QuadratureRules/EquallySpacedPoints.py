import numpy as np
from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad, NodeArrangementHex

def EquallySpacedPoints(ndim=2,C=0):
    """Produce equally spaced points in (ndim-1) dimension, for the boundaries
        of the mesh. For ndim=2 this is the region enclosed in [-1,1] range

        input:
            C:                  [int] order of polynomial interpolation
            Returns:            [ndarray] array of equally spaced points

        """

    if ndim==2:
        # 1D: FOR 2-DIMENSION BOUNDARIES
        return np.linspace(-1,1,C+2)[:,None]
    elif ndim==3:
        # 2D: FOR 3-DIMENSION BOUNDARIES
        xs = np.linspace(-1,1,C+2)[:,None]
        x,y = np.meshgrid(xs,xs)
        points = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),axis=1)
        node_aranger = NodeArrangementQuad(C)[2]
        return points[node_aranger,:]
    elif ndim==4:
        # 3D: ACTUAL 3D ELEMENTS
        xs = np.linspace(-1,1,C+2)[:,None]
        x,y,z = np.meshgrid(xs,xs,xs)
        points = np.concatenate((y.T.flatten()[:,None],x.T.flatten()[:,None],z.T.flatten()[:,None]),axis=1)
        node_aranger = NodeArrangementHex(C)[2]
        return points[node_aranger,:]


def EquallySpacedPointsTri(C):

    h0 = 2./(C+1)

    nodes = np.array([
        [-1.,-1.],
        [1.,-1.],
        [-1.,1.]
        ])

    for i in range(C):
        nodes = np.concatenate((nodes,[[-1.+(i+1.)*h0,-1.]]),axis=0)
    for j in range(C):
        for i in range(C+1-j):
            nodes = np.concatenate((nodes,[[-1.+i*h0,-1.+(j+1.)*h0]]),axis=0)

    return nodes


def EquallySpacedPointsTet(C, coordinates=None):
    """
        coordinates:            [ndarray] vertices of a tetrahedron
    """

    if coordinates is not None:
        xv = coordinates
    else:
        xv = np.array([
            [-1.,-1.,-1.],
            [ 1.,-1.,-1.],
            [-1., 1.,-1.],
            [-1.,-1., 1.],
            ])

    if C==0:
        return xv

    n = C+1
    p = 0

    nsize = int((n+1)*(n+2)*(n+3)/6)
    # xg = np.zeros((nsize,3))

    # for i in range ( 0, n + 1 ):
    #     for j in range ( 0, n + 1 - i ):
    #         for k in range ( 0, n + 1 - i - j ):
    #             l = n - i - j - k
    #             xg[p,0] = (i * xv[0,0] + j * xv[1,0] + k * xv[2,0] + l * xv[3,0]) / n
    #             xg[p,1] = (i * xv[0,1] + j * xv[1,1] + k * xv[2,1] + l * xv[3,1]) / n
    #             xg[p,2] = (i * xv[0,2] + j * xv[1,2] + k * xv[2,2] + l * xv[3,2]) / n
    #             # xg[p] = (i * xv[0] + j * xv[1] + k * xv[2] + l * xv[3]) / n
    #             p = p + 1

    xg = np.empty((nsize,4))
    for i in range ( 0, n + 1 ):
        for j in range ( 0, n + 1 - i ):
            for k in range ( 0, n + 1 - i - j ):
                l = n - i - j - k
                xg[p,0] = i
                xg[p,1] = j
                xg[p,2] = k
                xg[p,3] = l
                p = p + 1

    xg = (xg[:,:,None] * xv).sum(1) / n


    # xv = np.array([
    #     [-1.,-1.,-1.],
    #     [ 1.,-1.,-1.],
    #     [-1., 1.,-1.],
    #     [-1.,-1., 1.],
    #     ])

    # # spanning arrays of a 3d grid according to range(0,n+1)
    # ii,jj,kk = np.ogrid[:n+1,:n+1,:n+1]
    # # indices of the triples which fall inside the original for loop
    # inds = (jj < n+1-ii) & (kk < n+1-ii-jj)
    # # the [i,j,k] indices of the points that fall inside the for loop, in the same order
    # combs = np.vstack(np.where(inds)).T
    # # combs is now an (nsize,3)-shaped array

    # # compute "l" column too
    # lcol = n - combs.sum(axis=1)
    # combs = np.hstack((combs,lcol[:,None]))
    # # combs is now an (nsize,4)-shaped array

    # # all we need to do now is to take the matrix product of combs and xv, divide by n in the end
    # xg = np.matmul(combs,xv)/n


    # Sort accordingly
    xg = np.flipud(xg)
    msize = int((n)*(n+1)*(n+2)/6)
    tsize = int((n)*(n+1)/2)
    ind_vertices = [0,msize,msize+tsize,nsize-1]
    xg_vertices = xg[ind_vertices,:]
    xg_non_vertices = np.delete(xg,ind_vertices,axis=0)
    xg_non_vertices_sort = xg_non_vertices[np.lexsort((xg_non_vertices[:,0],
        xg_non_vertices[:,1],xg_non_vertices[:,2])),:]
    xg_ = np.concatenate((xg_vertices,xg_non_vertices_sort))

    return xg_