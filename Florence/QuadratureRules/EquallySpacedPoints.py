import numpy as np
from .NodeArrangement import NodeArrangementQuad, NodeArrangementHex

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
