import numpy as np
from .NodeArrangement import NodeArrangementQuad

def EquallySpacedPoints(ndim=2,C=0):
    """Produce equally spaced points in (ndim-1) dimension, for the boundaries
        of the mesh. For ndim=2 this is the region enclosed in [-1,1] range

        input:              
        C:                  [int] order of polynomial interpolation
        Returns:            [ndarray] array of equally spaced points 

        """

    assert ndim<4

    if ndim==2:
        # FOR 2-DIMENSION
        return np.linspace(-1,1,C+2)[:,None]
    elif ndim==3:
        xs = np.linspace(-1,1,C+2)[:,None]
        x,y = np.meshgrid(xs,xs)
        points = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),axis=1)
        node_aranger = NodeArrangementQuad(C)[2]
        return points[node_aranger,:]
        return


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
