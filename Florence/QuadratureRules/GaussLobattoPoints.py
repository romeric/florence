import numpy as np
from .NumericIntegrator import GaussLobattoQuadrature
from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad, NodeArrangementHex

def GaussLobattoPoints1D(C):
    return GaussLobattoQuadrature(C+2)[0]


def GaussLobattoPointsQuad(C):

    xs = GaussLobattoQuadrature(C+2)[0]
    x,y = np.meshgrid(xs,xs)
    points = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),axis=1)
    node_aranger = NodeArrangementQuad(C)[2]
    return points[node_aranger,:]


def GaussLobattoPointsHex(C):

    xs = GaussLobattoQuadrature(C+2)[0]
    x,y,z = np.meshgrid(xs,xs,xs)
    points = np.concatenate((y.T.flatten()[:,None],x.T.flatten()[:,None],z.T.flatten()[:,None]),axis=1)

     # p = 1
     # [-1. -1. -1.]
     # [ 1. -1. -1.]
     # [ 1.  1. -1.]
     # [-1.  1. -1.]
     # [-1. -1.  1.]
     # [ 1. -1.  1.]
     # [ 1.  1.  1.]
     # [-1.  1.  1.]

    node_aranger = NodeArrangementHex(C)[2]
    return points[node_aranger,:]


