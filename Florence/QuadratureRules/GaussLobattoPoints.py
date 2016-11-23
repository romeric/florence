import numpy as np
from .NumericIntegrator import GaussLobattoQuadrature
from .NodeArrangement import NodeArrangementQuad

def GaussLobattoPoints1D(C):
    return GaussLobattoQuadrature(C+2)[0]


def GaussLobattoPointsQuad(C):

    xs = GaussLobattoQuadrature(C+2)[0]
    x,y = np.meshgrid(xs,xs)
    points = np.concatenate((x.flatten()[:,None],y.flatten()[:,None]),axis=1)
    # points = np.concatenate((y.flatten()[:,None],x.flatten()[:,None]),axis=1)
    node_aranger = NodeArrangementQuad(C)[2]
    # print points
    # print points[node_aranger,:]
    # counter = 0
    # points = np.zeros_like(points)
    # for i in range(xs.shape[0]):
    #     for j in range(xs.shape[0]):
    #         points[counter,0] = xs[j]
    #         points[counter,1] = xs[i]
    #         counter += 1
    # # print points
    # print points[node_aranger,:]
    # exit()
    return points[node_aranger,:]


def GaussLobattoPointsHex(C):

    xs = GaussLobattoQuadrature(C+2)[0]
    x,y,z = np.meshgrid(xs,xs,xs)
    points = np.concatenate((x.flatten()[:,None],y.flatten()[:,None],z.flatten()[:,None]),axis=1)

    node_aranger = NodeArrangementHex(C)[2]
    return points[node_aranger,:]