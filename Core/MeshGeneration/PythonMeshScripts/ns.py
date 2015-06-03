import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def readmesh(fname):
    """
    input
    -----
        fname: string
            gmsh file name

    output
    ------
        V: array
            vertices
        E: array
            element ids
    """
    import gmsh
    mesh = gmsh.Mesh()
    mesh.read_msh(fname)
    return mesh.Verts[:, :2], mesh.Elmts[2][1]


def identify_boundary(V):
    """
    input
    -----
        V: array
            vertices

    output
    ------
        d: dictionary
            inflow, outflow, wall, cylinder
            (unsorted)
    """
    d = {}

    I = np.where(np.abs(V[:, 0]) < 1e-13)
    d['inflow'] = I

    I = np.where(np.abs(V[:, 0] - 22.0) < 1e-13)
    d['outflow'] = I

    I = np.where(np.abs(V[:, 1]) < 1e-13)
    J = np.where(np.abs(V[:, 1] - 4.1) < 1e-13)
    d['wall'] = np.vstack((I, J)).ravel()

    I = np.where(np.abs(2 *
                        np.sqrt((V[:, 0] - 2.0)**2 + (V[:, 1] - 2.0)**2)
                        - 1.0) < 1e-13)
    d['cylinder'] = I

    return d


def local_matrices():
    """
    create some local matrices
    """
    pass

def assemble_fem()
    """
    assemble a LS FEM
    """
    pass

if __name__ == '__main__':

    V, E = readmesh('ns.msh')
    d = identify_boundary(V)

    plt.ion()
    plt.triplot(V[:, 0], V[:, 1], E)
    plt.axis('scaled')

    I = d['inflow']
    plt.plot(V[I, 0], V[I, 1], 'ro', markersize=10)

    I = d['outflow']
    plt.plot(V[I, 0], V[I, 1], 'bo', markersize=10)

    I = d['wall']
    plt.plot(V[I, 0], V[I, 1], 'gs', markersize=10)

    I = d['cylinder']
    plt.plot(V[I, 0], V[I, 1], 'm*', markersize=10)
    plt.show()
