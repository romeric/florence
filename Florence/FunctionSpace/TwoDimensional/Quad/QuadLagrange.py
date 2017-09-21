import numpy as np
from Florence.FunctionSpace.OneDimensional import Line as OneD
from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad


def Lagrange(C,zeta,eta,arrange=1):
    """Computes higher order Lagrangian bases with equally spaced points
    """

    # Allocate
    Bases = np.zeros(((C+2)**2,1))
    Neta = np.zeros((C+2,1));   Nzeta = np.zeros((C+2,1))
    # Compute each from one-dimensional bases
    Nzeta[:,0] = OneD.Lagrange(C,zeta)[0]
    Neta[:,0] =  OneD.Lagrange(C,eta)[0]
    # Ternsorial product
    if arrange==1:
        node_arranger = NodeArrangementQuad(C)[2]
        Bases = np.dot(Neta,Nzeta.T).flatten()
        Bases = Bases[node_arranger]
        Bases = Bases[:,None]
    elif arrange==0:
        Bases[:,0] = np.dot(Neta,Nzeta.T).flatten()

    return Bases

def GradLagrange(C,zeta,eta,arrange=1):
    """Computes gradient of higher order Lagrangian bases with equally spaced points
    """

    # Allocate
    gBases = np.zeros(((C+2)**2,2))
    Nzeta = np.zeros((C+2,1));  Neta = np.zeros((C+2,1))
    gNzeta = np.zeros((C+2,1)); gNeta = np.zeros((C+2,1))
    # Compute each from one-dimensional bases
    Nzeta[:,0] = OneD.Lagrange(C,zeta)[0]
    Neta[:,0] = OneD.Lagrange(C,eta)[0]
    gNzeta[:,0] = OneD.Lagrange(C,zeta)[1]
    gNeta[:,0] = OneD.Lagrange(C,eta)[1]
    # Ternsorial product
    if arrange==1:
        node_arranger = NodeArrangementQuad(C)[2]
        # g0 = np.dot(gNeta,Nzeta.T).flatten()
        # g1 = np.dot(Neta,gNzeta.T).flatten()
        g0 = np.dot(Nzeta,gNeta.T).flatten()
        g1 = np.dot(gNzeta,Neta.T).flatten()
        gBases[:,0] = g0[node_arranger]
        gBases[:,1] = g1[node_arranger]
    elif arrange==0:
        gBases[:,0] = np.dot(gNeta,Nzeta.T).reshape((C+2)**2)
        gBases[:,1] = np.dot(Neta,gNzeta.T).reshape((C+2)**2)

    return gBases

