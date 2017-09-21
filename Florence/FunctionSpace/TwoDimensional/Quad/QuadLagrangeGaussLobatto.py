import numpy as np
from Florence.FunctionSpace.OneDimensional import Line as OneD
from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad


def LagrangeGaussLobatto(C,zeta,eta,arrange=1):
    """Computes stable higher order Lagrangian bases with Gauss-Lobatto-Legendre points
        Refer to: Spencer's Spectral hp elements for details
    """

    Bases = np.zeros(((C+2)**2,1))
    Neta = np.zeros((C+2,1));   Nzeta = np.zeros((C+2,1))
    # Compute each from one-dimensional bases
    Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
    Neta[:,0] =  OneD.LagrangeGaussLobatto(C,eta)[0]
    # Ternsorial product
    if arrange==1:
        # # Arrange in counterclockwise - THIS FUNCTION NEEDS TO BE OPTIMISED
        # zeta_index, eta_index = GetCounterClockwiseIndices(C)
        # TBases = np.dot(Nzeta,Neta.T)
        # for i in range(0,(C+2)**2):
            # Bases[i] = TBases[zeta_index[i],eta_index[i]]

        node_arranger = NodeArrangementQuad(C)[2]
        # Bases = np.dot(Nzeta,Neta.T).flatten()
        Bases = np.dot(Neta,Nzeta.T).flatten()
        Bases = Bases[node_arranger]
        Bases = Bases[:,None]

    elif arrange==0:
        # Bases[:,0] = np.dot(Nzeta,Neta.T).flatten()
        Bases[:,0] = np.dot(Neta,Nzeta.T).flatten()

    # check = np.array([
        # 0.25*(1-zeta)*(1-eta),
        # 0.25*(1+zeta)*(1-eta),
        # 0.25*(1+zeta)*(1+eta),
        # 0.25*(1-zeta)*(1+eta)
        # ])
    # print check


    return Bases

def GradLagrangeGaussLobatto(C,zeta,eta,arrange=1):
    """Computes gradients of stable higher order Lagrangian bases with Gauss-Lobatto-Legendre points
        Refer to: Spencer's Spectral hp elements for details
    """

    gBases = np.zeros(((C+2)**2,2))
    Nzeta = np.zeros((C+2,1));  Neta = np.zeros((C+2,1))
    gNzeta = np.zeros((C+2,1)); gNeta = np.zeros((C+2,1))
    # Compute each from one-dimensional bases
    Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
    Neta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[0]
    gNzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[1]
    gNeta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[1]
    # Ternsorial product
    if arrange==1:

        # # Arrange counterclockwise
        # zeta_index, eta_index = GetCounterClockwiseIndices(C)
        # gTBases0 = np.dot(gNzeta,Neta.T)
        # gTBases1 = np.dot(Nzeta,gNeta.T)

        # for i in range(0,(C+2)**2):
        #     gBases[i,0] = gTBases0[zeta_index[i],eta_index[i]]
        #     gBases[i,1] = gTBases1[zeta_index[i],eta_index[i]]

        node_arranger = NodeArrangementQuad(C)[2]
        # g0 = np.dot(gNzeta,Neta.T).flatten()
        # g1 = np.dot(Nzeta,gNeta.T).flatten()
        g0 = np.dot(Nzeta,gNeta.T).flatten()
        g1 = np.dot(gNzeta,Neta.T).flatten()
        # g0 = np.dot(gNeta,Nzeta.T).flatten()
        # g1 = np.dot(Neta,gNzeta.T).flatten()
        gBases[:,0] = g0[node_arranger]
        gBases[:,1] = g1[node_arranger]

    elif arrange==0:
        # gBases[:,0] = np.dot(gNzeta,Neta.T).reshape((C+2)**2)
        # gBases[:,1] = np.dot(Nzeta,gNeta.T).reshape((C+2)**2)
        gBases[:,0] = np.dot(gNeta,Nzeta.T).reshape((C+2)**2)
        gBases[:,1] = np.dot(Neta,gNzeta.T).reshape((C+2)**2)


    # check =  0.25*np.array([[eta-1.,1-eta,1+eta,-1.-eta],[zeta-1.,-zeta-1.,1+zeta,1-zeta]])


    return gBases