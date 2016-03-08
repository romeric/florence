import numpy as np
# from Florence.QuadratureRules import GaussQuadrature, QuadraturePointsWeightsTet, QuadraturePointsWeightsTri
from Florence import QuadratureRule
from Florence.FunctionSpace.GetBases import *

def GetBasesAtInegrationPoints(C,norder,QuadratureOpt,MeshType):
    """Compute interpolation functions at all integration points"""

    ndim = 2
    if MeshType == 'tet' or MeshType == 'hex':
        ndim = 3

    quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=MeshType)
    z = quadrature.points
    w = quadrature.weights


    if MeshType == 'tet' or MeshType == 'hex':
        # GET BASES AT ALL INTEGRATION POINTS (VOLUME)
        Domain = GetBases3D(C,quadrature,MeshType)

    elif MeshType == 'tri' or MeshType == 'quad':
        # Get basis at all integration points (surface)
        Domain = GetBases(C,quadrature,MeshType)
        # GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
        # Boundary = GetBasesBoundary(C,z,ndim)
    Boundary = []

    ############################################################################


    # COMPUTING GRADIENTS AND JACOBIAN A PRIORI FOR ALL INTEGRATION POINTS
    ############################################################################
    Domain.Jm = []; Domain.AllGauss=[]
    if MeshType == 'hex':
        Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim)) 
        Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))    
        counter = 0
        for g1 in range(0,w.shape[0]):
            for g2 in range(0,w.shape[0]): 
                for g3 in range(0,w.shape[0]):
                    # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
                    Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                    Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
                    Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

                    Domain.AllGauss[counter,0] = w[g1]*w[g2]*w[g3]

                    counter +=1

    elif MeshType == 'quad':
        Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim)) 
        Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))    
        counter = 0
        for g1 in range(0,w.shape[0]):
            for g2 in range(0,w.shape[0]): 
                # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
                Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

                Domain.AllGauss[counter,0] = w[g1]*w[g2]
                counter +=1

    elif MeshType == 'tet':
        Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
        Domain.AllGauss = np.zeros((w.shape[0],1))  
        for counter in range(0,w.shape[0]):
            # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
            Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
            Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
            Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

            Domain.AllGauss[counter,0] = w[counter]

    elif MeshType == 'tri':
        Domain.Jm = [];  Domain.AllGauss = []

        Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
        Domain.AllGauss = np.zeros((w.shape[0],1))  
        for counter in range(0,w.shape[0]):
            # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
            Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
            Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

            Domain.AllGauss[counter,0] = w[counter]

    return Domain, Boundary, quadrature