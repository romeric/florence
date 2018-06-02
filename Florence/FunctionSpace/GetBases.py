import numpy as np

# Modal Bases
# import Florence.FunctionSpace.TwoDimensional.Tri.hpModal as Tri
# import Florence.FunctionSpace.ThreeDimensional.Tet.hpModal as Tet
# Nodal Bases
from Florence.FunctionSpace import Line
from Florence.FunctionSpace import Tri
from Florence.FunctionSpace import Tet
from Florence.FunctionSpace import Quad, QuadES
from Florence.FunctionSpace import Hex, HexES

from Florence.QuadratureRules.FeketePointsTri import *
from Florence.QuadratureRules.FeketePointsTet import *
from Florence.QuadratureRules.GaussLobattoPoints import *
from Florence.QuadratureRules.EquallySpacedPoints import *


def GetBases1D(C, Quadrature, info=None, bases_type="nodal", equally_spaced=False, is_flattened=False):

    w = Quadrature.weights
    z = Quadrature.points

    ns=[]; Basis=[]; gBasis = []
    ns = int(C+2)
    Basis = np.zeros((ns,z.shape[0]),dtype=np.float64)
    gBasis = np.zeros((ns,z.shape[0]),dtype=np.float64)

    if not equally_spaced:
        hpBases = Line.LagrangeGaussLobatto
    else:
        hpBases = Line.Lagrange

    for i in range(0,w.shape[0]):
        ndummy, dummy = hpBases(C,z[i])[:2]
        Basis[:,i] = ndummy[:]
        gBasis[:,i] = dummy[:]


    class Domain(object):
        Bases = Basis
        gBasesx = gBasis
        gBasesy = np.zeros(gBasis.shape)
        gBasesz = np.zeros(gBasis.shape)


    return Domain



def GetBases2D(C, Quadrature, info, bases_type="nodal", equally_spaced=False, is_flattened=False):

    w = Quadrature.weights
    z = Quadrature.points

    ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]
    if info=='tri':
        p=C+1
        ns = int((p+1)*(p+2)/2)
        Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
        gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
        gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)
    elif info=='quad':
        ns = int((C+2)**2)
        if is_flattened is False:
            Basis = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)
            gBasisx = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)
            gBasisy = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)
        else:
            Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
            gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
            gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)

    if info == 'quad':
        if not equally_spaced:
            hpBases = Quad.LagrangeGaussLobatto
            GradhpBases = Quad.GradLagrangeGaussLobatto
        else:
            hpBases = QuadES.Lagrange
            GradhpBases = QuadES.GradLagrange

        if is_flattened is False:
            counter = 0
            for i in range(0,z.shape[0]):
                for j in range(0,z.shape[0]):
                    ndummy = hpBases(C,z[i],z[j])
                    Basis[:,counter] = ndummy[:,0]
                    dummy = GradhpBases(C,z[i],z[j])
                    gBasisx[:,counter] = dummy[:,0]
                    gBasisy[:,counter] = dummy[:,1]
                    counter+=1
        else:
            for i in range(0,w.shape[0]):
                ndummy = hpBases(C,z[i,0],z[i,1])
                dummy = GradhpBases(C,z[i,0],z[i,1])
                Basis[:,i] = ndummy[:,0]
                gBasisx[:,i] = dummy[:,0]
                gBasisy[:,i] = dummy[:,1]

    elif info == 'tri':
        hpBases = Tri.hpNodal.hpBases
        for i in range(0,w.shape[0]):
            # Better convergence for curved meshes when Quadrature.optimal!=0
            ndummy, dummy = hpBases(C,z[i,0],z[i,1], Quadrature.optimal, equally_spaced=equally_spaced)
            # ndummy, dummy = Tri.hpBases(C,z[i,0],z[i,1])
            Basis[:,i] = ndummy
            gBasisx[:,i] = dummy[:,0]
            gBasisy[:,i] = dummy[:,1]



    class Domain(object):
        Bases = Basis
        gBasesx = gBasisx
        gBasesy = gBasisy
        gBasesz = np.zeros(gBasisx.shape)


    return Domain



def GetBases3D(C, Quadrature, info, bases_type="nodal", equally_spaced=False, is_flattened=False):

    ndim = 3

    w = Quadrature.weights
    z = Quadrature.points

    ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]; gBasisz=[]
    if info=='hex':
        ns = int((C+2)**ndim)
        if is_flattened is False:
            Basis = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
            gBasisx = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
            gBasisy = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
            gBasisz = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
        else:
            Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
            gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
            gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)
            gBasisz = np.zeros((ns,w.shape[0]),dtype=np.float64)
    elif info=='tet':
        p=C+1
        ns = int((p+1)*(p+2)*(p+3)/6)
        Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
        gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
        gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)
        gBasisz = np.zeros((ns,w.shape[0]),dtype=np.float64)


    if info=='hex':
        if not equally_spaced:
            hpBases = Hex.LagrangeGaussLobatto
            GradhpBases = Hex.GradLagrangeGaussLobatto
        else:
            hpBases = HexES.Lagrange
            GradhpBases = HexES.GradLagrange

        if is_flattened is False:
            counter = 0
            for i in range(w.shape[0]):
                for j in range(w.shape[0]):
                    for k in range(w.shape[0]):
                        ndummy = hpBases(C,z[i],z[j],z[k])
                        dummy = GradhpBases(C,z[i],z[j],z[k])

                        Basis[:,counter] = ndummy[:,0]
                        gBasisx[:,counter] = dummy[:,0]
                        gBasisy[:,counter] = dummy[:,1]
                        gBasisz[:,counter] = dummy[:,2]
                        counter+=1
        else:
            for i in range(0,w.shape[0]):
                ndummy = hpBases(C,z[i,0],z[i,1],z[i,2])
                dummy = GradhpBases(C,z[i,0],z[i,1],z[i,2])
                Basis[:,i] = ndummy[:,0]
                gBasisx[:,i] = dummy[:,0]
                gBasisy[:,i] = dummy[:,1]
                gBasisz[:,i] = dummy[:,2]


    elif info=='tet':
        hpBases = Tet.hpNodal.hpBases
        for i in range(0,w.shape[0]):
            # Better convergence for curved meshes when Quadrature.optimal!=0
            ndummy, dummy = hpBases(C,z[i,0],z[i,1],z[i,2],Quadrature.optimal, equally_spaced=equally_spaced)
            # ndummy, dummy = Tet.hpBases(C,z[i,0],z[i,1],z[i,2])
            Basis[:,i] = ndummy
            gBasisx[:,i] = dummy[:,0]
            gBasisy[:,i] = dummy[:,1]
            gBasisz[:,i] = dummy[:,2]


    class Domain(object):
        """docstring for Domain"""
        Bases = Basis
        gBasesx = gBasisx
        gBasesy = gBasisy
        gBasesz = gBasisz


    return Domain




def GetBasesAtNodes(C, Quadrature, info, bases_type="nodal", equally_spaced=False):

    ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]; gBasisz=[]
    if info == 'hex' or info == "quad":
        w=2
        if info=="quad": ndim = 2
        elif info=="hex": ndim = 3
        ns = int((C+2)**ndim)
        # GET THE BASES AT NODES INSTEAD OF GAUSS POINTS
        Basis = np.zeros((ns,ns))
        gBasisx = np.zeros((ns,ns))
        gBasisy = np.zeros((ns,ns))
        gBasisz = np.zeros((ns,ns))
    elif info =='tet':
        p=C+1
        ns = int((p+1)*(p+2)*(p+3)/6)
        # GET BASES AT NODES INSTEAD OF GAUSS POINTS
        Basis = np.zeros((ns,ns))
        gBasisx = np.zeros((ns,ns))
        gBasisy = np.zeros((ns,ns))
        gBasisz = np.zeros((ns,ns))
    elif info =='tri':
        p=C+1
        ns = int((p+1)*(p+2)/2)
        # GET BASES AT NODES INSTEAD OF GAUSS POINTS
        Basis = np.zeros((ns,ns))
        gBasisx = np.zeros((ns,ns))
        gBasisy = np.zeros((ns,ns))
    elif info == 'line':
        ns = int(C+2)
        # GET THE BASES AT NODES INSTEAD OF GAUSS POINTS
        Basis = np.zeros((ns,ns))
        gBasisx = np.zeros((ns,ns))


    eps=[]
    if info == 'hex':
        counter = 0
        if not equally_spaced:
            eps = GaussLobattoPointsHex(C)
            hpBases = Hex.LagrangeGaussLobatto
            GradhpBases = Hex.GradLagrangeGaussLobatto
        else:
            eps = EquallySpacedPoints(4,C)
            hpBases = HexES.Lagrange
            GradhpBases = HexES.GradLagrange

        for i in range(0,eps.shape[0]):
            ndummy = hpBases(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)
            Basis[:,counter] = ndummy[:,0]
            dummy = GradhpBases(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)
            gBasisx[:,counter] = dummy[:,0]
            gBasisy[:,counter] = dummy[:,1]
            gBasisz[:,counter] = dummy[:,2]
            counter +=1

    elif info == 'quad':
        if not equally_spaced:
            eps = GaussLobattoPointsQuad(C)
            hpBases = Quad.LagrangeGaussLobatto
            GradhpBases = Quad.GradLagrangeGaussLobatto
        else:
            eps = EquallySpacedPoints(3,C)
            hpBases = QuadES.Lagrange
            GradhpBases = QuadES.GradLagrange

        counter = 0
        for i in range(0,eps.shape[0]):
            ndummy = hpBases(C,eps[i,0],eps[i,1],arrange=1)
            Basis[:,counter] = ndummy[:,0]
            dummy = GradhpBases(C,eps[i,0],eps[i,1],arrange=1)
            gBasisx[:,counter] = dummy[:,0]
            gBasisy[:,counter] = dummy[:,1]
            counter+=1

    elif info == 'tet':
        if not equally_spaced:
            eps = FeketePointsTet(C)
        else:
            eps = EquallySpacedPointsTet(C)
        hpBases = Tet.hpNodal.hpBases

        counter = 0
        for i in range(0,eps.shape[0]):
            ndummy, dummy = hpBases(C,eps[i,0],eps[i,1],eps[i,2],1,1, equally_spaced=equally_spaced)
            ndummy = ndummy.reshape(ndummy.shape[0],1)
            Basis[:,counter] = ndummy[:,0]
            gBasisx[:,counter] = dummy[:,0]
            gBasisy[:,counter] = dummy[:,1]
            gBasisz[:,counter] = dummy[:,2]
            counter+=1

    elif info == 'tri':
        if not equally_spaced:
            eps = FeketePointsTri(C)
        else:
            eps = EquallySpacedPointsTri(C)
        hpBases = Tri.hpNodal.hpBases

        for i in range(0,eps.shape[0]):
            ndummy, dummy = hpBases(C,eps[i,0],eps[i,1],1,1, equally_spaced=equally_spaced)
            ndummy = ndummy.reshape(ndummy.shape[0],1)
            Basis[:,i] = ndummy[:,0]
            gBasisx[:,i] = dummy[:,0]
            gBasisy[:,i] = dummy[:,1]

    elif info == 'line':
        if not equally_spaced:
            eps = GaussLobattoQuadrature(C+2)[0]
            hpBases = Line.LagrangeGaussLobatto
        else:
            eps = EquallySpacedPoints(2,C)
            hpBases = Line.Lagrange

        # We probably need node arrangment for lines
        counter = 0
        for i in range(0,eps.shape[0]):
            ndummy = hpBases(C,eps[i,0])
            Basis[:,counter] = ndummy[0][i]
            gBasisx[:,counter] = ndummy[1][i]
            counter+=1



    class Domain(object):
        Bases = Basis
        gBasesx = gBasisx
        gBasesy = gBasisy
        w = np.ones(eps.shape[0])

    if info == "hex" or info == "tet":
        Domain.gBasesz = gBasisz
    elif info == "tri" or info == "quad":
        Domain.gBasesz = np.zeros_like(gBasisx)
    elif info == "line":
        Domain.gBasesy = np.zeros_like(gBasisx)
        Domain.gBasesz = np.zeros_like(gBasisx)

    if info == "hex" or info == "quad":
        Domain.w = np.ones(C+2)

    return Domain



def GetBoundaryBases(C, Quadrature, info, bases_type="nodal", equally_spaced=False, is_flattened=False):

    binfo, ndim = None, None
    if info == "hex":
        binfo = "quad"
        ndim == 3
    elif info == "tet":
        binfo = "tri"
        ndim == 3
    elif info == "quad" or info == "tri":
        binfo = "line"
        ndim == 2

    if Quadrature is None:
        norder = C+2
        from Florence import QuadratureRule
        Quadrature = QuadratureRule(norder=norder, mesh_type=binfo, is_flattened=is_flattened)


    if ndim == 3:
        return GetBases2D(C, Quadrature, info, bases_type=bases_type, equally_spaced=equally_spaced, is_flattened=is_flattened)
    elif ndim == 2:
        return GetBases1D(C, Quadrature, info, bases_type=bases_type, equally_spaced=equally_spaced, is_flattened=is_flattened)






########################################################
# DEPRECATED
# def GetBasesBoundary(C, z, ndim):

#     BasisBoundary = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
#     gBasisBoundaryx = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
#     gBasisBoundaryy = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
#     gBasisBoundaryz = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))

#     # eps = OneD.LagrangeGaussLobatto(C,0)
#     eps = np.array([-1.,1.,-1.,1.,-1.,1.])


#     for k in range(0,eps.shape[0]):
#         counter = 0
#         for i in range(0,z.shape[0]):
#             for j in range(0,z.shape[0]):
#                 if k==0 or k==1:
#                     ndummy = Hex.LagrangeGaussLobatto(C,eps[k],z[i],z[j])[0]
#                     BasisBoundary[:,counter,k] = ndummy[:,0]

#                     dummy = Hex.GradLagrangeGaussLobatto(C,eps[k],z[i],z[j])
#                     gBasisBoundaryx[:,counter,k] = dummy[:,0]
#                     gBasisBoundaryy[:,counter,k] = dummy[:,1]
#                     gBasisBoundaryz[:,counter,k] = dummy[:,2]

#                 elif k==2 or k==3:
#                     ndummy = Hex.LagrangeGaussLobatto(C,z[i],eps[k],z[j])[0]
#                     BasisBoundary[:,counter,k] = ndummy[:,0]

#                     dummy = ThreeD.GradLagrangeGaussLobatto(C,z[i],eps[k],z[j])
#                     gBasisBoundaryx[:,counter,k] = dummy[:,0]
#                     gBasisBoundaryy[:,counter,k] = dummy[:,1]
#                     gBasisBoundaryz[:,counter,k] = dummy[:,2]

#                 elif k==4 or k==5:
#                     ndummy = Hex.LagrangeGaussLobatto(C,z[i],z[j],eps[k])[0]
#                     BasisBoundary[:,counter,k] = ndummy[:,0]

#                     dummy = Hex.GradLagrangeGaussLobatto(C,z[i],z[j],eps[k])
#                     gBasisBoundaryx[:,counter,k] = dummy[:,0]
#                     gBasisBoundaryy[:,counter,k] = dummy[:,1]
#                     gBasisBoundaryz[:,counter,k] = dummy[:,2]


#                 counter+=1

#     class Boundary(object):
#         """docstring for BasisBoundary"""
#         def __init__(self, arg):
#             super(BasisBoundary, self).__init__()
#             self.arg = arg
#         Basis  = BasisBoundary
#         gBasisx = gBasisBoundaryx
#         gBasisy = gBasisBoundaryy
#         gBasisz = gBasisBoundaryz


#     return Boundary

