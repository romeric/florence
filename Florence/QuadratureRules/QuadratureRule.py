import numpy as np
from Florence.QuadratureRules import GaussQuadrature, QuadraturePointsWeightsTet, QuadraturePointsWeightsTri
from .WitherdenQuadraturePoints import WitherdenQuadraturePointsHex

class QuadratureRule(object):

    def __init__(self, qtype="gauss", norder=2, mesh_type="tri", optimal=3, is_flattened=False):
        """
            input:
                is_flattened:           [bool] only used for quads and hexes as tensor based 
                                        quadrature is not flattened where tabulated values are
        """

        self.qtype = qtype
        self.norder = norder
        # OPTIMAL QUADRATURE POINTS FOR TRIS AND TETS
        self.optimal = optimal

        if optimal is False or optimal is None:
            self.qtype = None


        ndim = 2
        if mesh_type == 'tet' or mesh_type == 'hex':
            ndim = 3


        z=[]; w=[]; 

        if mesh_type == "quad" or mesh_type == "hex":
            if is_flattened:
                zw = WitherdenQuadraturePointsHex(norder)
                z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
            else:
                z, w = GaussQuadrature(norder,-1.,1.)
        elif mesh_type == "tet":
            zw = QuadraturePointsWeightsTet.QuadraturePointsWeightsTet(norder,optimal)
            z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
        elif mesh_type == "tri":
            zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(norder,optimal)
            z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]

        self.points = z
        self.weights = w 