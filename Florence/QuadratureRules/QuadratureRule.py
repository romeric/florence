import numpy as np
from warnings import warn
from Florence.QuadratureRules import GaussQuadrature
from Florence.QuadratureRules import QuadraturePointsWeightsTet
from Florence.QuadratureRules import QuadraturePointsWeightsTri
from Florence.QuadratureRules import WVQuadraturePointsWeightsHex


class QuadratureRule(object):


    def __init__(self, qtype="gauss", norder=2, mesh_type="tri", optimal=3, flatten=True, evaluate=True):
        """
            input:
                flatten:                [bool] only used for quads and hexes as tensor based
                                        quadrature is not flattened where tabulated values are.
                                        Optimal quadrature points for all element types are in a
                                        flattened representation
        """

        self.qtype = qtype
        self.norder = norder
        self.element_type = mesh_type
        self.points = []
        self.weights = []
        self.flatten = flatten
        # OPTIMAL QUADRATURE POINTS FOR TRIS AND TETS
        self.optimal = optimal

        if evaluate is False:
            return

        if optimal is False or optimal is None:
            self.qtype = None

        z=[]; w=[];

        if mesh_type == "hex":
            if self.optimal==4:
                zw = WVQuadraturePointsWeightsHex.WVQuadraturePointsWeightsHex(self.norder)
                z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
            else:
                z, w = GaussQuadrature(self.norder,-1.,1.)
        elif mesh_type == "quad":
            z, w = GaussQuadrature(self.norder,-1.,1.)
        elif mesh_type == "tet":
            zw = QuadraturePointsWeightsTet.QuadraturePointsWeightsTet(self.norder,self.optimal)
            z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
        elif mesh_type == "tri":
            zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(self.norder,self.optimal)
            z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
        elif mesh_type == "line":
            z, w = GaussQuadrature(self.norder,-1.,1.)

        self.points = z
        self.weights = w

        if mesh_type == "quad" or mesh_type == "hex":
            if z.ravel().shape[0] == w.ravel().shape[0]:
                self.Flatten(mesh_type=mesh_type)


    def Flatten(self, mesh_type=None):
        """Flateen a quadrature rule given its tensor product form
        """

        if mesh_type == "quad":

            w = np.zeros((int(self.points.shape[0]**2)))
            z = np.zeros((int(self.points.shape[0]**2),2))

            counter = 0
            for i in range(self.points.shape[0]):
                for j in range(self.points.shape[0]):
                    w[counter] = self.weights[i]*self.weights[j]

                    z[counter,0] = self.points[i]
                    z[counter,1] = self.points[j]
                    counter += 1

        elif mesh_type == "hex":

            w = np.zeros((int(self.points.shape[0]**3)))
            z = np.zeros((int(self.points.shape[0]**3),3))

            counter = 0
            for i in range(self.points.shape[0]):
                for j in range(self.points.shape[0]):
                    for k in range(self.points.shape[0]):
                        w[counter] = self.weights[i]*self.weights[j]*self.weights[k]

                        z[counter,0] = self.points[i]
                        z[counter,1] = self.points[j]
                        z[counter,2] = self.points[k]
                        counter += 1

        else:
            raise ValueError("Element type not understood")

        self.points = z
        self.weights = w


    def GetRule(self):
        return self.__dict__

    def SetRule(self, in_dict):
        return self.__dict__.update(in_dict)
