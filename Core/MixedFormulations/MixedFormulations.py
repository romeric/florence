import numpy as np
from Core import Mesh
from Core.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints

class MixedFormulations():

    def __init__(self,mesh,variables_order=(1,0)):
        self.variables_order = variables_order
        self.quadrature_rules = None
        self.median = None


class upFormulation():
    pass


class FiveFieldPenalty(MixedFormulations):

    def __init__(self,mesh):

        self.submeshes = []*4
        # PREPARE SUBMESHES
        if mesh.element_type == "tri":
            self.submeshes[0].elements = mesh.elements[:,:3]
            self.submeshes[0].nelem = mesh.elements.shape[0]
            self.submeshes[0].points = mesh.points[:self.submeshes[0].elements.max()+1,:]
            self.submeshes[0].element_type = "tri"



class NearlyIncompressibleHuWashizu(MixedFormulations):

    def __init__(self, mesh, variables_order=(2,0,0)):

        if mesh.element_type != "tet" and mesh.element_type != "tri":
            raise NotImplementedError( type(self.__name__), "has not been implemented for", mesh.element_type, "elements")

        self.variables_order = variables_order
        # BUILD MESHES FOR THESE ORDRES
        p = mesh.InferPolynomialDegree()
        if p != self.variables_order[0] and self.variables_order[0]!=0:
            mesh.GetHighOrderMesh(C=self.variables_order[0]-1)
        if self.variables_order[2] == 0 or self.variables_order[1]==0 or self.variables_order[0]==0:
            # GET MEDIAN OF THE ELEMENTS FOR CONSTANT VARIABLES
            self.median, self.bases_at_median = mesh.Median

        # GET QUADRATURE RULES
        self.quadrature_rules = [None]*len(self.variables_order)
        QuadratureOpt = 3
        for counter, degree in enumerate(self.variables_order):
            if degree==0:
                degree=1
            self.quadrature_rules[counter] = list(GetBasesAtInegrationPoints(degree-1, 2*degree,
                QuadratureOpt,mesh.element_type))


    def GetLocalResiduals(self):
        pass

    def GetLocalTractions(self):
        pass

    def GetLocalStiffness(self,condensate=True):
        pass


            

            