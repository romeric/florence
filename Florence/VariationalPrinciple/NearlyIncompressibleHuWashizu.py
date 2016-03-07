import numpy as np
from .VariationalPrinciple import MixedFormulations
from Florence.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints


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

        # print self.quadrature_rules[0][0].Jm
        print self.median


    def GetLocalResiduals(self):
        pass

    def GetLocalTractions(self):
        pass

    def GetLocalStiffness(self,condensate=True):
        pass