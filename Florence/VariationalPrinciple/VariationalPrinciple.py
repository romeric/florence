import numpy as np
from Florence import Mesh
from Florence.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints

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


            

            