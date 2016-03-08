import numpy as np
from Florence import QuadratureRule, FunctionSpace, Mesh

class VariationalPrinciple():

    def __init__(self, mesh, variables_order=(1,0), 
        analysis_type='static', analysis_nature='nonlinear',
        quadrature_rules=None, median=None, prestress_required=True,
        requires_geometry_update=True):

        self.variables_order = variables_order
        self.quadrature_rules = quadrature_rules
        self.median = median
        self.analysis_type = analysis_type
        self.analysis_nature = analysis_nature
        self.requires_prestress = prestress_required
        self.requires_geometry_update = requires_geometry_update 



class upFormulation():
    pass


class FiveFieldPenalty(VariationalPrinciple):

    def __init__(self,mesh):

        self.submeshes = []*4
        # PREPARE SUBMESHES
        if mesh.element_type == "tri":
            self.submeshes[0].elements = mesh.elements[:,:3]
            self.submeshes[0].nelem = mesh.elements.shape[0]
            self.submeshes[0].points = mesh.points[:self.submeshes[0].elements.max()+1,:]
            self.submeshes[0].element_type = "tri"


            

            