import numpy as np


class FunctionSpace(object):
    """Base class for all interpolation functions for finite element
        and boundary element analyses
    """

    def __init__(self, mesh, quadrature=None, p=1, bases_type="nodal", bases_kind="CG", evaluate_at_nodes=False, equally_spaced=False):
        """ 

            input:
                mesh:                       [Mesh] an instance of class Mesh
                p:                          [int] degree of interpolation bases
                bases_type:                 [str] type of interpolation bases, 
                                            either "nodal" for higher order of 
                                            "modal" for hierarchical bases
                bases_kind:                 [str] kind of interpolation bases,
                                            either "CG" for continuous Galerkin
                                            or "DG" for discontinuous Galerkin 
                equally_spaced:             Only applicable to nodal bases functions. 
                                            Wether the position of nodes in the
                                            isoparametric domain should be equally spaced
                                            or not  
        """
    
        # from Florence import QuadratureRule
        from Florence.FunctionSpace.GetBases import GetBases, GetBases3D, GetBasesBoundary, GetBasesAtNodes
        

        QuadratureOpt=3
        # norder=5 # 2*C for stiffness and 2*(C+1) for mass

        ndim = mesh.InferSpatialDimension()
        C = p - 1
        if mesh.InferPolynomialDegree() - 1 != C:
            raise ValueError("Function space of the polynomial does not match element type")

        if evaluate_at_nodes is False:
            if quadrature is None:
                raise ValueError("Function space requires a quadrature rule")
            # quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=mesh.element_type)
            z = quadrature.points
            w = quadrature.weights

        if evaluate_at_nodes is False:
            if mesh.element_type == "tet" or mesh.element_type == "hex":
                # GET BASES AT ALL INTEGRATION POINTS (VOLUME)
                Domain = GetBases3D(C,quadrature,mesh.element_type,equally_spaced=equally_spaced)
                # GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
                # Boundary = GetBasesBoundary(C,z,ndim)
            elif mesh.element_type == 'tri' or mesh.element_type == 'quad':
                # GET BASES AT ALL INTEGRATION POINTS (AREA)
                Domain = GetBases(C,quadrature,mesh.element_type,equally_spaced=equally_spaced)
                # GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
                # Boundary = GetBasesBoundary(C,z,ndim)
            Boundary = []
        else:
            Domain = GetBasesAtNodes(C,quadrature,mesh.element_type,equally_spaced=equally_spaced)
            Boundary = []
            w = Domain.w


        # COMPUTING GRADIENTS AND JACOBIAN A PRIORI FOR ALL INTEGRATION POINTS
        ############################################################################
        Domain.Jm = []; Domain.AllGauss=[]
        if mesh.element_type == 'hex':
            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim)) 
            Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))    
            counter = 0
            for g1 in range(w.shape[0]):
                for g2 in range(w.shape[0]): 
                    for g3 in range(w.shape[0]):
                        # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
                        Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                        Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
                        Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

                        Domain.AllGauss[counter,0] = w[g1]*w[g2]*w[g3]

                        counter +=1

        elif mesh.element_type == 'quad':
            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim)) 
            Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))    
            counter = 0
            for g1 in range(w.shape[0]):
                for g2 in range(w.shape[0]): 
                    # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
                    Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                    Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

                    Domain.AllGauss[counter,0] = w[g1]*w[g2]
                    counter +=1

        elif mesh.element_type == 'tet':
            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
            Domain.AllGauss = np.zeros((w.shape[0],1))  
            for counter in range(0,w.shape[0]):
                # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
                Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
                Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

                Domain.AllGauss[counter,0] = w[counter]

        elif mesh.element_type == 'tri':
            Domain.Jm = [];  Domain.AllGauss = []

            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
            Domain.AllGauss = np.zeros((w.shape[0],1))  
            for counter in range(0,w.shape[0]):
                # GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
                Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

                Domain.AllGauss[counter,0] = w[counter]


        self.Jm = Domain.Jm
        self.AllGauss = Domain.AllGauss
        self.Bases = Domain.Bases
        self.gBasesx = Domain.gBasesx 
        self.gBasesy = Domain.gBasesy
        self.gBasesz = Domain.gBasesz

        self.Boundary = Boundary

