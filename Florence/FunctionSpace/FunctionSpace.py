import numpy as np


class FunctionSpace(object):
    """Base class for all interpolation functions for finite element
        and boundary element analyses
    """

    def __init__(self, mesh, quadrature, p=1, bases_type="nodal", bases_kind="CG"):
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
        """
    
        # from Florence import QuadratureRule
        from Florence.FunctionSpace.GetBases import GetBases, GetBases3D, GetBasesBoundary, GetBasesAtNodes
        

        QuadratureOpt=3
        norder=5

        ndim = mesh.InferSpatialDimension()
        C = p - 1
        if mesh.InferPolynomialDegree() - 1 != C:
            raise ValueError("Function space of the polynomial does not match element type")

        # quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=mesh.element_type)
        z = quadrature.points
        w = quadrature.weights

        if mesh.element_type == "tet" or mesh.element_type == "hex":
            # GET BASES AT ALL INTEGRATION POINTS (VOLUME)
            Domain = GetBases3D(C,quadrature,mesh.element_type)
            # GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
            # Boundary = GetBasesBoundary(C,z,ndim)

        elif mesh.element_type == 'tri' or mesh.element_type == 'quad':
            # GET BASES AT ALL INTEGRATION POINTS (AREA)
            Domain = GetBases(C,quadrature,mesh.element_type)
            # GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
            # Boundary = GetBasesBoundary(C,z,ndim)
        Boundary = []


        # COMPUTING GRADIENTS AND JACOBIAN A PRIORI FOR ALL INTEGRATION POINTS
        ############################################################################
        Domain.Jm = []; Domain.AllGauss=[]
        if mesh.element_type == 'hex':
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

        elif mesh.element_type == 'quad':
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

        elif mesh.element_type == 'tet':
            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
            Domain.AllGauss = np.zeros((w.shape[0],1))  
            for counter in range(0,w.shape[0]):
                # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
                Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
                Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
                Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

                Domain.AllGauss[counter,0] = w[counter]

        elif mesh.element_type == 'tri':
            Domain.Jm = [];  Domain.AllGauss = []

            Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))   
            Domain.AllGauss = np.zeros((w.shape[0],1))  
            for counter in range(0,w.shape[0]):
                # Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
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

