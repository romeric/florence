import numpy as np


def test_BEM():
    """Unnecessary test for the ugly and non-working and legacy BEM
        for the sake of coverage
    """


    from Florence.BoundaryElements import GetBases as GetBasesBEM2D
    from Florence.BoundaryElements import GenerateCoordinates
    from Florence.BoundaryElements import CoordsJacobianRadiusatGaussPoints, CoordsJacobianRadiusatGaussPoints_LM
    from Florence.BoundaryElements import AssemblyBEM2D
    from Florence.BoundaryElements.Assembly import AssemblyBEM2D_Sparse
    from Florence.BoundaryElements import Sort_BEM
    from Florence import QuadratureRule, FunctionSpace, Mesh

    # Unnecessary loop
    for i in range(10):

        mesh = Mesh()
        mesh.element_type = "line"
        mesh.points = np.array([
            [0.,0.],
            [1.,0.],
            [1.,1.],
            [0.,1.],
            ])
        mesh.elements = np.array([
            [0,1],
            [1,2],
            [2,3],
            [3,0],
            ])
        mesh.nelem = 4



        q = QuadratureRule(mesh_type="line")
        for C in range(10):
            N, dN = GetBasesBEM2D(C,q.points)

        N, dN = GetBasesBEM2D(2,q.points)
        global_coord = np.zeros((mesh.points.shape[0],3))
        global_coord[:,:2] = mesh.points
        Jacobian = 2*np.ones((q.weights.shape[0],mesh.nelem))
        nx = 4*np.ones((q.weights.shape[0],mesh.nelem))
        ny = 3*np.ones((q.weights.shape[0],mesh.nelem))
        XCO = 2*np.ones((q.weights.shape[0],mesh.nelem))
        YCO = np.ones((q.weights.shape[0],mesh.nelem))
        N = np.ones((mesh.elements.shape[1],q.weights.shape[0]))
        dN = 0.5*np.ones((mesh.elements.shape[1],q.weights.shape[0]))
        
        GenerateCoordinates(mesh.elements,mesh.points,0,q.points)
        CoordsJacobianRadiusatGaussPoints(mesh.elements,global_coord,0,N,dN,q.weights)
        # Not working
        # CoordsJacobianRadiusatGaussPoints_LM(mesh.elements,global_coord[:,:3],0,N,dN,q.weights,mesh.elements)

        class GeoArgs(object):
            Lagrange_Multipliers = "activated"
            def __init__(self):
                Lagrange_Multipliers = "activated"

        geo_args = GeoArgs()
        K1, K2 = AssemblyBEM2D(0,global_coord,mesh.elements,mesh.elements,dN,N,
            q.weights,q.points,Jacobian, nx, ny, XCO, YCO, geo_args)
        AssemblyBEM2D_Sparse(0,global_coord,mesh.elements,mesh.elements,dN,N,
            q.weights,q.points,Jacobian, nx, ny, XCO, YCO, geo_args)

        bdata = np.zeros((2*mesh.points.shape[0],2))
        bdata[:4,1] = -1
        bdata[4:,0] = -1
        Sort_BEM(bdata,K1, K2)




if __name__ == "__main__":
    test_BEM()