import numpy as np
from Florence import *



def test_mesh_postprocess_material():

    print("Running tests on Mesh, PostProcess and Material modules")

    mesh = Mesh()
    mesh.Line()
    mesh.GetHighOrderMesh(p=5, check_duplicates=False)
    mesh.Smooth()
    mesh.Refine()
    mesh.GetNumberOfElements()
    mesh.GetNumberOfNodes()
    mesh.InferElementType()
    mesh.InferBoundaryElementType()
    mesh.InferPolynomialDegree()
    mesh.InferSpatialDimension()
    mesh.InferNumberOfNodesPerElement()
    mesh.InferNumberOfNodesPerLinearElement()
    mesh.NodeArranger(C=2)
    mesh.CreateDummyLowerDimensionalMesh()
    mesh.CreateDummyUpperDimensionalMesh()

    pp = PostProcess(2,2)
    pp.SetMesh(mesh)
    pp.Tessellate(interpolation_degree=3)
    mesh.__reset__()



    etypes = ["tri", "quad"]

    for etype in etypes:
        
        mesh = Mesh()
        mesh.Square(element_type=etype, nx=5,ny=5)
        mesh.GetEdges()
        mesh.Smooth()
        mesh.Refine()
        mesh.GetNumberOfElements()
        mesh.GetNumberOfNodes()
        mesh.InferElementType()
        mesh.InferBoundaryElementType()
        mesh.InferPolynomialDegree()
        mesh.InferSpatialDimension()
        mesh.InferNumberOfNodesPerElement()
        mesh.InferNumberOfNodesPerLinearElement()
        mesh.NodeArranger(C=2)
        mesh.CreateDummyLowerDimensionalMesh()
        mesh.CreateDummyUpperDimensionalMesh()

        pp = PostProcess(2,2)
        pp.SetMesh(mesh)
        pp.Tessellate(interpolation_degree=3)
        mesh.__reset__()

        
        mesh.CircularPlate(element_type=etype)
        mesh.RemoveElements(mesh.Bounds)
        mesh.GetHighOrderMesh(p=2, check_duplicates=False)
        mesh.GetHighOrderMesh(p=3, check_duplicates=False)
        mesh = mesh.GetLinearMesh(remap=True)
        mesh = mesh.GetLocalisedMesh(elements=range(mesh.nelem))
        mesh.CircularArcPlate(element_type=etype)
        mesh.GetHighOrderMesh(p=4, check_duplicates=False)

        pp = PostProcess(2,2)
        pp.SetMesh(mesh)
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.ConstructDifferentOrderSolution(p=4)
        pp.ConstructDifferentOrderSolution(p=5)
        pp.Tessellate(interpolation_degree=3)
        pp.SetFormulation(DisplacementFormulation(mesh))
        pp.SetFEMSolver(FEMSolver())
        pp.SetMaterial(MooneyRivlin(2,mu1=1.,mu2=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NeoHookeanCoercive(2,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NeoHookean_1(2,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(MooneyRivlin_1(2,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NearlyIncompressibleMooneyRivlin(2,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NearlyIncompressibleNeoHookean(2,mu=1.,pressure=np.zeros(mesh.nelem)))
        pp.GetAugmentedSolution()

        pp = PostProcess(2,3)
        pp.SetMesh(mesh)
        pp.SetFormulation(DisplacementPotentialFormulation(mesh))
        pp.SetFEMSolver(FEMSolver())
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_0(2,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_1(2,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_2(2,mu=1.,lamb=10.,c1=1e-5, c2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(SteinmannModel(2,mu=1.,lamb=10.,c1=1e-5, c2=1e-5, eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_1(2,mu=1.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_200(2,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_201(2,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_101(2,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_105(2,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_106(2,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_107(2,mu1=1.,mu2=2.,mue=0.5,lamb=10.,eps_1=1e-5,eps_2=1e-5,eps_e=1e-7))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(IsotropicElectroMechanics_108(2,mu1=1.,mu2=2.,lamb=10.,eps_2=1e-7))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,3)))
        pp.SetMaterial(Piezoelectric_100(2,mu1=1.,mu2=2.,mu3=0.5,lamb=10.,eps_1=1e-5,eps_2=1e-5,eps_3=1e-7, 
            anisotropic_orientations=np.zeros((mesh.nelem,2))))
        pp.GetAugmentedSolution()
        mesh.__reset__()


        mesh.InverseArc(element_type=etype)
        mesh.HollowCircle(element_type=etype)
        mesh.HollowArc(element_type=etype)
        mesh.Circle(element_type=etype)
        mesh.Arc(element_type=etype)
        mesh.Triangle(element_type=etype)
        mesh.GetNodeCommonality()
        mesh.GetInteriorEdges()
        mesh.__reset__()




    etypes = ["tet", "hex"]

    for etype in etypes:
        
        mesh = Mesh()
        mesh.Cube(element_type=etype, nx=5,ny=5,nz=5)
        mesh.Refine()
        mesh.GetFaces()
        mesh.GetEdges()
        mesh.GetNumberOfElements()
        mesh.GetNumberOfNodes()
        mesh.InferElementType()
        mesh.InferBoundaryElementType()
        mesh.InferPolynomialDegree()
        mesh.InferSpatialDimension()
        mesh.InferNumberOfNodesPerElement()
        mesh.InferNumberOfNodesPerLinearElement()
        mesh.NodeArranger(C=2)
        mesh.CreateDummyLowerDimensionalMesh()
        mesh.CreateDummyUpperDimensionalMesh()

        pp = PostProcess(3,3)
        pp.SetMesh(mesh)
        pp.Tessellate(interpolation_degree=3)
        mesh.__reset__()

        
        mesh.SphericalArc(element_type=etype)
        mesh.RemoveElements(mesh.Bounds)
        mesh.GetHighOrderMesh(p=2, check_duplicates=False)
        mesh.GetHighOrderMesh(p=3, check_duplicates=False)
        mesh = mesh.GetLinearMesh(remap=True)
        mesh = mesh.GetLocalisedMesh(elements=range(mesh.nelem))
        mesh.GetInteriorEdges()
        mesh.GetInteriorFaces()

        pp = PostProcess(3,3)
        pp.SetMesh(mesh)
        pp.Tessellate(interpolation_degree=3)
        mesh.__reset__()


        mesh.HollowSphere(element_type=etype, ncirc=3, nrad=2)
        mesh = mesh.GetLinearMesh(remap=True)
        mesh = mesh.GetLocalisedMesh(elements=range(mesh.nelem))
        mesh.GetHighOrderMesh(p=2, check_duplicates=False)
        mesh = mesh.ConvertToLinearMesh()
        mesh == mesh
        mesh < mesh
        mesh <= mesh
        mesh > mesh
        mesh >= mesh

        pp = PostProcess(3,3)
        pp.SetMesh(mesh)
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.Tessellate(interpolation_degree=3)
        pp.ConstructDifferentOrderSolution(p=2)
        pp.ConstructDifferentOrderSolution(p=3)
        pp.Tessellate(interpolation_degree=3)
        pp.SetFormulation(DisplacementFormulation(mesh))
        pp.SetFEMSolver(FEMSolver())
        pp.SetMaterial(MooneyRivlin(3,mu1=1.,mu2=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NeoHookeanCoercive(3,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NeoHookean_1(3,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(MooneyRivlin_1(3,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NearlyIncompressibleMooneyRivlin(3,mu=1.,lamb=10.))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros_like(mesh.points))
        pp.SetMaterial(NearlyIncompressibleNeoHookean(3,mu=1.,pressure=np.zeros(mesh.nelem)))
        pp.GetAugmentedSolution()

        pp = PostProcess(3,4)
        pp.SetMesh(mesh)
        pp.SetFormulation(DisplacementPotentialFormulation(mesh))
        pp.SetFEMSolver(FEMSolver())
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_0(3,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_1(3,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_2(3,mu=1.,lamb=10.,c1=1e-5, c2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(SteinmannModel(3,mu=1.,lamb=10.,c1=1e-5, c2=1e-5, eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_1(3,mu=1.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_200(3,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_201(3,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_101(3,mu=1.,lamb=10.,eps_1=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_105(3,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_106(3,mu1=1.,mu2=2.,lamb=10.,eps_1=1e-5,eps_2=1e-5))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_107(3,mu1=1.,mu2=2.,mue=0.5,lamb=10.,eps_1=1e-5,eps_2=1e-5,eps_e=1e-7))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(IsotropicElectroMechanics_108(3,mu1=1.,mu2=2.,lamb=10.,eps_2=1e-7))
        pp.GetAugmentedSolution()
        pp.SetSolution(np.zeros((mesh.nnode,4)))
        pp.SetMaterial(Piezoelectric_100(3,mu1=1.,mu2=2.,mu3=0.5,lamb=10.,eps_1=1e-5,eps_2=1e-5,eps_3=1e-7, 
            anisotropic_orientations=np.zeros((mesh.nelem,3))))
        pp.GetAugmentedSolution()
        mesh.__reset__()



    mesh.Cylinder(element_type="hex")
    mesh.ArcCylinder(element_type="hex")
    mesh.HollowCylinder(element_type="hex")
    mesh = mesh.GetLinearMesh(remap=True)
    mesh = mesh.GetLocalisedMesh(elements=range(mesh.nelem))
    mesh.GetHighOrderMesh(p=2, check_duplicates=False)
    mesh = mesh.ConvertToLinearMesh()
    mesh == mesh
    mesh < mesh
    mesh <= mesh
    mesh > mesh
    mesh >= mesh

    pp = PostProcess(3,3)
    pp.SetMesh(mesh)
    pp.Tessellate(interpolation_degree=3)
    mesh.__reset__()



    mesh = mesh.TriangularProjection()
    mesh.ConvertTrisToQuads()
    mesh = mesh.QuadrilateralProjection()
    mesh.ConvertQuadsToTris()
    mesh = mesh.TetrahedralProjection()
    mesh.ConvertTetsToHexes()
    mesh = mesh.HexahedralProjection()
    mesh.ConvertHexesToTets()


    print("Successfully finished running tests on Mesh, PostProcess and Material modules")



if __name__ == "__main__":
    test_mesh_postprocess_material()