import numpy as np
import os
from Florence import *
from Florence.MeshGeneration.CustomMesher import SubdivisionArc, SubdivisionCircle, HarvesterPatch


def test_quadrature_functionspace():

    print("Running tests on QuadratureRule and FunctionSpace modules")

    mesh = Mesh()
    etypes = ["line","tri","quad","tet","hex"]
    for etype in etypes:

        if etype == "line":
            mesh.Line()
        elif etype == "tri":
            mesh.Circle(element_type="tri")
        elif etype == "quad":
            mesh.Circle(element_type="quad")
        elif etype == "tet":
            mesh.Cube(element_type="tet",nx=1,ny=1,nz=1)
        elif etype == "hex":
            mesh.Cube(element_type="hex",nx=1,ny=1,nz=1)

        for p in range(2,7):
            mesh.GetHighOrderMesh(p=p, check_duplicates=False)

            q = QuadratureRule(mesh_type=etype, norder=p+2, flatten=False)
            FunctionSpace(mesh, q, p=p, equally_spaced=False)
            FunctionSpace(mesh, q, p=p, equally_spaced=True)

            q = QuadratureRule(mesh_type=etype, norder=p+2, flatten=True)
            FunctionSpace(mesh, q, p=p, equally_spaced=False)
            FunctionSpace(mesh, q, p=p, equally_spaced=True)

    # now test all Fekete/ES point creation
    from Florence.QuadratureRules import FeketePointsTri
    from Florence.QuadratureRules.QuadraturePointsWeightsTri import QuadraturePointsWeightsTri
    from Florence.QuadratureRules import FeketePointsTet
    from Florence.QuadratureRules.QuadraturePointsWeightsTet import QuadraturePointsWeightsTet
    from Florence.QuadratureRules import GaussLobattoPoints1D, GaussLobattoPointsQuad, GaussLobattoPointsHex
    from Florence.QuadratureRules.QuadraturePointsWeightsTri import QuadraturePointsWeightsTri
    from Florence.QuadratureRules.WVQuadraturePointsWeightsQuad import WVQuadraturePointsWeightsQuad
    from Florence.QuadratureRules.WVQuadraturePointsWeightsHex import WVQuadraturePointsWeightsHex
    from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints, EquallySpacedPointsTri, EquallySpacedPointsTet
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementLine, NodeArrangementTri, NodeArrangementQuad
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementTet, NodeArrangementHex, NodeArrangementQuadToTri
    from Florence.MeshGeneration.NodeArrangement import NodeArrangementHexToTet, NodeArrangementLayeredToHex


    for i in range(21):
        FeketePointsTri(i)
        QuadraturePointsWeightsTri(i,3)
        QuadraturePointsWeightsTet(i)
        EquallySpacedPoints(2,i)
        EquallySpacedPoints(3,i)
        EquallySpacedPoints(4,i)
        EquallySpacedPointsTri(i)
        EquallySpacedPointsTet(i)

        NodeArrangementLine(i)
        NodeArrangementTri(i)
        NodeArrangementQuad(i)
        NodeArrangementHex(i)
        NodeArrangementLayeredToHex(i)

        if i<6:
            NodeArrangementQuadToTri(i)

        if i<2:
            NodeArrangementHexToTet(i)

        if i<18:
            FeketePointsTet(i)
            NodeArrangementTet(i)
        WVQuadraturePointsWeightsQuad(i)
        if i<=16:
            WVQuadraturePointsWeightsHex(i)

    print("Successfully finished running tests on QuadratureRule and FunctionSpace modules\n")



def test_mesh_postprocess_material():

    print("Running tests on Mesh, PostProcess and Material modules")

    mesh = Mesh()
    mesh.Line()
    mesh.GetHighOrderMesh(p=5, check_duplicates=False)
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
    mesh.IsHighOrder
    mesh.IsCurvilinear
    mesh.IsEquallySpaced
    mesh.IsSimilar(mesh)
    mesh.IsEqualOrder(mesh)
    mesh.Partition(10)

    pp = PostProcess(2,2)
    pp.SetMesh(mesh)
    pp.Tessellate(interpolation_degree=3)
    mesh.__reset__()


    etypes = ["tri", "quad"]

    for etype in etypes:

        mesh = Mesh()
        mesh.Square(element_type=etype, nx=3,ny=3)
        mesh.GetEdges()
        mesh.GetInteriorEdges()
        mesh.GetBoundaryEdges()
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
        mesh.IsHighOrder
        mesh.IsCurvilinear
        mesh.IsEquallySpaced
        mesh.Partition(7)
        mesh.Normals()
        mesh.Areas()
        mesh.Lengths()
        mesh.EdgeLengths()
        mesh.AspectRatios()
        mesh.Medians()
        mesh.LargestSegment(nsamples=10)
        mesh.CheckNodeNumbering()
        mesh.BoundingBoxes()

        pp = PostProcess(2,2)
        pp.SetMesh(mesh)
        pp.Tessellate(interpolation_degree=3)
        mesh.__reset__()


        mesh.CircularPlate(element_type=etype)
        mesh.RemoveElements(mesh.Bounds)
        mesh.LaplacianSmoothing(niter=1)
        mesh.LaplacianSmoothing(niter=1, smart=True)
        mesh.LaplacianSmoothing(niter=1,algorithm="gauss_seidel")
        mesh.LaplacianSmoothing(niter=1,algorithm="gauss_seidel", smart=True)
        mesh.GetHighOrderMesh(p=2, check_duplicates=False)
        mesh.GetHighOrderMesh(p=3, check_duplicates=False)
        mesh.LaplacianSmoothing(niter=3,pnodes_movement="laplacian")
        mesh.LaplacianSmoothing(niter=3,pnodes_movement="interpolation")
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
        pp.GetAugmentedSolution(parallelise=True)

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
        pp.GetAugmentedSolution(parallelise=True)

        mesh = mesh.GetLinearMesh(remap=True)
        try:
            mesh.SimplePlot(show_plot=False)
        except:
            pass
        try:
            mesh.WriteVTK("dummy")
        except:
            pass
        try:
            pp.WriteHDF5("dummy")
        except:
            pass

        try:
            pp.Plot(show_plot=False)
        except:
            pass
        try:
            pp.Animate(show_plot=False)
        except:
            pass
        try:
            pp.CurvilinearPlot(show_plot=False)
        except:
            pass
        try:
            pp.WriteVTK("dummy", fmt="xml")
        except:
            pass
        try:
            pp.WriteVTK("dummy", fmt="binary")
        except:
            pass
        try:
            pp.WriteHDF5("dummy")
        except:
            pass

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
        mesh.Cube(element_type=etype, nx=3,ny=3,nz=3)
        mesh.Refine()
        mesh.GetFaces()
        mesh.GetInteriorFaces()
        mesh.GetBoundaryFaces()
        mesh.GetEdges()
        mesh.GetInteriorEdges()
        mesh.GetBoundaryEdges()
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
        mesh.Normals()
        mesh.FaceNormals()
        mesh.Volumes()
        mesh.Lengths()
        mesh.EdgeLengths()
        mesh.AspectRatios()
        mesh.Medians()
        mesh.LargestSegment(nsamples=10)
        mesh.CheckNodeNumbering()
        mesh.BoundingBoxes()

        pp = PostProcess(3,3)
        pp.SetMesh(mesh)
        pp.Tessellate(interpolation_degree=3)
        mesh.__reset__()


        mesh.SphericalArc(element_type=etype)
        mesh.RemoveElements(mesh.Bounds)
        mesh.LaplacianSmoothing(niter=1)
        mesh.LaplacianSmoothing(niter=1, smart=True)
        mesh.LaplacianSmoothing(niter=1,algorithm="gauss_seidel")
        mesh.LaplacianSmoothing(niter=1,algorithm="gauss_seidel", smart=True)
        mesh.GetHighOrderMesh(p=2, check_duplicates=False)
        mesh.GetHighOrderMesh(p=3, check_duplicates=False)
        mesh.LaplacianSmoothing(niter=3,pnodes_movement="laplacian")
        mesh.LaplacianSmoothing(niter=3,pnodes_movement="interpolation")
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
        mesh.IsHighOrder
        mesh.IsCurvilinear
        mesh.IsEquallySpaced
        mesh.Partition(7)

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
        pp.GetAugmentedSolution(parallelise=True)

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
        pp.GetAugmentedSolution(parallelise=True)

        mesh = mesh.GetLinearMesh(remap=True)
        try:
            mesh.SimplePlot(show_plot=False)
        except:
            pass
        try:
            mesh.WriteVTK("dummy")
        except:
            pass
        try:
            pp.WriteHDF5("dummy")
        except:
            pass

        try:
            pp.Plot(show_plot=False)
        except:
            pass
        try:
            pp.Animate(show_plot=False)
        except:
            pass
        try:
            pp.CurvilinearPlot(show_plot=False)
        except:
            pass
        try:
            pp.WriteVTK("dummy", fmt="xml")
        except:
            pass
        try:
            pp.WriteVTK("dummy", fmt="binary")
        except:
            pass
        try:
            pp.WriteHDF5("dummy")
        except:
            pass

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
    mesh.edge_to_element, mesh.boundary_edge_to_element = None, None
    mesh.GetElementsEdgeNumbering()
    mesh.GetElementsWithBoundaryEdges()

    mesh = mesh.QuadrilateralProjection()
    mesh.ConvertQuadsToTris()
    mesh.edge_to_element, mesh.boundary_edge_to_element = None, None
    mesh.GetElementsEdgeNumbering()
    mesh.GetElementsWithBoundaryEdges()

    mesh = mesh.TetrahedralProjection()
    mesh.ConvertTetsToHexes()
    mesh.face_to_element, mesh.boundary_face_to_element = None, None
    mesh.GetElementsFaceNumbering()
    mesh.GetElementsWithBoundaryFaces()

    mesh = mesh.HexahedralProjection()
    mesh.ConvertHexesToTets()
    mesh.face_to_element, mesh.boundary_face_to_element = None, None
    mesh.GetElementsFaceNumbering()
    mesh.GetElementsWithBoundaryFaces()


    mesh.Square(element_type="quad")
    mesh.points *= 1+np.random.rand(mesh.nnode,mesh.points.shape[1])/20.
    assert np.isclose(mesh.Areas().sum(), mesh.Sizes().sum(),atol=1e-6,rtol=1e-6)
    mesh.Square(element_type="quad")
    idx0 = mesh.FindElementContainingPoint([.31,.55],algorithm="fem")
    idx1 = mesh.FindElementContainingPoint([.31,.55],algorithm="geometric")
    assert idx0[0] == idx1[0]

    mesh.Cube(element_type="tet")
    idx0 = mesh.FindElementContainingPoint([.31,.55,0.27],algorithm="fem")
    idx1 = mesh.FindElementContainingPoint([.31,.55,0.27],algorithm="geometric")
    assert idx0[0] == idx1[0]


    try:
        mesh.ReadDCM("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadFRO("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadGIDMesh("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadOBJ("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadGmsh("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadSalome("None",element_type="tri")
    except:
        pass
    try:
        mesh.ReadUNV("None",element_type="tri")
    except:
        pass

    mesh = HarvesterPatch()
    mesh = SubdivisionCircle(element_type="quad")
    mesh.Circle(algorithm="midpoint_subdivision")
    mesh = SubdivisionArc(element_type="quad")
    mesh.Arc(algorithm="midpoint_subdivision")

    try:
        mesh.WriteGmsh("dummy.txt")
    except:
        pass
    try:
        mesh.WriteOBJ("dummy.txt")
    except:
        pass
    try:
        mesh.WriteMFEM("dummy.txt")
    except:
        pass

    try:
        os.remove("dummy.txt")
        os.remove("dummy.vtu")
        os.remove("dummy.mat")
    except:
        pass

    try:
        dummy_files = os.listdir(os.getcwd())
        for file in dummy_files:
            if "dummy" in file:
                os.remove(file)
    except:
        pass

    print("Successfully finished running tests on Mesh, PostProcess and Material modules\n")





def test_material():

    print("Running tests on high and low level Material modules")


    material_list = [
                        "IdealDielectric",
                        "AnisotropicIdealDielectric",
                        "LinearElastic",
                        "IncrementalLinearElastic",
                        "NeoHookean",
                        "NeoHookean_1",
                        "RegularisedNeoHookean",
                        "NeoHookeanCoercive",
                        "MooneyRivlin",
                        "MooneyRivlin_1",
                        "NearlyIncompressibleNeoHookean",
                        "NearlyIncompressibleMooneyRivlin",
                        "AnisotropicMooneyRivlin_0",
                        "AnisotropicMooneyRivlin_1",
                        "BonetTranservselyIsotropicHyperElastic",
                        "TranservselyIsotropicHyperElastic",
                        "TranservselyIsotropicLinearElastic",
                        "ExplicitMooneyRivlin",
                        "IsotropicElectroMechanics_0",
                        "IsotropicElectroMechanics_1",
                        "IsotropicElectroMechanics_2",
                        "IsotropicElectroMechanics_3",
                        "SteinmannModel",
                        "IsotropicElectroMechanics_200",
                        "IsotropicElectroMechanics_201",
                        "IsotropicElectroMechanics_100",
                        "IsotropicElectroMechanics_101",
                        "IsotropicElectroMechanics_102",
                        "IsotropicElectroMechanics_103",
                        "IsotropicElectroMechanics_104",
                        "IsotropicElectroMechanics_105",
                        "IsotropicElectroMechanics_106",
                        "IsotropicElectroMechanics_107",
                        "IsotropicElectroMechanics_108",
                        "IsotropicElectroMechanics_109",
                        "Piezoelectric_100",
                        "ExplicitIsotropicElectroMechanics_108",
                        "Multi_IsotropicElectroMechanics_101",
                        "Multi_Piezoelectric_100",
                        "CoupleStressModel",
                        "IsotropicLinearFlexoelectricModel",
                    ]

    mu,mu1,mu2,mu3,mue,mu_v,lamb,eps,eps_1,eps_2,eps_3,eps_e,c1,c2,eta,kappa,E,E_A,G_A,nu = [1.]*20
    gamma = 0.5

    # The outer loop is not necessary - done for the purpose of coverage
    for i in range(5):
        for ndim in [2,3]:
            P = np.zeros((6,3)) if ndim==3 else np.zeros((3,2))
            f = np.eye(ndim,ndim)
            e = np.eye(ndim,ndim)
            for material_name in material_list:
                material = eval(material_name)(ndim,mu=mu,mu1=mu1,mu2=mu2,mu3=mu3,mue=mue,mu_v=mu_v,lamb=lamb,
                    E=E,E_A=E_A,G_A=G_A,nu=nu, gamma=gamma, Jbar=[0],
                    mus=[mu],mu1s=[mu1],mu2s=[mu2],mu3s=[mu3],lambs=[lamb],eps_1s=[eps_1],eps_2s=[eps_2],eps_3s=[eps_3],
                    eps=eps,eps_1=eps_1,eps_2=eps_2,eps_3=eps_3,eps_e=eps_e,c1=c1,c2=c2,
                    eta=eta,kappa=kappa, P=P, e=e, f=f,
                    pressure=[0],anisotropic_orientations=np.random.rand(1,ndim))
                F = np.eye(material.ndim,material.ndim)[None,:,:]
                E = np.random.rand(material.ndim)
                StrainTensor = KinematicMeasures(F,material.nature)

                # Python level
                material.CauchyStress(StrainTensor,E)
                material.Hessian(StrainTensor,E)
                if hasattr(material,"ElectricDisplacementx"):
                    material.ElectricDisplacementx(StrainTensor,E)
                if hasattr(material,"InternalEnergy"):
                    material.InternalEnergy(StrainTensor,E)

                # Check low level
                if hasattr(material,"KineticMeasures"):
                    if material.mtype != "CoupleStressModel" and material.mtype !="IsotropicLinearFlexoelectricModel":
                        E = np.random.rand(material.ndim)[None,:]
                        material.KineticMeasures(F,E)


    print("Successfully finished running tests on high and low level Material modules\n")



if __name__ == "__main__":
    test_quadrature_functionspace()
    test_mesh_postprocess_material()
    test_material()