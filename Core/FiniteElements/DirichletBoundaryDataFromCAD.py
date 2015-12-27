import numpy as np, sys

from Core.QuadratureRules import GaussLobattoQuadrature
from Core.QuadratureRules.FeketePointsTri import FeketePointsTri
from Core.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints

from Core.MeshGeneration.CurvilinearMeshing.IGAKitPlugin.IdentifyNURBSBoundaries import GetDirichletData
from Core import PostMeshCurvePy as PostMeshCurve 
from Core import PostMeshSurfacePy as PostMeshSurface 

def IGAKitWrapper(MainData,mesh):
    """Calls IGAKit wrapper to get exact Dirichlet boundary conditions"""

    # GET THE NURBS CURVE FROM PROBLEMDATA
    nurbs = MainData.BoundaryData().NURBSParameterisation()
    # IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
    nodesDBC, Dirichlet = GetDirichletData(mesh,nurbs,MainData.BoundaryData,MainData.C) 

    return nodesDBC[:,None], Dirichlet



def PostMeshWrapper(MainData,mesh):
    """Calls PostMesh wrapper to get exact Dirichlet boundary conditions"""

    # GET BOUNDARY FEKETE POINTS
    if MainData.ndim == 2:
        
        # CHOOSE TYPE OF BOUNDARY SPACING 
        boundary_fekete = np.array([[]])
        spacing_type = getattr(MainData.BoundaryData,'CurvilinearMeshNodalSpacing',None)
        if spacing_type == 'fekete':
            boundary_fekete = GaussLobattoQuadrature(MainData.C+2)[0]
        else:
            boundary_fekete = EquallySpacedPoints(MainData.ndim,MainData.C)
        # IT IS IMPORTANT TO ENSURE THAT THE DATA IS C-CONITGUOUS
        boundary_fekete = boundary_fekete.copy(order="c")

        curvilinear_mesh = PostMeshCurve(mesh.element_type,dimension=MainData.ndim)
        curvilinear_mesh.SetMeshElements(mesh.elements)
        curvilinear_mesh.SetMeshPoints(mesh.points)
        curvilinear_mesh.SetMeshEdges(mesh.edges)
        curvilinear_mesh.SetMeshFaces(np.zeros((1,4),dtype=np.uint64))
        curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
        curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
        curvilinear_mesh.SetProjectionPrecision(1.0e-04)
        curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
        curvilinear_mesh.ScaleMesh()
        # curvilinear_mesh.InferInterpolationPolynomialDegree() 
        curvilinear_mesh.SetFeketePoints(boundary_fekete)
        curvilinear_mesh.GetBoundaryPointsOrder()
        # READ THE GEOMETRY FROM THE IGES FILE
        curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
        # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
        geometry_points = curvilinear_mesh.GetGeomVertices()
        curvilinear_mesh.GetGeomEdges()
        curvilinear_mesh.GetGeomFaces()

        curvilinear_mesh.GetGeomPointsOnCorrespondingEdges()
        # FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
        curvilinear_mesh.IdentifyCurvesContainingEdges()
        # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
        curvilinear_mesh.ProjectMeshOnCurve()
        # FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
        curvilinear_mesh.RepairDualProjectedParameters()
        # PERFORM POINT INVERTION FOR THE INTERIOR POINTS
        projection_type = getattr(MainData.BoundaryData,'ProjectionType',None)
        if projection_type == 'orthogonal':
            curvilinear_mesh.MeshPointInversionCurve()
        elif projection_type == 'arc_length':
            curvilinear_mesh.MeshPointInversionCurveArcLength()
        else:
            print("projection type not understood. Arc length based projection is going to be used")
            curvilinear_mesh.MeshPointInversionCurveArcLength()
        # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
        curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
        # GET DIRICHLET DATA
        nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 
        # FIND UNIQUE VALUES OF DIRICHLET DATA
        posUnique = np.unique(nodesDBC,return_index=True)[1]
        nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:]

        # GET ACTUAL CURVE POINTS - THIS FUNCTION IS EXPENSIVE
        # MainData.ActualCurve = curvilinear_mesh.DiscretiseCurves(100)

    elif MainData.ndim == 3:

        boundary_fekete = FeketePointsTri(MainData.C)

        curvilinear_mesh = PostMeshSurface(mesh.element_type,dimension=MainData.ndim)
        curvilinear_mesh.SetMeshElements(mesh.elements)
        curvilinear_mesh.SetMeshPoints(mesh.points)
        if mesh.edges.ndim == 2 and mesh.edges.shape[1]==0:
            mesh.edges = np.zeros((1,4),dtype=np.uint64)
        else:
            curvilinear_mesh.SetMeshEdges(mesh.edges)
        curvilinear_mesh.SetMeshFaces(mesh.faces)
        curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
        curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
        curvilinear_mesh.SetProjectionPrecision(1.0e-04)
        curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
        curvilinear_mesh.ScaleMesh()
        curvilinear_mesh.SetFeketePoints(boundary_fekete)
        # curvilinear_mesh.GetBoundaryPointsOrder()
        # READ THE GEOMETRY FROM THE IGES FILE
        curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
        # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
        geometry_points = curvilinear_mesh.GetGeomVertices()
        # curvilinear_mesh.GetGeomEdges()
        curvilinear_mesh.GetGeomFaces()
        curvilinear_mesh.GetGeomPointsOnCorrespondingFaces()
        # FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
        curvilinear_mesh.IdentifySurfacesContainingFaces()
        # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
        curvilinear_mesh.ProjectMeshOnSurface()
        # FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
        # curvilinear_mesh.RepairDualProjectedParameters()
        # PERFORM POINT INVERTION FOR THE INTERIOR POINTS
        curvilinear_mesh.MeshPointInversionSurface()
        # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
        curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
        # GET DIRICHLET DATA
        nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 
        # FIND UNIQUE VALUES OF DIRICHLET DATA
        posUnique = np.unique(nodesDBC,return_index=True)[1]
        nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:]


    return nodesDBC, Dirichlet