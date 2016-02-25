# import numpy as np, sys, gc
# from warnings import warn

# from Florence.QuadratureRules import GaussLobattoQuadrature
# from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
# from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
# import Florence.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri

# from Florence.MeshGeneration.CurvilinearMeshing.IGAKitPlugin.IdentifyNURBSBoundaries import GetDirichletData
# from Florence import PostMeshCurvePy as PostMeshCurve 
# from Florence import PostMeshSurfacePy as PostMeshSurface 

# def IGAKitWrapper(MainData,mesh):
#     """Calls IGAKit wrapper to get exact Dirichlet boundary conditions"""

#     # GET THE NURBS CURVE FROM PROBLEMDATA
#     nurbs = MainData.BoundaryData().NURBSParameterisation()
#     # IDENTIFIY DIRICHLET BOUNDARY CONDITIONS BASED ON THE EXACT GEOMETRY
#     nodesDBC, Dirichlet = GetDirichletData(mesh,nurbs,MainData.BoundaryData,MainData.C) 

#     return nodesDBC[:,None], Dirichlet



# def PostMeshWrapper(MainData,mesh,material):
#     """Calls PostMesh wrapper to get exact Dirichlet boundary conditions"""

#     # GET BOUNDARY FEKETE POINTS
#     if MainData.ndim == 2:
        
#         # CHOOSE TYPE OF BOUNDARY SPACING 
#         boundary_fekete = np.array([[]])
#         spacing_type = getattr(MainData.BoundaryData,'CurvilinearMeshNodalSpacing',None)
#         if spacing_type == 'fekete':
#             boundary_fekete = GaussLobattoQuadrature(MainData.C+2)[0]
#         else:
#             boundary_fekete = EquallySpacedPoints(MainData.ndim,MainData.C)
#         # IT IS IMPORTANT TO ENSURE THAT THE DATA IS C-CONITGUOUS
#         boundary_fekete = boundary_fekete.copy(order="c")

#         curvilinear_mesh = PostMeshCurve(mesh.element_type,dimension=MainData.ndim)
#         curvilinear_mesh.SetMeshElements(mesh.elements)
#         curvilinear_mesh.SetMeshPoints(mesh.points)
#         curvilinear_mesh.SetMeshEdges(mesh.edges)
#         curvilinear_mesh.SetMeshFaces(np.zeros((1,4),dtype=np.uint64))
#         curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
#         curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
#         curvilinear_mesh.SetProjectionPrecision(1.0e-04)
#         curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
#         curvilinear_mesh.ScaleMesh()
#         # curvilinear_mesh.InferInterpolationPolynomialDegree() 
#         curvilinear_mesh.SetNodalSpacing(boundary_fekete)
#         curvilinear_mesh.GetBoundaryPointsOrder()
#         # READ THE GEOMETRY FROM THE IGES FILE
#         curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
#         # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
#         geometry_points = curvilinear_mesh.GetGeomVertices()
#         # print np.max(geometry_points[:,0]), mesh.Bounds
#         # exit()
#         curvilinear_mesh.GetGeomEdges()
#         curvilinear_mesh.GetGeomFaces()

#         curvilinear_mesh.GetGeomPointsOnCorrespondingEdges()
#         # FIRST IDENTIFY WHICH CURVES CONTAIN WHICH EDGES
#         curvilinear_mesh.IdentifyCurvesContainingEdges()
#         # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE CURVE
#         curvilinear_mesh.ProjectMeshOnCurve()
#         # FIX IMAGES AND ANTI IMAGES IN PERIODIC CURVES/SURFACES
#         curvilinear_mesh.RepairDualProjectedParameters()
#         # PERFORM POINT INVERSION FOR THE INTERIOR POINTS
#         projection_type = getattr(MainData.BoundaryData,'ProjectionType',None)
#         if projection_type == 'orthogonal':
#             curvilinear_mesh.MeshPointInversionCurve()
#         elif projection_type == 'arc_length':
#             curvilinear_mesh.MeshPointInversionCurveArcLength()
#         else:
#             print("projection type not understood. Arc length based projection is going to be used")
#             curvilinear_mesh.MeshPointInversionCurveArcLength()
#         # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
#         curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
#         # GET DIRICHLET MainData
#         nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData() 
#         # FIND UNIQUE VALUES OF DIRICHLET DATA
#         # posUnique = np.unique(nodesDBC,return_index=True)[1]
#         # nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique,:]

#         # GET ACTUAL CURVE POINTS - THIS FUNCTION IS EXPENSIVE
#         # MainData.ActualCurve = curvilinear_mesh.DiscretiseCurves(100)

#     elif MainData.ndim == 3:

#         boundary_fekete = FeketePointsTri(MainData.C)

#         curvilinear_mesh = PostMeshSurface(mesh.element_type,dimension=MainData.ndim)
#         curvilinear_mesh.SetMeshElements(mesh.elements)
#         curvilinear_mesh.SetMeshPoints(mesh.points)
#         if mesh.edges.ndim == 2 and mesh.edges.shape[1]==0:
#             mesh.edges = np.zeros((1,4),dtype=np.uint64)
#         else:
#             curvilinear_mesh.SetMeshEdges(mesh.edges)
#         curvilinear_mesh.SetMeshFaces(mesh.faces)
#         curvilinear_mesh.SetScale(MainData.BoundaryData.scale)
#         curvilinear_mesh.SetCondition(MainData.BoundaryData.condition)
#         curvilinear_mesh.SetProjectionPrecision(1.0e-04)
#         curvilinear_mesh.SetProjectionCriteria(MainData.BoundaryData().ProjectionCriteria(mesh))
#         curvilinear_mesh.ScaleMesh()
#         curvilinear_mesh.SetNodalSpacing(boundary_fekete)
#         # curvilinear_mesh.GetBoundaryPointsOrder()
#         # READ THE GEOMETRY FROM THE IGES FILE
#         curvilinear_mesh.ReadIGES(MainData.BoundaryData.IGES_File)
#         # EXTRACT GEOMETRY INFORMATION FROM THE IGES FILE
#         geometry_points = curvilinear_mesh.GetGeomVertices()
#         # print np.max(geometry_points[:,2]), mesh.Bounds
#         # exit()
#         curvilinear_mesh.GetGeomEdges()
#         curvilinear_mesh.GetGeomFaces()
#         print "CAD geometry has", curvilinear_mesh.NbPoints(), "points,", \
#         curvilinear_mesh.NbCurves(), "curves and", curvilinear_mesh.NbSurfaces(), \
#         "surfaces"
#         curvilinear_mesh.GetGeomPointsOnCorrespondingFaces()
#         # FIRST IDENTIFY WHICH SURFACES CONTAIN WHICH FACES
#         # mesh.face_to_surface = None
#         if getattr(mesh,"face_to_surface",None) is not None:
#             if mesh.faces.shape[0] == mesh.face_to_surface.shape[0]:
#                 curvilinear_mesh.SupplySurfacesContainingFaces(mesh.face_to_surface,already_mapped=1)
#             else:
#                 raise AssertionError("face-to-surface mapping does not seem correct. Point projection is going to stop")
#         else:
#             # curvilinear_mesh.IdentifySurfacesContainingFacesByPureProjection()
#             curvilinear_mesh.IdentifySurfacesContainingFaces() 
        
#         # IDENTIFY WHICH EDGES ARE SHARED BETWEEN SURFACES
#         curvilinear_mesh.IdentifySurfacesIntersections()

#         # PERFORM POINT INVERSION FOR THE INTERIOR POINTS
#         Neval = np.zeros((3,boundary_fekete.shape[0]),dtype=np.float64)
#         for i in range(3,boundary_fekete.shape[0]):
#             Neval[:,i]  = Tri.hpBases(0,boundary_fekete[i,0],boundary_fekete[i,1],1)[0]
#         OrthTol = 0.5

#         project_on_curves = 0

#         projection_type = getattr(MainData.BoundaryData,'ProjectionType',None)
#         if projection_type == 'orthogonal':
#             curvilinear_mesh.MeshPointInversionSurface(project_on_curves)
#         elif projection_type == 'arc_length':
#             # PROJECT ALL BOUNDARY POINTS FROM THE MESH TO THE SURFACE
#             curvilinear_mesh.ProjectMeshOnSurface()
#             # curvilinear_mesh.RepairDualProjectedParameters()
#             curvilinear_mesh.MeshPointInversionSurfaceArcLength(project_on_curves,OrthTol,Neval)
#         else:
#             warn("projection type not understood. Orthogonal projection is going to be used")
#             curvilinear_mesh.MeshPointInversionSurface(project_on_curves)

#         # OBTAIN MODIFIED MESH POINTS - THIS IS NECESSARY TO ENSURE LINEAR MESH IS ALSO CORRECT
#         curvilinear_mesh.ReturnModifiedMeshPoints(mesh.points)
#         # GET DIRICHLET DATA
#         nodesDBC, Dirichlet = curvilinear_mesh.GetDirichletData()
#         # GET DIRICHLET FACES (IF REQUIRED)
#         dirichlet_faces = curvilinear_mesh.GetDirichletFaces()

#         # np.savetxt("/home/roman/Dropbox/drill_log2",dirichlet_faces)
#         # np.savetxt("/home/roman/Dropbox/valve_log2",dirichlet_faces)
#         # np.savetxt("/home/roman/Dropbox/almond_log",dirichlet_faces)
#         # np.savetxt("/home/roman/Dropbox/f6BL_log3",dirichlet_faces)
#         # np.savetxt("/home/roman/Dropbox/f6_iso_log2",dirichlet_faces)

#         # np.savetxt("/home/roman/Dropbox/almond_deb1",dirichlet_faces)
#         # np.savetxt("/home/roman/Dropbox/almond_deb2",dirichlet_faces)        
#         # exit()

#         # # FIND UNIQUE VALUES OF DIRICHLET DATA
#         # posUnique = np.unique(nodesDBC,return_index=True)[1]
#         # nodesDBC, Dirichlet = nodesDBC[posUnique], Dirichlet[posUnique


#         # FOR GEOMETRIES CONTAINING PLANAR SURFACES
#         planar_mesh_faces = curvilinear_mesh.GetMeshFacesOnPlanarSurfaces()
#         MainData.planar_mesh_faces = planar_mesh_faces

#         # np.savetxt("/home/roman/Dropbox/nodesDBC_.dat",nodesDBC) 
#         # np.savetxt("/home/roman/Dropbox/Dirichlet_.dat",Dirichlet)
#         # np.savetxt("/home/roman/Dropbox/planar_mesh_faces_.dat",planar_mesh_faces)

#         # if planar_mesh_faces.shape[0] != 0:
#         #     # SOLVE A 2D PROBLEM FOR PLANAR SURFACES
#         #     switcher = MainData.Parallel
#         #     if MainData.Parallel is True or MainData.__PARALLEL__ is True:
#         #         MainData.Parallel = False
#         #         MainData.__PARALLEL__ = False

#         #     GetDirichletDataForPlanarFaces(MainData,material,mesh,planar_mesh_faces,nodesDBC,Dirichlet,plot=False)
#         #     MainData.__PARALLEL__ == switcher
#         #     MainData.Parallel = switcher

#     return nodesDBC, Dirichlet


# def GetDirichletDataForPlanarFaces(MainData,material,mesh,planar_mesh_faces,nodesDBC,Dirichlet,plot=False):
#     """Solve a 2D problem for planar faces. Modifies Dirichlet"""

#     from Core.Supplementary.Tensors import itemfreq_py, makezero
#     from Core import Mesh
#     from Core.FiniteElements.Solvers.Solver import MainSolver
#     from Core.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints
#     from PostProcess import PostProcess

#     surface_flags = itemfreq_py(planar_mesh_faces[:,1])
#     number_of_planar_surfaces = surface_flags.shape[0]

#     E1 = [1.,0.,0.]
#     E2 = [0.,1.,0.]
#     E3 = [0.,0.,1.]

#     # MAKE A SINGLE INSTANCE OF MATERIAL AND UPDATE IF NECESSARY
#     import Core.MaterialLibrary
#     pmaterial_func = getattr(Core.MaterialLibrary,material.mtype,None)
#     pmaterial = pmaterial_func(2,E=material.E,nu=material.nu,E_A=material.E_A,G_A=material.G_A)
    
#     print "The problem requires 2D analyses. Solving", number_of_planar_surfaces, "2D problems"
#     for niter in range(number_of_planar_surfaces):
        
#         pmesh = Mesh()
#         pmesh.element_type = "tri"
#         pmesh.elements = mesh.faces[planar_mesh_faces[planar_mesh_faces[:,1]==surface_flags[niter,0],0],:]
#         pmesh.nelem = np.int64(surface_flags[niter,1])
#         pmesh.GetBoundaryEdgesTri()
#         unique_edges = np.unique(pmesh.edges)
#         Dirichlet2D = np.zeros((unique_edges.shape[0],3))
#         nodesDBC2D = np.zeros(unique_edges.shape[0])
        
#         unique_elements, inv  = np.unique(pmesh.elements, return_inverse=True)
#         aranger = np.arange(unique_elements.shape[0],dtype=np.uint64)
#         pmesh.elements = aranger[inv].reshape(pmesh.elements.shape)

#         # elements = np.zeros_like(pmesh.elements)
#         # unique_elements = np.unique(pmesh.elements)
#         # counter = 0
#         # for i in unique_elements:
#         #     elements[pmesh.elements==i] = counter
#         #     counter += 1
#         # pmesh.elements = elements

#         counter = 0
#         for i in unique_edges:
#             # nodesDBC2D[counter] = whereEQ(nodesDBC,i)[0][0]
#             nodesDBC2D[counter] = np.where(nodesDBC==i)[0][0]
#             Dirichlet2D[counter,:] = Dirichlet[nodesDBC2D[counter],:]
#             counter += 1
#         nodesDBC2D = nodesDBC2D.astype(np.int64)

#         temp_dict = []
#         for i in nodesDBC[nodesDBC2D].flatten():
#             temp_dict.append(np.where(unique_elements==i)[0][0])
#         nodesDBC2D = np.array(temp_dict,copy=False)

#         pmesh.points = mesh.points[unique_elements,:]

#         one_element_coord = pmesh.points[pmesh.elements[0,:3],:]

#         # FOR COORDINATE TRANSFORMATION
#         AB = one_element_coord[0,:] - one_element_coord[1,:]
#         AC = one_element_coord[0,:] - one_element_coord[2,:]

#         normal = np.cross(AB,AC)
#         unit_normal = normal/np.linalg.norm(normal)

#         e1 = AB/np.linalg.norm(AB)
#         e2 = np.cross(normal,AB)/np.linalg.norm(np.cross(normal,AB))
#         e3 = unit_normal

#         # TRANSFORMATION MATRIX
#         Q = np.array([
#             [np.einsum('i,i',e1,E1), np.einsum('i,i',e1,E2), np.einsum('i,i',e1,E3)],
#             [np.einsum('i,i',e2,E1), np.einsum('i,i',e2,E2), np.einsum('i,i',e2,E3)],
#             [np.einsum('i,i',e3,E1), np.einsum('i,i',e3,E2), np.einsum('i,i',e3,E3)]
#             ])

#         pmesh.points = np.dot(pmesh.points,Q.T)
#         # assert np.allclose(pmesh.points[:,2],pmesh.points[0,2])
#         # z_plane = pmesh.points[0,2]

#         pmesh.points = pmesh.points[:,:2]

#         Dirichlet2D = np.dot(Dirichlet2D,Q.T)
#         Dirichlet2D = Dirichlet2D[:,:2]

#         pmesh.edges = None
#         pmesh.GetBoundaryEdgesTri()

#         # DEEP COPY BY SUBCLASSING
#         class MainData2D(MainData):
#             ndim = pmaterial.ndim
#             nvar = pmaterial.nvar
#             __PARALLEL__ = False

#         # FOR DYNAMICALLY PATCHED ITEMS
#         MainData2D.BoundaryData.IsDirichletComputed = True
#         MainData2D.BoundaryData.nodesDBC = nodesDBC2D
#         MainData2D.BoundaryData.Dirichlet = Dirichlet2D
#         MainData2D.MeshInfo.MeshType = "tri"

#         # COMPUTE BASES FOR TRIANGULAR ELEMENTS
#         QuadratureOpt = 3   # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
#         norder = MainData.C+MainData.C
#         if norder == 0:
#             # TAKE CARE OF C=0 CASE
#             norder = 1
#         MainData2D.Domain, MainData2D.Boundary, MainData2D.Quadrature = GetBasesAtInegrationPoints(MainData2D.C,
#             norder,QuadratureOpt,MainData.MeshInfo.MeshType)
#         # SEPARATELY COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
#         norder_post = (MainData.C+1)+(MainData.C+1)
#         MainData2D.PostDomain, MainData2D.PostBoundary, MainData2D.PostQuadrature = GetBasesAtInegrationPoints(MainData2D.C,
#             norder_post,QuadratureOpt,MainData2D.MeshInfo.MeshType)
        
        
#         print 'Solvingq planar problem number', niter, 'Number of DoF is', pmesh.points.shape[0]*MainData2D.nvar
#         if pmesh.points.shape[0] != Dirichlet2D.shape[0]:
#             # CALL THE MAIN SOLVER FOR SOLVING THE 2D PROBLEM
#             TotalDisp = MainSolver(MainData2D,pmesh,pmaterial)
#         else:
#             # IF THERE IS NO DEGREE OF FREEDOM TO SOLVE FOR (ONE ELEMENT CASE)
#             TotalDisp = Dirichlet2D[:,:,None]

#         Disp = np.zeros((TotalDisp.shape[0],3))
#         Disp[:,:2] = TotalDisp[:,:,-1]

#         temp_dict = []
#         for i in unique_elements:
#             temp_dict.append(np.where(nodesDBC==i)[0][0])

#         Dirichlet[temp_dict,:] = np.dot(Disp,Q)

#         if plot:
#             PostProcess.HighOrderCurvedPatchPlot(pmesh,TotalDisp,QuantityToPlot=MainData2D.ScaledJacobian,InterpolationDegree=40)
#             import matplotlib.pyplot as plt
#             plt.show()

#         del pmesh

#     gc.collect()

