# THIS FILE IS PART OF FLORENCE
from warnings import warn
from Florence.Base import SetPath
# import Core.MaterialLibrary as MatLib 
from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence.MeshGeneration.SalomeMeshReader import ReadMesh
from Florence.FiniteElements.GetBasesAtInegrationPoints import *
import Florence.Formulations.DisplacementElectricPotentialApproach as DEPB
import Florence.Formulations.DisplacementApproach as DB
from Florence import Mesh

from Florence.Supplementary.Timing import timing

# @profile
@timing
def PreProcess(MainData,mesh,material,Pr,pwd):

    # PARALLEL PROCESSING
    ############################################################################
    try:
        # CHECK IF MULTI-PROCESSING IS ACTIVATED
        MainData.__PARALLEL__
        MainData.Parallel = MainData.__PARALLEL__
    except NameError:
        # IF NOT THEN ASSUME SINGLE PROCESS
        MainData.Parallel = False
        MainData.numCPU = 1

    #############################################################################

    # READ MESH-FILE
    ############################################################################
    # mesh = Mesh()

    # MeshReader = getattr(mesh,MainData.MeshInfo.Reader,None)
    # if MeshReader is not None:
    #     if MainData.MeshInfo.Reader is 'Read': 
    #         if getattr(MainData.MeshInfo,'Format',None) is 'GID':
    #             mesh.ReadGIDMesh(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)
    #         else:   
    #             MeshReader(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)
    #     elif MainData.MeshInfo.Reader is 'ReadSeparate':
    #         # READ MESH FROM SEPARATE FILES FOR CONNECTIVITY AND COORDINATES
    #         mesh.ReadSeparate(MainData.MeshInfo.ConnectivityFile,MainData.MeshInfo.CoordinatesFile,MainData.MeshInfo.MeshType,
    #             delimiter_connectivity=',',delimiter_coordinates=',')
    #         # mesh.ReadSeparate(MainData.MeshInfo.ConnectivityFile,MainData.MeshInfo.CoordinatesFile,MainData.MeshInfo.MeshType,
    #         #   edges_file=MainData.MeshInfo.EdgesFile,delimiter_connectivity=',',delimiter_coordinates=',')
    #     elif MainData.MeshInfo.Reader is 'ReadHighOrderMesh':
    #         mesh.ReadHighOrderMesh(MainData.MeshInfo.FileName.split(".")[0],MainData.C,MainData.MeshInfo.MeshType)
    #     elif MainData.MeshInfo.Reader is 'ReadHDF5':
    #         mesh.ReadHDF5(MainData.MeshInfo.FileName)
    #     elif MainData.MeshInfo.Reader is 'UniformHollowCircle':
    #         # mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=4,ncirc=12)
    #         # mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=7,ncirc=7) # isotropic
    #         mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=False,nrad=7,ncirc=7)
    #     elif MainData.MeshInfo.Reader is 'Sphere':
    #         # mesh.Sphere()
    #         # mesh.Sphere(points=200)
    #         mesh.Sphere(points=10)

    if MainData.__NO_DEBUG__ is False:
        mesh.CheckNodeNumbering()

    
    if 'MechanicalComponent2D' in Pr.__file__.split('/') or \
        'Misc' in Pr.__file__.split('/'):
        mesh.points *=1000.
        # mesh.points /=1000.

    # mesh.points *=1000. 

    # mesh.SimplePlot()
    # mesh.PlotMeshNumbering()

    # print mesh.GetElementsWithBoundaryEdgesTri()
    # mesh.RetainElementsWithin((-0.52,-0.08,0.72,0.08))
    # mesh.RetainElementsWithin((-0.502,-0.06,0.505,0.06283))
    # mesh.RemoveElements((-0.9,-0.1,1.9,0.1),keep_boundary_only=True)
    # mesh.RemoveElements((-0.9,-0.1,1.9,0.1),keep_boundary_only=True,plot_new_mesh=False) #
    # mesh.RemoveElements((-0.6,-0.1,1.9,0.6),keep_boundary_only=True,plot_new_mesh=False)
    # mesh.RemoveElements((-0.55,-0.1,-0.4,0.1),plot_new_mesh=False) 


    # un_faces = np.unique(mesh.faces)
    # vpoints = mesh.points[un_faces,:]
    # print np.linalg.norm(vpoints,axis=1)
    # exit()

    # mesh.Median
    # from Core.MixedFormulations.MixedFormulations import NearlyIncompressibleHuWashizu
    # xx = NearlyIncompressibleHuWashizu(mesh)
    # exit()


    # from scipy.io import loadmat
    # loadedmat = loadmat(MainData.MeshInfo.MatFile)
    # mesh.points = np.ascontiguousarray(loadedmat['X'])
    # mesh.elements = np.ascontiguousarray(loadedmat['T'])-1


    # STORE PATHS FOR MAIN, CORE & PROBLEM DIRECTORIES
    ############################################################################
    MainData.Path = SetPath(Pr,pwd,MainData.C,mesh.nelem,
        MainData.Analysis,MainData.AnalysisType,material.GetType())

    # GENERATE pMESHES FOR HIGH C
    ############################################################################

    # IsHighOrder = getattr(MainData.MeshInfo,"IsHighOrder",False)
    IsHighOrder = False
    if mesh.InferPolynomialDegree() > 1:
        IsHighOrder = True
    
    if MainData.C>0:
        if IsHighOrder is False:
            mesh.GetHighOrderMesh(MainData.C,Parallel=MainData.Parallel,
                nCPU=MainData.numCPU,ComputeAll=True)
    else:
        mesh.ChangeType()

    ############################################################################
    # from Core.Supplementary.Tensors import makezero, unique2d
    # un_faces =  unique2d(mesh.faces,consider_sort=False)
    # print un_faces.shape, mesh.faces.shape
    # ##########################################################################

    # mesh.PlotMeshNumbering()
    # mesh.SimplePlot()
    # exit()

    # print MainData.MeshInfo.FileName.split(".")[0]+"_elements_"+"P"+str(MainData.C+1)+".dat"
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_elements_"+"P"+str(MainData.C+1)+".dat",mesh.elements,delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_points_"+"P"+str(MainData.C+1)+".dat",mesh.points,fmt="%9.16f",delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_edges_"+"P"+str(MainData.C+1)+".dat",mesh.edges,delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_faces_"+"P"+str(MainData.C+1)+".dat",mesh.faces,delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_face_to_surface_"+"P"+str(MainData.C+1)+".dat",mesh.face_to_surface,delimiter=',')
    # exit()


    # # For saving 3D problems
    # from time import time
    # tt = time()
    # from scipy.io import savemat
    # mesh.GetFacesTet()
    # mesh.GetEdgesTet()
    # face_flags = mesh.GetInteriorFacesTet()
    # mesh.GetElementsFaceNumberingTet()
    # boundary_face_to_element = mesh.GetElementsWithBoundaryFacesTet()

    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 'faces':mesh.faces,
    #     'edges':mesh.edges, 'all_faces':mesh.all_faces, 'all_edges':mesh.all_edges,
    #     'face_flags':face_flags,'face_to_element':mesh.face_to_element,
    #     'boundary_face_to_element':boundary_face_to_element}
    # # Dict['face_to_surface'] = mesh.face_to_surface
    # # savemat(MainData.MeshInfo.FileName.split(".")[0]+"_P"+str(MainData.C+1)+".mat",Dict,do_compression=True)
    # savemat(mesh.filename.split(".")[0]+"_P"+str(MainData.C+1)+".mat",Dict,do_compression=True)
    # # savemat(MainData.MeshInfo.FileName.split(".")[0]+"_P"+str(MainData.C+1)+"_New.mat",Dict,do_compression=True)
    # print mesh.filename.split(".")[0]+"_P"+str(MainData.C+1)+".mat"
    # print 'rest of the time', time() - tt
    # # exit()



    # # For saving 2D problems
    # from time import time
    # tt = time()
    # from scipy.io import savemat
    # mesh.GetEdgesTri()

    # Dict = {'points':mesh.points, 'elements':mesh.elements, 
    #     'element_type':mesh.element_type, 
    #     'edges':mesh.edges, 'all_edges':mesh.all_edges}
    # # Dict['face_to_surface'] = mesh.face_to_surface
    # savemat(MainData.MeshInfo.FileName.split(".")[0]+"_P"+str(MainData.C+1)+".mat",Dict,do_compression=True)
    # print MainData.MeshInfo.FileName.split(".")[0]+"_P"+str(MainData.C+1)+".mat"
    # print 'rest of the time', time() - tt
    # # exit()


    # For F6
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6_iso_face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6BL_face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Drill/face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Valve/face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/face_to_surface_mapped.dat").astype(np.int64)
    # face_to_surface = np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/face_to_surface_mapped.dat").astype(np.int64)
    # mesh.face_to_surface = face_to_surface


    # Falcon
    # lmesh = Mesh()
    # lmesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso.dat","tet")
    # lmesh.ReadGIDMesh("/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/falcon/falcon_big.dat","tet")
    # mesh.face_to_surface = lmesh.face_to_surface
    # print mesh.face_to_surface
    # exit()

    # print "The linear mesh has", 2*mesh.elements.shape[0], "elements", 2*np.unique(mesh.elements[:,:4]).shape[0], \
    # "vertices and", 2*mesh.faces[MainData.BoundaryData().ProjectionCriteria(mesh).flatten()==1,:].shape[0], "triangular faces."
    # print "The resulting high order mesh with p=", (MainData.C+1), "has", 2*mesh.points.shape[0]-yy, "nodes and",  \
    # 2*np.unique(mesh.faces[MainData.BoundaryData().ProjectionCriteria(mesh).flatten()==1,:]).shape[0], "nodes on the"

    # aspect = mesh.AspectRatios
    # print aspect.min(), aspect.max()
    # exit()

    # MainData.MaterialArgs.AnisotropicOrientations = np.zeros((mesh.elements.shape[0],3))
    # MainData.MaterialArgs.AnisotropicOrientations[:,0] = 1.0




    # COMPARE STRINGS WHICH MIGHT CONTAIN UNICODES
    ############################################################################
    if getattr(str,'casefold',None) is not None:
        insensitive = lambda str_name: str_name.casefold()
    else:
        insensitive = lambda str_name: str_name.upper().lower()




    # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR ALL ELEMENTAL INTEGRATON
    ############################################################################
    # FOR DISPLACEMENT-BASED FORMULATIONS (GRADIENT-GRADIENT) WE NEED (P-1)+(P-1) TO EXACTLY
    # INTEGRATE THE INTEGRANDS
    QuadratureOpt = 3   # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
    norder = MainData.C+MainData.C
    if norder == 0:
        # TAKE CARE OF C=0 CASE
        norder = 1
    MainData.Domain, MainData.Boundary, MainData.Quadrature = GetBasesAtInegrationPoints(MainData.C,
        norder,QuadratureOpt,mesh.element_type)
    # SEPARATELY COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
    # E.G. FOR COMPUTATION OF SCALED JACOBIAN. NOTE THAT THIS SHOULD ONLY BE USED FOR POST PROCESSING
    # FOR ELEMENTAL INTEGRATION ALWAYS USE DOMIAN, BOUNDARY AND QUADRATURE AND NOT POSTDOMAIN, 
    # POSTBOUNDARY ETC
    # FOR SCALED JACOBIAN WE NEED QUADRATURE FOR P*P 
    norder_post = (MainData.C+1)+(MainData.C+1)
    MainData.PostDomain, MainData.PostBoundary, MainData.PostQuadrature = GetBasesAtInegrationPoints(MainData.C,
        norder_post,QuadratureOpt,mesh.element_type)

    ############################################################################






    #############################################################################
                            # MATERIAL MODEL PRE-PROCESS
    #############################################################################
    #############################################################################

    # CHECK IF MATERIAL MODEL AND ANALYSIS TYPE ARE COMPATIBLE
    #############################################################################
    if "nonlinear" in insensitive(MainData.AnalysisType):
        if "linear" in  insensitive(material.mtype) or \
            "increment" in insensitive(material.mtype):
            warn("Incompatible material model and analysis type. I'm going to change analysis type")
            MainData.AnalysisType = "Linear"


    # COMPUTE 4TH ORDER IDENTITY TENSORS/HESSIANS BEFORE-HAND 
    # #############################################################################
    # if material.mtype == 'LinearModel' or \
    #     material.mtype == 'IncrementalLinearElastic':

    #     if MainData.ndim == 2:
    #         MainData.MaterialArgs.H_Voigt = MainData.MaterialArgs.lamb*np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]]) +\
    #          MainData.MaterialArgs.mu*np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
    #     else:
    #         block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
    #         block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
    #         MainData.MaterialArgs.H_Voigt = MainData.MaterialArgs.lamb*block_1 + MainData.MaterialArgs.mu*block_2
    # else:
    #     if MainData.ndim == 2:
    #         MainData.MaterialArgs.vIijIkl = np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]])
    #         MainData.MaterialArgs.vIikIjl = np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
    #     else:
    #         block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
    #         block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
    #         MainData.MaterialArgs.vIijIkl = block_1
    #         MainData.MaterialArgs.vIikIjl = block_2 

    #     I = np.eye(MainData.ndim,MainData.ndim)
    #     MainData.MaterialArgs.Iijkl = np.einsum('ij,kl',I,I)    
    #     MainData.MaterialArgs.Iikjl = np.einsum('ik,jl',I,I) + np.einsum('il,jk',I,I) 


    # # COMPUTE ANISOTROPY DIRECTIONS/STRUCTURAL TENSORS FOR ANISOTROPIC MATERIAL
    # ##############################################################################
    # # THIS HAS TO BE CALLED BEFORE CALLING THE ANY HESSIAN METHODS AS HESSIAN OF 
    # # ANISOTROPIC MODELS REQUIRE FIBRE ORIENTATION
    # AnisoFuncName = getattr(MainData.MaterialArgs,"AnisotropicFibreOrientation",None)
    # if AnisoFuncName is not None:
    #     aniso_obj = AnisoFuncName(mesh,plot=False)
    #     material.AnisotropicOrientations = aniso_obj.directions
    # if material.mtype == "BonetTranservselyIsotropicHyperElastic"\
    # and AnisoFuncName is None:
    #     MainData.MaterialArgs.AnisotropicOrientations = np.zeros((mesh.nelem,MainData.ndim))
    #     MainData.MaterialArgs.AnisotropicOrientations[:,0] = 1.0

    if material.is_transversely_isotropic:
        material.GetFibresOrientation(mesh)
    ##############################################################################
    MainData.nvar = material.nvar
    # MainData.boundary_condition.SetProjectionCriteria(ProjectionCriteria,mesh)


    # # CHOOSE AND INITIALISE THE RIGHT MATERIAL MODEL 
    # ##############################################################################

    # # GET THE MEHTOD NAME FOR THE RIGHT MATERIAL MODEL
    # MaterialFuncName = getattr(MatLib,MainData.MaterialArgs.Type,None)
    # if MaterialFuncName is not None:
    #     # INITIATE THE FUNCTIONS FROM THIS MEHTOD
    #     # MaterialInstance = MaterialFuncName(MainData.ndim,MainData.MaterialArgs)

    #     kwargs = MainData.MaterialArgs.__dict__
    #     material = MaterialFuncName(MainData.MaterialArgs.Type, MainData.ndim, **kwargs)
    #     MainData.nvar, MainData.MaterialModelName = material.nvar, \
    #                                                 type(material).__name__


    #     # MainData.Hessian = material.Hessian     
    #     # MainData.CauchyStress = material.CauchyStress

    #     # INITIALISE
    #     StrainTensors = KinematicMeasures(np.asarray([np.eye(MainData.ndim,MainData.ndim)]*\
    #         MainData.Domain.AllGauss.shape[0]),MainData.AnalysisType)


    #     # MaterialInstance.Hessian(MainData.MaterialArgs,
    #     #     StrainTensors,elem=0,gcounter=0)
    #     # MainData.MaterialArgs.H_VoigtSize = 
    #     material.Hessian(StrainTensors,elem=0,gcounter=0)

    # MainData.material = material

    # else:
    #     raise AttributeError('Material model with name '+MainData.MaterialArgs.Type + ' not found')

    ##############################################################################



    # FORMULATION TYPE FLAGS
    #############################################################################   
    if MainData.Formulation == 'DisplacementApproach':
        MainData.ConstitutiveStiffnessIntegrand = DB.ConstitutiveStiffnessIntegrand
        MainData.GeometricStiffnessIntegrand = DB.GeometricStiffnessIntegrand
        MainData.MassIntegrand =  DB.MassIntegrand
    
    elif MainData.Formulation == 'DisplacementElectricPotentialApproach':
        MainData.ConstitutiveStiffnessIntegrand = DEPB.ConstitutiveStiffnessIntegrand
        MainData.GeometricStiffnessIntegrand = DEPB.GeometricStiffnessIntegrand
        MainData.MassIntegrand =  DEPB.MassIntegrand






    # STRESS COMPUTATION FLAGS FOR LINEARISED ELASTICITY
    ###########################################################################
    MainData.Prestress = 0
    if "nonlinear" not in insensitive(MainData.AnalysisType) and MainData.Fields == "Mechanics":
        # RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE
        if material.mtype != "IncrementalLinearElastic" and \
            material.mtype != "LinearModel":
            MainData.Prestress = 1
        else:
            MainData.Prestress = 0

    # GEOMETRY UPDATE FLAGS
    ###########################################################################
    # DO NOT UPDATE THE GEOMETRY IF THE MATERIAL MODEL NAME CONTAINS 
    # INCREMENT (CASE INSENSITIVE). VALID FOR ELECTROMECHANICS FORMULATION. 
    MainData.GeometryUpdate = 0
    if MainData.Fields == "ElectroMechanics":
        if "Increment" in insensitive(MainData.MaterialArgs.Type):
            # RUN THE SIMULATION WITHIN A NONLINEAR ROUTINE WITHOUT UPDATING THE GEOMETRY
            MainData.GeometryUpdate = 0
        else:
            MainData.GeometryUpdate = 1
    elif MainData.Fields == "Mechanics":
        if MainData.AnalysisType == "Nonlinear" or MainData.Prestress:
            MainData.GeometryUpdate = 1








    # CHOOSING THE SOLVER/ASSEMBLY ROUTINES BASED ON PROBLEM SIZE
    #############################################################################

    # if mesh.points.shape[0]*MainData.nvar > 100000:
    #     MainData.solver.solver_type = "multigrid"
    #     MainData.solver.solver_subtype = "amg"
    #     print 'Large system of equations. Switching to algebraic multigrid solver'
    # # elif mesh.points.shape[0]*MainData.nvar > 50000 and MainData.C < 4:
    #     # solve.type = "direct"
    #     # solve.sub_type = "MUMPS"
    #     # print 'Large system of equations. Switching to MUMPS solver'
    # else:
    #     MainData.solver.solver_type = "direct"
    #     MainData.solver.solver_subtype = "umfpack"

    # solve.type = "direct"
    # solve.sub_type = "UMFPACK"

    # solve.type = "multigrid"
    # solve.sub_type = "amg"
    
    # solve.type = "direct"
    # solve.sub_type = "MUMPS"



    # class solve(object):
    #     tol = 5.0e-07

    # if mesh.points.shape[0]*MainData.nvar > 100000:
    #     solve.type = "multigrid"
    #     solve.sub_type = "amg"
    #     print 'Large system of equations. Switching to algebraic multigrid solver'
    # # elif mesh.points.shape[0]*MainData.nvar > 50000 and MainData.C < 4:
    #     # solve.type = "direct"
    #     # solve.sub_type = "MUMPS"
    #     # print 'Large system of equations. Switching to MUMPS solver'
    # else:
    #     solve.type = "direct"
    #     solve.sub_type = "UMFPACK"

    # # solve.type = "direct"
    # # solve.sub_type = "UMFPACK"

    # # solve.type = "multigrid"
    # # solve.sub_type = "amg"
    
    # # solve.type = "direct"
    # # solve.sub_type = "MUMPS"

    # MainData.solve = solve 
            

    if mesh.nelem > 1e07:
        MainData.AssemblyRoutine = 'Large'
        print 'Large number of elements. Switching to out of core assembly routine'
    else:
        MainData.AssemblyRoutine = 'Small'
        # print 'Small number of elements. Sticking to small assembly routine'

    # FORCE QUIT PARALLELISATION  
    if mesh.elements.shape[0] < 200:
        MainData.__PARALLEL__ = False
        MainData.Parallel = False
        MainData.numCPU = 1


    #############################################################################




    #############################################################################
    MainData.IsSymmetricityComputed = False


    #############################################################################



    # DICTIONARY OF SAVED VARIABLES
    #############################################################################
    if MainData.write == 1:
        MainData.MainDict = {'ProblemPath': MainData.Path.ProblemResults, 
         'MeshPoints':mesh.points, 'MeshElements':mesh.elements, 'MeshFaces':mesh.faces, 'MeshEdges':mesh.edges,
         'Solution':[], 'DeformationGradient':[], 'CauchyStress':[],
         'SecondPiolaStress':[], 'ElectricField':[], 'ElectricDisplacement':[]}

    #############################################################################



    return mesh




