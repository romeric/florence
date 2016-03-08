# THIS FILE IS PART OF FLORENCE
from warnings import warn
from Florence.Base import SetPath
from Florence import QuadratureRule
from Florence import FunctionSpace
from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Florence.MeshGeneration.SalomeMeshReader import ReadMesh
import Florence.Formulations.DisplacementElectricPotentialApproach as DEPB
import Florence.Formulations.DisplacementApproach as DB
from Florence.Utils import insensitive
from Florence import Mesh

from Florence.Supplementary.Timing import timing

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

    # mesh.Median
    # from Core.MixedFormulations.MixedFormulations import NearlyIncompressibleHuWashizu
    # xx = NearlyIncompressibleHuWashizu(mesh)
    # print mesh.elements.shape, mesh.points.shape
    # exit()


    # from scipy.io import loadmat
    # loadedmat = loadmat(MainData.MeshInfo.MatFile)
    # mesh.points = np.ascontiguousarray(loadedmat['X'])
    # mesh.elements = np.ascontiguousarray(loadedmat['T'])-1

    # del mesh
    # mesh = Mesh()
    # mesh.ReadGmsh("/home/roman/Dropbox/florence/Examples/FiniteElements/Misc3D/hand.mesh")
    # mesh.element_type = "tet"
    # mesh.points = np.loadtxt("/home/roman/Dropbox/florence/Examples/FiniteElements/Misc3D/hand_mesh_points.dat")[:,:3]
    # mesh.elements = np.loadtxt("/home/roman/Dropbox/florence/Examples/FiniteElements/Misc3D/hand_mesh_elements.dat")[:,:4].astype(np.int64) - 1
    # # mesh.faces = np.loadtxt("/home/roman/Dropbox/florence/Examples/FiniteElements/Misc3D/hand_mesh_elements.dat")[:,:3].astype(np.int64) - 1
    # mesh.GetBoundaryFacesTet()
    # mesh.SimplePlot()
    # exit()


    # STORE PATHS FOR MAIN, CORE & PROBLEM DIRECTORIES
    ############################################################################
    MainData.Path = SetPath(Pr,pwd,MainData.C,mesh.nelem,
        MainData.Analysis,MainData.AnalysisType,material.GetType())

    # GENERATE pMESHES FOR HIGH C
    # ############################################################################

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

    # mesh.PlotMeshNumbering()
    # mesh.SimplePlot()
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




    # from Florence.VariationalPrinciple import NearlyIncompressibleHuWashizu
    # formulation = NearlyIncompressibleHuWashizu(mesh)
    # print dir(formulation)
    # exit()




    # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR ALL ELEMENTAL INTEGRATON
    ############################################################################
    # FOR DISPLACEMENT-BASED FORMULATIONS (GRADIENT-GRADIENT) WE NEED (P-1)+(P-1) TO EXACTLY
    # INTEGRATE THE INTEGRANDS
    QuadratureOpt = 3   # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
    norder = MainData.C+MainData.C
    if norder == 0:
        # TAKE CARE OF C=0 CASE
        norder = 1
    # GET QUADRATURE
    quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder, mesh_type=mesh.element_type)
    MainData.quadrature = quadrature

    function_space = FunctionSpace(mesh, quadrature, p=MainData.C+1)
    MainData.Domain, MainData.Boundary = function_space, function_space.Boundary
    # SEPARATELY COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
    # E.G. FOR COMPUTATION OF SCALED JACOBIAN. NOTE THAT THIS SHOULD ONLY BE USED FOR POST PROCESSING
    # FOR ELEMENTAL INTEGRATION ALWAYS USE DOMIAN, BOUNDARY AND QUADRATURE AND NOT POSTDOMAIN, 
    # POSTBOUNDARY ETC
    # FOR SCALED JACOBIAN WE NEED QUADRATURE FOR P*P 
    norder_post = (MainData.C+1)+(MainData.C+1)
    post_quadrature = QuadratureRule(optimal=QuadratureOpt, norder=norder_post, mesh_type=mesh.element_type)

    function_space = FunctionSpace(mesh, post_quadrature, p=MainData.C+1)
    MainData.PostDomain, MainData.PostBoundary = function_space, function_space.Boundary
    MainData.post_quadrature = post_quadrature

    ############################################################################

    # from Florence.FunctionSpace.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints
    # # COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR ALL ELEMENTAL INTEGRATON
    # ############################################################################
    # # FOR DISPLACEMENT-BASED FORMULATIONS (GRADIENT-GRADIENT) WE NEED (P-1)+(P-1) TO EXACTLY
    # # INTEGRATE THE INTEGRANDS
    # QuadratureOpt = 3   # OPTION FOR QUADRATURE TECHNIQUE FOR TRIS AND TETS
    # norder = MainData.C+MainData.C
    # if norder == 0:
    #     # TAKE CARE OF C=0 CASE
    #     norder = 1
    # MainData.Domain, MainData.Boundary, MainData.Quadrature = GetBasesAtInegrationPoints(MainData.C,
    #     norder,QuadratureOpt,mesh.element_type)
    # # SEPARATELY COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
    # # E.G. FOR COMPUTATION OF SCALED JACOBIAN. NOTE THAT THIS SHOULD ONLY BE USED FOR POST PROCESSING
    # # FOR ELEMENTAL INTEGRATION ALWAYS USE DOMIAN, BOUNDARY AND QUADRATURE AND NOT POSTDOMAIN, 
    # # POSTBOUNDARY ETC
    # # FOR SCALED JACOBIAN WE NEED QUADRATURE FOR P*P 
    # norder_post = (MainData.C+1)+(MainData.C+1)
    # MainData.PostDomain, MainData.PostBoundary, MainData.PostQuadrature = GetBasesAtInegrationPoints(MainData.C,
    #     norder_post,QuadratureOpt,mesh.element_type)

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

    if material.is_transversely_isotropic:
        material.GetFibresOrientation(mesh)
    ##############################################################################
    MainData.nvar = material.nvar

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
    MainData.IsSymmetricityComputed = False


    # DICTIONARY OF SAVED VARIABLES
    #############################################################################
    if MainData.write == 1:
        MainData.MainDict = {'ProblemPath': MainData.Path.ProblemResults, 
         'MeshPoints':mesh.points, 'MeshElements':mesh.elements, 'MeshFaces':mesh.faces, 'MeshEdges':mesh.edges,
         'Solution':[], 'DeformationGradient':[], 'CauchyStress':[],
         'SecondPiolaStress':[], 'ElectricField':[], 'ElectricDisplacement':[]}

    #############################################################################
