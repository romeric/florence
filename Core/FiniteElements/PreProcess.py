# THIS FILE IS PART OF FLORENCE
from warnings import warn
from Base import SetPath
import Core.MaterialLibrary as MatLib 
from Core.FiniteElements.ElementalMatrices.KinematicMeasures import *
from Core.MeshGeneration.SalomeMeshReader import ReadMesh
from Core.FiniteElements.GetBasesAtInegrationPoints import *
import Core.Formulations.DisplacementElectricPotentialApproach as DEPB
import Core.Formulations.DisplacementApproach as DB
from Core import Mesh

from Core.Supplementary.Timing.Timing import timing

@timing
def PreProcess(MainData,Pr,pwd):


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
    mesh = Mesh()

    MeshReader = getattr(mesh,MainData.MeshInfo.Reader,None)
    if MeshReader is not None:
        if MainData.MeshInfo.Reader is 'Read':
            if getattr(MainData.MeshInfo,'Format',None) is 'GID':
                mesh.ReadGIDMesh(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)
            else:   
                MeshReader(MainData.MeshInfo.FileName,MainData.MeshInfo.MeshType,MainData.C)
        elif MainData.MeshInfo.Reader is 'ReadSeparate':
            # READ MESH FROM SEPARATE FILES FOR CONNECTIVITY AND COORDINATES
            mesh.ReadSeparate(MainData.MeshInfo.ConnectivityFile,MainData.MeshInfo.CoordinatesFile,MainData.MeshInfo.MeshType,
                delimiter_connectivity=',',delimiter_coordinates=',')
            # mesh.ReadSeparate(MainData.MeshInfo.ConnectivityFile,MainData.MeshInfo.CoordinatesFile,MainData.MeshInfo.MeshType,
            #   edges_file=MainData.MeshInfo.EdgesFile,delimiter_connectivity=',',delimiter_coordinates=',')
        elif MainData.MeshInfo.Reader is 'UniformHollowCircle':
            # mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=4,ncirc=12)
            # mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=7,ncirc=7) # isotropic
            mesh.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=False,nrad=7,ncirc=7)
        elif MainData.MeshInfo.Reader is 'ReadHighOrderMesh':
            mesh.ReadHighOrderMesh(MainData.MeshInfo.FileName.split(".")[0],MainData.C,MainData.MeshInfo.MeshType)
        elif MainData.MeshInfo.Reader is 'Sphere':
            # mesh.Sphere()
            # mesh.Sphere(points=10)
            mesh.Sphere(points=4)
            # mesh.SimplePlot()

    if MainData.__NO_DEBUG__ is False:
        mesh.CheckNodeNumbering()

    # mesh.ReadGIDMesh("/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/falcon/falcon_iso.dat","tet",0)
    # mesh.ReadGIDMesh("/home/roman/Dropbox/2015_HighOrderMeshing/geometriesAndMeshes/almond/almond_H1.dat","tet",0)

    # np.savetxt('/home/roman/Desktop/elements_falcon.dat', mesh.elements,fmt='%d',delimiter=',')
    # np.savetxt('/home/roman/Desktop/points_falcon.dat', mesh.points,fmt='%10.9f',delimiter=',')

    
    if 'MechanicalComponent2D' in Pr.__file__.split('/') or \
        'Misc' in Pr.__file__.split('/'):
        mesh.points *=1000.
        # mesh.points /=1000.


    # mesh.points *=1000. 
    # mesh.SimplePlot()
    # mesh.PlotMeshNumberingTri()
    # print mesh.GetElementsWithBoundaryEdgesTri()
    # mesh.RetainElementsWithin((-0.52,-0.08,0.72,0.08))
    # mesh.RetainElementsWithin((-0.502,-0.06,0.505,0.06283))
    # mesh.RemoveElements((-0.9,-0.1,1.9,0.1),keep_boundary_only=True)
    # mesh.RemoveElements((-0.9,-0.1,1.9,0.1),keep_boundary_only=True,plot_new_mesh=False) #

    # mesh.RemoveElements((-0.6,-0.1,1.9,0.6),keep_boundary_only=True,plot_new_mesh=False)

    # mesh.SimplePlot()
    # mesh.SimplePlot(save=True,filename="/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Mesh_Stretch_25")
    # mesh.SimplePlot(save=True,filename="/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Mesh_Stretch_200")
    # mesh.SimplePlot(save=True,filename="/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Mesh_Stretch_1600")
    # mesh.PlotMeshNumberingTri()
    

    # mesh.RemoveElements((-0.55,-0.1,-0.4,0.1),plot_new_mesh=False) 
    # mesh.SimplePlot() 
    # mesh.PlotMeshNumberingTri()

    # un_faces = np.unique(mesh.faces)
    # vpoints = mesh.points[un_faces,:]
    # print np.linalg.norm(vpoints,axis=1)
    # exit()



    # from scipy.io import loadmat
    # loadedmat = loadmat(MainData.MeshInfo.MatFile)
    # mesh.points = np.ascontiguousarray(loadedmat['X'])
    # mesh.elements = np.ascontiguousarray(loadedmat['T'])-1


    # print mesh.nelem, mesh.points.shape[0], mesh.edges.shape[0]
    # mesh.WriteVTK(fname="/home/roman/Dropbox/dd2.vtu")
    # print mesh.faces
    # print mesh.points

    # print mesh.GetElementsWithBoundaryFacesTet() - mesh.ArrangeFacesTet()
    # print 
    # mesh.ArrangeFacesTet()

    # print mesh.GetFacesTet()
    # mesh.GetElementsFaceNumberingTet()
    # mesh.ArrangeFacesTet()
    # mesh.GetEdgesTri()
    # mesh.GetInteriorEdgesTri()
    # mesh.GetInteriorFacesTet()
    # mesh.CheckNodeNumbering()

    # print 
    # mesh.GetElementsWithBoundaryFacesTet()
    # exit()

    # STORE PATHS FOR MAIN, CORE & PROBLEM DIRECTORIES
    ############################################################################
    MainData.Path = SetPath(Pr,pwd,MainData.C,mesh.nelem,
        MainData.Analysis,MainData.AnalysisType,MainData.MaterialArgs.Type)

    # GENERATE pMESHES FOR HIGH C
    ############################################################################

    IsHighOrder = getattr(MainData.MeshInfo,"IsHighOrder",None)
    if IsHighOrder is None:
        IsHighOrder = False

    if MainData.C>0:
        if IsHighOrder is False:
            mesh.GetHighOrderMesh(MainData.C,Parallel=MainData.Parallel,
                nCPU=MainData.numCPU,ComputeAll=True)
    else:
        mesh.ChangeType()

    ############################################################################
    # t1=time()
    # mesh.GetElementsWithBoundaryEdgesTri()
    # print time()-t1

    # index_sort_x = np.argsort(nmesh.points[:,0])
    # sorted_repoints = nmesh.points[index_sort_x,:]
    # ##############################################################################
    # np.savetxt('/home/roman/Dropbox/time_3.dat',np.array([time()-t_mesh, mesh.points.shape[0]]))

    # mesh.PlotMeshNumberingTri()

    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_elements_"+"P"+str(MainData.C+1)+".dat",mesh.elements,delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_points_"+"P"+str(MainData.C+1)+".dat",mesh.points,fmt="%9.16f",delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_edges_"+"P"+str(MainData.C+1)+".dat",mesh.edges,delimiter=',')
    # np.savetxt(MainData.MeshInfo.FileName.split(".")[0]+"_faces_"+"P"+str(MainData.C+1)+".dat",mesh.faces,delimiter=',')


    # print mesh.elements.shape, mesh.points.shape, mesh.edges.shape
    # print np.min(mesh.points[:,0]), np.min(mesh.points[:,1]), np.min(mesh.points[:,2]) 
    # print np.max(mesh.points[:,0]), np.max(mesh.points[:,1]), np.max(mesh.points[:,2]) 
    # print mesh.elements
    # print mesh.faces
    # print mesh.edges.shape
    # print mesh.points

    # un_faces = np.unique(mesh.faces)
    # vpoints = mesh.points[un_faces,:]
    # print np.linalg.norm(vpoints,axis=1)

    # mesh.GetFaceFlagsTets()
    # mesh.GetFacesTet()
    # mesh.GetBoundaryFacesTet()
    # mesh.GetBoundaryEdgesTet()
    # mesh.GetInteriorFacesTet()
    # mesh.GetEdgesTet()
    # print mesh.all_edges
    # print mesh.all_faces.shape
    
    # mesh.GetEdgesTri()
    # mesh.GetBoundaryEdgesTri()
    # mesh.GetInteriorEdgesTri()

    # exit()



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
        norder,QuadratureOpt,MainData.MeshInfo.MeshType)
    # SEPARATELY COMPUTE INTERPOLATION FUNCTIONS AT ALL INTEGRATION POINTS FOR POST-PROCESSING
    # E.G. FOR COMPUTATION OF SCALED JACOBIAN. NOTE THAT THIS SHOULD ONLY BE USED FOR POST PROCESSING
    # FOR ELEMENTAL INTEGRATION ALWAYS USE DOMIAN, BOUNDARY AND QUADRATURE AND NOT POSTDOMAIN, 
    # POSTBOUNDARY ETC
    # FOR SCALED JACOBIAN WE NEED QUADRATURE FOR P*P 
    norder_post = (MainData.C+1)+(MainData.C+1)
    MainData.PostDomain, MainData.PostBoundary, MainData.PostQuadrature = GetBasesAtInegrationPoints(MainData.C,
        norder_post,QuadratureOpt,MainData.MeshInfo.MeshType)

    ############################################################################






    #############################################################################
                            # MATERIAL MODEL PRE-PROCESS
    #############################################################################
    #############################################################################

    # CHECK IF MATERIAL MODEL AND ANALYSIS TYPE ARE COMPATIBLE
    #############################################################################
    if "nonlinear" in insensitive(MainData.AnalysisType):
        if "linear" in  insensitive(MainData.MaterialArgs.Type) or \
            "increment" in insensitive(MainData.MaterialArgs.Type):
            warn("Incompatible material model and analysis type. I'm going to change analysis type")
            MainData.AnalysisType = "Linear"


    # COMPUTE 4TH ORDER IDENTITY TENSORS/HESSIANS BEFORE-HAND 
    #############################################################################
    if MainData.MaterialArgs.Type == 'LinearModel' or \
        MainData.MaterialArgs.Type == 'IncrementalLinearElastic':

        if MainData.ndim == 2:
            MainData.MaterialArgs.H_Voigt = MainData.MaterialArgs.lamb*np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]]) +\
             MainData.MaterialArgs.mu*np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
        else:
            block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
            block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
            MainData.MaterialArgs.H_Voigt = MainData.MaterialArgs.lamb*block_1 + MainData.MaterialArgs.mu*block_2
    else:
        if MainData.ndim == 2:
            MainData.MaterialArgs.vIijIkl = np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]])
            MainData.MaterialArgs.vIikIjl = np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
        else:
            block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
            block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
            MainData.MaterialArgs.vIijIkl = block_1
            MainData.MaterialArgs.vIikIjl = block_2 

        I = np.eye(MainData.ndim,MainData.ndim)
        MainData.MaterialArgs.Iijkl = np.einsum('ij,kl',I,I)    
        MainData.MaterialArgs.Iikjl = np.einsum('ik,jl',I,I) + np.einsum('il,jk',I,I) 


    # COMPUTE ANISOTROPY DIRECTIONS/STRUCTURAL TENSORS FOR ANISOTROPIC MATERIAL
    ##############################################################################
    # THIS HAS TO BE CALLED BEFORE CALLING THE ANY HESSIAN METHODS AS HESSIAN OF 
    # ANISOTROPIC MODELS REQUIRE FIBRE ORIENTATION
    AnisoFuncName = getattr(MainData.MaterialArgs,"AnisotropicFibreOrientation",None)
    if AnisoFuncName is not None:
        aniso_obj = AnisoFuncName(mesh,plot=False)
        MainData.MaterialArgs.AnisotropicOrientations = aniso_obj.directions


    # CHOOSE AND INITIALISE THE RIGHT MATERIAL MODEL 
    ##############################################################################

    # GET THE MEHTOD NAME FOR THE RIGHT MATERIAL MODEL
    MaterialFuncName = getattr(MatLib,MainData.MaterialArgs.Type,None)
    if MaterialFuncName is not None:
        # INITIATE THE FUNCTIONS FROM THIS MEHTOD
        MaterialInstance = MaterialFuncName(MainData.ndim,MainData.MaterialArgs)
        MainData.nvar, MainData.MaterialModelName = MaterialInstance.nvar, \
                                                    type(MaterialInstance).__name__

        MainData.Hessian = MaterialInstance.Hessian     
        MainData.CauchyStress = MaterialInstance.CauchyStress

        # INITIALISE
        StrainTensors = KinematicMeasures(np.asarray([np.eye(MainData.ndim,MainData.ndim)]*\
            MainData.Domain.AllGauss.shape[0]),MainData.AnalysisType)

        MaterialInstance.Hessian(MainData.MaterialArgs,
            StrainTensors,elem=0,gcounter=0)

    else:
        raise AttributeError('Material model with name '+MainData.MaterialArgs.Type + ' not found')

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
        if MainData.MaterialArgs.Type != "IncrementalLinearElastic" and \
            MainData.MaterialArgs.Type != "LinearModel":
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
    class solve(object):
        tol = 1e-06

    if mesh.points.shape[0]*MainData.nvar > 100000:
        solve.type = "iterative"
        solve.sub_type = ""
        print 'Large system of equations. Switching to iterative solver'

    # if mesh.points.shape[0]*MainData.nvar > 100000:
    #     solve.type = "direct"
    #     solve.sub_type = "MUMPS"
    #     print 'Large system of equations. Switching to MUMPS solver'
    else:
        solve.type = "direct"
        solve.sub_type = "UMFPACK"

    MainData.solve = solve 
            

    if mesh.nelem > 1e07:
        MainData.AssemblyRoutine = 'Large'
        print 'Large number of elements. Switching to faster assembly routine'
    else:
        MainData.AssemblyRoutine = 'Small'
        # print 'Small number of elements. Sticking to small assembly routine'

    # FORCE QUIT PARALLELISATION  
    if mesh.elements.shape[0] < 100:
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




