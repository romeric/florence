from sys import exit
import numpy as np
from scipy.io import savemat, loadmat

from Core import Mesh
from Core.FiniteElements.PostProcess import *

#  
def ProjectionCriteria(mesh):
    scale = 1.
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
        x *= scale
        y *= scale
        z *= scale 
        if x > -20*scale and x < 40*scale and y > -30.*scale \
            and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
            projection_faces[iface]=1
    
    return projection_faces

def RunCurvedPlotter(filename):
    DictOutput = loadmat(filename)

    mesh = Mesh()
    mesh.elements = DictOutput['elements']
    mesh.nelem = mesh.elements.shape[0]
    mesh.element_type = DictOutput['element_type']
    mesh.points = DictOutput['points']
    mesh.faces = DictOutput['faces']
    TotalDisp = DictOutput['TotalDisp']
    # TotalDisp = np.zeros_like(mesh.points)[:,:,None]
    C = DictOutput['C']
    # C=1
    # ScaledJacobian = DictOutput['ScaledJacobian']
    ScaledJacobian = np.ones(mesh.nelem)
    ProjFlags = DictOutput['ProjFlags']
    # ProjFlags = ProjectionCriteria(mesh)

    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
            ProjectionFlags=ProjFlags,InterpolationDegree=1)


if __name__ == '__main__':

    directory = "/home/roman/Dropbox/"
    # filename = directory+"Almond3D_P2.mat"
    # filename = directory+"Almond3D_P3.mat"
    # filename = directory+"Almond3D_P4.mat"
    # filename = directory+"Almond3D_H2_P4.mat"
    # filename = directory+"Sphere.mat"
    filename = directory+"Falcon3DIso_P2.mat"

    # filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso_P2.mat"
    # filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso.mat"
    RunCurvedPlotter(filename)

    exit()


    # xx = loadmat("/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles/Mech2D.mat")
    # print xx.keys()

    # exit()

    #######################################################################
    mesh = Mesh()
    mesh.points = np.array(
[[ -9.44332213e-01,  -1.35427333e-01,   5.53225893e-18],
 [ -9.60937274e-01,  -3.93351334e-17,   3.71673294e-02],
 [ -9.70925653e-01,  -9.38901477e-02,   3.83544884e-18],
 [ -9.88808095e-01,   3.98104868e-02,   1.48378780e-01],
 [ -9.54842191e-01,  -1.17129319e-01,   9.51490161e-03],
 [ -9.59831886e-01,  -7.02745865e-02,   2.97616226e-02],
 [ -9.61123207e-01,  -2.37730152e-02,   3.62646191e-02],
 [ -9.49427049e-01,  -1.28620712e-01,  -8.26259514e-05],
 [ -9.63091629e-01,  -1.00896280e-01,   1.32583180e-02],
 [ -9.65685502e-01,  -5.32755271e-02,   2.96063673e-02],
 [ -9.64887882e-01,  -1.64025070e-02,   3.45715886e-02],
 [ -9.58556354e-01,  -1.15253325e-01,  -1.35515745e-04],
 [ -9.71079426e-01,  -8.61852053e-02,   1.27514521e-02],
 [ -9.71788692e-01,  -4.85347990e-02,   2.61499851e-02],
 [ -9.66887729e-01,  -1.01372367e-01,  -7.00596598e-05],
 [ -9.75518685e-01,  -8.05544595e-02,   8.65479574e-03],
 [ -9.60894727e-01,  -1.09981502e-01,   3.29844948e-02],
 [ -9.64630481e-01,  -6.82386744e-02,   4.79646184e-02],
 [ -9.66542600e-01,  -2.08667140e-02,   5.75699330e-02],
 [ -9.66630152e-01,   6.89128302e-03,   5.72628360e-02],
 [ -9.68513399e-01,  -9.04766324e-02,   4.08848868e-02],
 [ -9.70964986e-01,  -4.81221238e-02,   5.16608673e-02],
 [ -9.71913545e-01,  -1.19149528e-02,   5.65974840e-02],
 [ -9.76475915e-01,  -7.56347628e-02,   4.03430414e-02],
 [ -9.77522919e-01,  -4.36016066e-02,   4.67475700e-02],
 [ -9.80989835e-01,  -7.26848686e-02,   3.14260516e-02],
 [ -9.69197993e-01,  -4.77755097e-02,   7.76502308e-02],
 [ -9.74379948e-01,  -7.72611860e-03,   9.24194996e-02],
 [ -9.77139947e-01,   1.96361233e-02,   9.54212121e-02],
 [ -9.76392630e-01,  -2.91732684e-02,   8.37053425e-02],
 [ -9.80556039e-01,   1.09922015e-03,   9.18678121e-02],
 [ -9.81588289e-01,  -2.69640702e-02,   7.57924017e-02],
 [ -9.83531095e-01,   8.89182097e-03,   1.24240799e-01],
 [ -9.86274946e-01,   3.21309278e-02,   1.31179492e-01],
 [ -9.88290448e-01,   1.62291949e-02,   1.24633809e-01]])

    mesh.elements = np.arange(35).reshape(1,35)
    mesh.element_type = "tet"
    mesh.nelem = 1
    mesh.GetBoundaryFacesTet()
    mesh.GetBoundaryEdgesTet()
    ScaledJacobian = np.array([1])
    TotalDisp = np.zeros_like(mesh.points)
    ProjFlags = np.ones(4)


    # mesh.PlotMeshNumbering()
    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp[:,:,None],QuantityToPlot=ScaledJacobian.flatten(),
            ProjectionFlags=ProjFlags,InterpolationDegree=50)



    # ######################################################################
    # mesh = Mesh()
    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_iso.dat","tet",0)
    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H2.dat","tet",0)

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
    # savemat('/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso.mat',Dict)












    # # ######################################################################
    # from Base.Base import Base as MainData
    # from Main.FiniteElements.MainFEM import main

    # import imp, os, sys, time
    # from sys import exit
    # from datetime import datetime
    # import cProfile, pdb
    # import numpy as np
    # import scipy as sp
    # import numpy.linalg as la
    # from numpy.linalg import norm
    # from datetime import datetime
    # import multiprocessing as MP

    # # AVOID WRITING .pyc OR .pyo FILES
    # sys.dont_write_bytecode
    # # SET NUMPY'S LINEWIDTH PRINT OPTION
    # np.set_printoptions(linewidth=300)

    # # START THE ANALYSIS
    # print("Initiating the routines... Current time is", datetime.now().time())
    
    # MainData.__NO_DEBUG__ = True
    # MainData.__VECTORISATION__ = True
    # MainData.__PARALLEL__ = True
    # MainData.numCPU = MP.cpu_count()
    # # MainData.__PARALLEL__ = False
    # # nCPU = 8
    # __MEMORY__ = 'SHARED'
    # # __MEMORY__ = 'DISTRIBUTED'
    
    # MainData.C = 1
    # MainData.norder = 2
    # MainData.plot = (0, 3)
    # nrplot = (0, 'last')
    # MainData.write = 0


    # for p in range(2,7):
    #     MainData.C = p - 1
    #     main(MainData)