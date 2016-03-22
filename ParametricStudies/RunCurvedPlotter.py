import os, sys, gc
from sys import exit
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pyplot as plt
from copy import deepcopy

from Core import Mesh
from Core.FiniteElements.PostProcess import *
from Core.Supplementary.Tensors import itemfreq_py

#  
def ProjectionCriteria(mesh):
    scale = 25.4
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
        x *= scale
        y *= scale
        z *= scale 
        # if x > -20*scale and x < 40*scale and y > -30.*scale \
            # and y < 30.*scale and z > -20.*scale and z < 20.*scale:  
        if x > 0.*scale and x < 8*scale and y > -30.*scale \
            and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
            projection_faces[iface]=1
    
    return projection_faces

def ProjectionCriteriaFalcon(mesh):
    class self():
        scale = 25.4
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
        x *= self.scale
        y *= self.scale
        z *= self.scale
        # if x > -20*self.scale and x < 40*self.scale and y > -30.*self.scale \
            # and y < 30.*self.scale and z > -20.*self.scale and z < 20.*self.scale:  
        if x > -10*self.scale and x < 30*self.scale and y > -20.*self.scale \
            and y < 20.*self.scale and z > -15.*self.scale and z < 15.*self.scale:   
            projection_faces[iface]=1
    
    return projection_faces

def ProjectionCriteriaAlmondElems(mesh):
    scale = 25.4
    projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    num = mesh.elements.shape[1]
    for ielem in range(mesh.elements.shape[0]):
        x = np.sum(mesh.points[mesh.elements[ielem,:],0])/num
        y = np.sum(mesh.points[mesh.elements[ielem,:],1])/num
        z = np.sum(mesh.points[mesh.elements[ielem,:],2])/num
        x *= scale
        y *= scale
        z *= scale  
        # x one
        # if x > 0.*scale and x < 8*scale and y > -30.*scale \
            # and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
            # projection_faces[ielem]=1
        # x two
        # if x > -8.*scale and x < -0.1*scale and y > -30.*scale \
        #     and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
        #     projection_faces[ielem]=1
        # y = 1
        # if x > -8.*scale and x < 8*scale and y > 0.*scale \
        #     and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
        #     projection_faces[ielem]=1
        if x > -8.*scale and x < 8*scale and y < 0.*scale \
            and y > -30.*scale and z > -20.*scale and z < 20.*scale:   
            projection_faces[ielem]=1

    # print mesh.elements.shape
    mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    mesh.all_faces = None
    mesh.GetFacesTet()


def ProjectionCriteriaSphere(mesh):
    scale = 1.
    mesh.GetFacesTet()
    projection_faces = np.zeros((mesh.all_faces.shape[0],1),dtype=np.uint64)
    num = mesh.all_faces.shape[1]
    for iface in range(mesh.all_faces.shape[0]):
        x = np.sum(mesh.points[mesh.all_faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.all_faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.all_faces[iface,:],2])/num
        x *= scale
        y *= scale
        z *= scale  
        if x >= 0.*scale:   
            projection_faces[iface]=1
    
    return projection_faces

def ProjectionCriteriaAlmond(mesh):
    class self():
        scale = 25.4
    scale = 25.4

    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        x = np.sum(mesh.points[mesh.faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.faces[iface,:],2])/num
        x *= scale
        y *= scale
        z *= scale 
        # if np.sqrt(x*x+y*y+z*z)< self.condition:
        if x > -2.5*self.scale and x < 2.5*self.scale and y > -2.*self.scale \
            and y < 2.*self.scale and z > -2.*self.scale and z < 2.*self.scale:    
            projection_faces[iface]=1
    
    return projection_faces


def ProjectionCriteriaAlmondAllFaces(mesh):
    scale = 25.4
    projection_faces = np.zeros((mesh.all_faces.shape[0],1),dtype=np.uint64)
    num = mesh.all_faces.shape[1]
    for iface in range(mesh.all_faces.shape[0]):
        x = np.sum(mesh.points[mesh.all_faces[iface,:],0])/num
        y = np.sum(mesh.points[mesh.all_faces[iface,:],1])/num
        z = np.sum(mesh.points[mesh.all_faces[iface,:],2])/num
        x *= scale
        y *= scale
        z *= scale  
        if x > 0.*scale and x < 8*scale and y > -30.*scale \
            and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
            projection_faces[iface]=1
    
    return projection_faces


def ProjectionCriteriaFalconElems(mesh):
    class self():
        scale = 25.4
    scale = 25.4
    projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    num = mesh.elements.shape[1]
    for ielem in range(mesh.elements.shape[0]):
        x = np.sum(mesh.points[mesh.elements[ielem,:],0])/num
        y = np.sum(mesh.points[mesh.elements[ielem,:],1])/num
        z = np.sum(mesh.points[mesh.elements[ielem,:],2])/num
        x *= scale
        y *= scale
        z *= scale  
        # x one
        # if x > 0.*scale and x < 8*scale and y > -30.*scale \
            # and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
            # projection_faces[ielem]=1
        # x two
        # if x > -8.*scale and x < -0.1*scale and y > -30.*scale \
        #     and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
        #     projection_faces[ielem]=1
        # y = 1
        # if x > -8.*scale and x < 8*scale and y > 0.*scale \
        #     and y < 30.*scale and z > -20.*scale and z < 20.*scale:   
        #     projection_faces[ielem]=1
        if x > -30*self.scale and x < 60*self.scale and y > -50.*self.scale \
            and y < 0.*self.scale and z > -55.*self.scale and z < 55.*self.scale:   
            projection_faces[ielem]=1

    # print mesh.elements.shape
    mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    mesh.all_faces = None
    mesh.GetFacesTet()

def ProjectionCriteriaFalconJacobianElems(mesh,ScaledJacobian):

    projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    pos = np.where(ScaledJacobian<0.1)[0]
    projection_faces[pos,0] = 1

    mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    mesh.all_faces = None
    # mesh.faces = None
    mesh.GetFacesTet()
    projection_faces = np.ones(mesh.all_faces.shape[0])
    # mesh.GetBoundaryFacesTet()
    
    return projection_faces

def ProjectionCriteriaF6(mesh):
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        Y = np.where(abs(mesh.points[mesh.faces[iface,:3],1])<1e-07)[0]
        if Y.shape[0]!=3:
            projection_faces[iface]=1
    
    return projection_faces

def ProjectionCriteriaF6Faces(mesh):
    class self():
        scale = 25.4
    scale = 25.4
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for ielem in range(mesh.faces.shape[0]):
        y = np.sum(mesh.points[mesh.faces[ielem,:],1])/num
        y *= scale 
        if y < 0.1*self.scale and y > -3.5*self.scale:   
            projection_faces[ielem]=1

    return projection_faces


def ProjectionCriteriaF6Elems(mesh):
    class self():
        scale = 25.4
    scale = 25.4
    projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    num = mesh.elements.shape[1]
    for ielem in range(mesh.elements.shape[0]):
        x = np.sum(mesh.points[mesh.elements[ielem,:],0])/num
        y = np.sum(mesh.points[mesh.elements[ielem,:],1])/num
        z = np.sum(mesh.points[mesh.elements[ielem,:],2])/num
        x *= scale
        y *= scale
        z *= scale  
        if x > 3.*self.scale and x < 50*self.scale and y > -50.*self.scale \
            and y < 50.*self.scale and z > -55.*self.scale and z < 55.*self.scale:   
            projection_faces[ielem]=1

    mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    mesh.all_faces = None
    mesh.GetFacesTet()


def ProjectionCriteriaF6Wing(mesh):
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    # print mesh.points[mesh.faces[:,:3],1]
    # Y = np.where((mesh.points[mesh.faces[:,:3],1])<-5)[0]
    # print Y
    for iface in range(mesh.faces.shape[0]):
        Y = np.min(mesh.points[mesh.faces[iface,:3],1])
        Zmin = np.min(mesh.points[mesh.faces[iface,:3],0])
        Zmax = np.max(mesh.points[mesh.faces[iface,:3],2])
        # print Y
        if Y <-2.5:
            projection_faces[iface]=1
    
    return projection_faces


def ProjectionCriteriraF6Layer(mesh):
    print mesh.Bounds
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    # print mesh.points[mesh.faces[:,:3],1]
    # Y = np.where((mesh.points[mesh.faces[:,:3],1])<-5)[0]
    # print Y
    for iface in range(mesh.faces.shape[0]):
        Y = np.min(mesh.points[mesh.faces[iface,:3],1])
        if Y <-6 and Y > -9:
            projection_faces[iface]=1
    
    return projection_faces


def ProjectionCriteriaMech3D(mesh,ScaledJacobian):
    projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    # smin = np.argmin(ScaledJacobian)
    smin = np.where(ScaledJacobian<0.1)[1]
    projection_faces[smin,0] = 1

    mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    mesh.all_faces = None
    mesh.GetFacesTet()
    projection_faces = np.ones(mesh.all_faces.shape[0])
    return projection_faces

    # print mesh.Bounds

    # class self():
    #     scale = 1.
    # scale = 1.
    # projection_faces = np.zeros((mesh.elements.shape[0],1),dtype=np.uint64)
    # num = mesh.elements.shape[1]
    # for ielem in range(mesh.elements.shape[0]):
    #     x = np.sum(mesh.points[mesh.elements[ielem,:],0])/num
    #     y = np.sum(mesh.points[mesh.elements[ielem,:],1])/num
    #     z = np.sum(mesh.points[mesh.elements[ielem,:],2])/num
    #     x *= scale
    #     y *= scale
    #     z *= scale  
    #     if x > -2.*self.scale and x < 50*self.scale and y > -5.*self.scale \
    #         and y < 5.*self.scale and z > -5.*self.scale and z < 5.*self.scale:   
    #         projection_faces[ielem]=1

    # mesh.elements = mesh.elements[projection_faces.flatten()==1,:]
    # mesh.all_faces = None
    # mesh.GetFacesTet()

    # return projection_faces.flatten()





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
    ScaledJacobian = DictOutput['ScaledJacobian']
    # ScaledJacobian = np.ones(mesh.nelem)


    ProjFlags = DictOutput['ProjFlags']
    # ProjFlags = ProjectionCriteria(mesh)

    # ProjFlags[:ProjFlags.shape[0]*8/10]=0
    # ProjFlags[ProjFlags.shape[0]*2/10:]=0
    # print itemfreq_py(ProjFlags.flatten())


    # #######################
    # # mesh.GetFacesTet()
    # # ProjFlags = ProjectionCriteriaAlmond(mesh)    
    # # mesh.all_faces = mesh.all_faces[ProjFlags.flatten()==1,:]

    # print ScaledJacobian[ScaledJacobian<0.92].shape
    # ProjFlags = ProjectionCriteriaMech3D(mesh,ScaledJacobian)
    # print ProjFlags
    # exit()

    vmesh = deepcopy(mesh)
    ProjFlags = ProjectionCriteriaAlmondElems(vmesh)
    # import matplotlib as mpl
    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab
    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))

    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=1,tube_radius=0.1,color=(73/255.,89/255.,133/255.),
    #             representation='surface')
    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=1,tube_radius=1,color=(0,0,0),
    #             representation='wireframe')
    # mlab.show()
    # # mesh.all_faces = None
    ProjFlags = DictOutput['ProjFlags']
    # # exit()
    # # figure = None

    # PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
    #         ProjectionFlags=ProjFlags,InterpolationDegree=20,figure=figure)
    # #######################
    # # exit()
    
    # ProjFlags = DictOutput['ProjFlags']
    # ProjFlags = ProjectionCriteriaSphere(mesh)

    # exit()
    # TotalDisp = np.zeros_like(TotalDisp)
    # TotalDisp = np.sum(TotalDisp,axis=2)[:,:,None]
    # print TotalDisp
    # exit()
    # mesh.dd=1
    # post_process = PostProcess(3,3)
    # post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,ProjectionFlags=ProjFlags,
    #     InterpolationDegree=50,figure=figure)

    post_process = PostProcess(3,3)
    post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,ProjectionFlags=ProjFlags,
        InterpolationDegree=60,figure=figure,plot_points=True,point_radius=0.004)

    # mesh.dd=2
    # ProjFlags = ProjectionCriteriaMech3D(mesh,ScaledJacobian)
    # # print mesh.faces
    # PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
    #         ProjectionFlags=ProjFlags,InterpolationDegree=10,figure=figure)


def RunCurvedPlotterFalcon3D(filename):
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
    ScaledJacobian = DictOutput['ScaledJacobian'].flatten()
    # ScaledJacobian = np.ones(mesh.nelem)
    ProjFlags = DictOutput['ProjFlags']
    # print np.where(ScaledJacobian<0.1)


    # #######################
    # # mesh.GetFacesTet()
    # # ProjFlags = ProjectionCriteriaAlmond(mesh)    
    # # mesh.all_faces = mesh.all_faces[ProjFlags.flatten()==1,:]

    # ProjFlags = ProjectionCriteriaFalconJacobianElems(mesh,ScaledJacobian)
    # exit()

    vmesh = deepcopy(mesh)
    ProjFlags = ProjectionCriteriaFalconElems(vmesh)


    # print np.histogram(ScaledJacobian,bins=[0.1,0.5,0.6,0.7,0.8,0.9,1.0])
    # print np.where(ScaledJacobian<0.33)[0].shape
    # # print (402+1271+35436)/37717.
    # exit()


    import matplotlib as mpl
    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab
    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))

    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=1,tube_radius=0.1,color=(73/255.,89/255.,133/255.),
    #             representation='surface')
    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=0.000001,tube_radius=0.00001,color=(0,0,0),
    #             representation='wireframe')

    # # mlab.view(azimuth=59, elevation=80, distance=60, focalpoint=None,
    # #             roll=None, reset_roll=True, figure=None)
    # mlab.view(azimuth=-220, elevation=80, distance=60, focalpoint=None,
    #             roll=None, reset_roll=True, figure=None)
    # mlab.show()
    # exit()
    # mesh.all_faces = None
    ProjFlags = DictOutput['ProjFlags']
    # exit()
    # # figure = None
    post_process = PostProcess(3,3)
    post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,
            ProjectionFlags=ProjFlags,InterpolationDegree=15,figure=figure)

    # # print np.where(ScaledJacobian<0.1)
    # ProjFlags = ProjectionCriteriaFalconJacobianElems(mesh,ScaledJacobian)
    # # print np.where(ScaledJacobian<0.1)
    # mesh.dd = 2
    # PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
    #         ProjectionFlags=ProjFlags,InterpolationDegree=15,figure=figure)
    # #######################

    # TotalDisp = np.zeros_like(TotalDisp)
    # TotalDisp = np.sum(TotalDisp,axis=2)[:,:,None]
    # PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
    #         ProjectionFlags=ProjFlags,InterpolationDegree=0)

    
def RunCurvedPlotterF6(filename):


    post_process = PostProcess(3,3)

    DictOutput = loadmat(filename)
    mesh = Mesh()
    mesh.elements = DictOutput['elements']
    mesh.nelem = mesh.elements.shape[0]
    mesh.element_type = DictOutput['element_type']
    mesh.points = DictOutput['points']
    mesh.faces = DictOutput['faces']
    TotalDisp = DictOutput['TotalDisp']
    # TotalDisp = np.zeros_like(mesh.points)[:,:,None]
    C = mesh.InferPolynomialDegree() - 1
    ScaledJacobian = DictOutput['ScaledJacobian'].flatten()
    # ScaledJacobian = np.ones(mesh.nelem)
    # ProjFlags = DictOutput['ProjFlags'].flatten()
    # print np.where(ScaledJacobian<0.1)
    ProjFlags = ProjectionCriteriaF6(mesh).flatten()
    # ProjFlags = ProjectionCriteriaF6Faces(mesh).flatten()

    # print TotalDisp.max()
    # exit()

    # vmesh = deepcopy(mesh)
    # ProjFlags = ProjectionCriteriaF6Elems(vmesh)
    # print mesh.Bounds
    # exit()
    del DictOutput
    gc.collect()


    # print np.histogram(ScaledJacobian,bins=[0.1,0.5,0.6,0.7,0.8,0.9,1.0])
    # mm = 0.9
    # print np.where(ScaledJacobian<mm)[0].shape
    # print ScaledJacobian[ScaledJacobian<mm]
    # # print (402+1271+35436)/37717.


    # ProjFlags = ProjectionCriteriaF6Wing(mesh)
    # mesh.faces = mesh.faces[ProjFlags.flatten()==1,:]
    # mesh.SimplePlot()
    # exit()


    import matplotlib as mpl
    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab
    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))

    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=1,tube_radius=0.1,color=(73/255.,89/255.,133/255.),
    #             representation='surface')
    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=0.000001,tube_radius=0.00001,color=(0,0,0),
    #             representation='wireframe')

    # mesh.points[:,1] *= -1.

    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=1,tube_radius=0.1,color=(73/255.,89/255.,133/255.),
    #             representation='surface')
    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], vmesh.all_faces[:,:3],
    #             line_width=0.000001,tube_radius=0.00001,color=(0,0,0),
    #             representation='wireframe')

    # # # mlab.view(azimuth=59, elevation=80, distance=60, focalpoint=None,
    # # #             roll=None, reset_roll=True, figure=None)
    # mlab.view(azimuth=-220, elevation=80, distance=70, focalpoint=None,
    #             roll=None, reset_roll=True, figure=None)
    # # mlab.show()
    # # exit()
    # # mesh.all_faces = None

    # mesh.points[:,1] *= -1.


    # ProjFlags = DictOutput['ProjFlags'].flatten()
    # TotalDisp = np.sum(TotalDisp,axis=2)[:,:,None]

    # TotalDisp = np.zeros_like(TotalDisp)
    # print TotalDisp.max()
    # TotalDisp = TotalDisp[:,:,None]
    # print TotalDisp.max()
    # print mesh.Bounds
    # exit()

    # AppliedDirichlet = np.loadtxt("/media/MATLAB/f6BL_Dirichlet_P3.dat",dtype=np.float32)
    # ColumnsOut = np.loadtxt("/media/MATLAB/f6BL_ColumnsOut_P3.dat").astype(np.int32)
    
    # nvar = 3
    # TotalSol = np.zeros((mesh.points.shape[0]*nvar,1))
    # # GET TOTAL SOLUTION
    # TotalSol[ColumnsOut,0] = AppliedDirichlet
    # TotalDisp = TotalSol.reshape(TotalSol.shape[0]//nvar,nvar)[:,:,None]


    # lmesh = Mesh()
    # Dict = loadmat("/media/MATLAB/Layers_P3/f6BL_Layer_0_Sol_P3.mat")
    # # Dict = loadmat("/media/MATLAB/f6BL_P3.mat")
    # lmesh.elements = Dict['elements']
    # lmesh.points = Dict['points']
    # lmesh.faces = Dict['faces']
    # lmesh.element_type = "tet"
    # TotalSol = Dict['TotalDisp']

    # nvar = 3
    # TotalSol = np.zeros((lmesh.points.shape[0]*nvar,1))
    # # GET TOTAL SOLUTION
    # TotalSol[ColumnsOut,0] = AppliedDirichlet
    # TotalSol = TotalSol.reshape(TotalSol.shape[0]//nvar,nvar)[:,:,None]





    # PostProcess.HighOrderCurvedPatchPlot(lmesh,Dict['TotalDisp'],QuantityToPlot=np.ones((lmesh.elements.shape[0])),
    #         ProjectionFlags=np.ones((lmesh.faces.shape[0])),InterpolationDegree=2,
    #         color=(73/255.,89/255.,133/255.),figure=figure,show_plot=False)

    # PostProcess.HighOrderCurvedPatchPlot(lmesh,Dict['TotalDisp'],QuantityToPlot=np.ones((lmesh.elements.shape[0])),
    #         ProjectionFlags=ProjectionCriteriraF6Layer(lmesh).flatten(),InterpolationDegree=2,
    #         color=(73/255.,89/255.,133/255.),figure=figure,show_plot=False)

    # ProjectionCriteriraF6Layer(mesh).flatten()

    # Dict = loadmat("/media/MATLAB/Layers/f6BL_Layer_9_P3.mat")
    # lmesh = Mesh()
    # lmesh.ReadHDF5("/media/MATLAB/Layers/f6BL_Layer_0_P3.mat")
    # lmesh.SimplePlot(figure=figure,color=(73/255.,89/255.,133/255.),show_plot=True)
    # mesh.SimplePlot(figure=figure, show_plot=False,color=(73/255.,89/255.,133/255.))

    # faces = np.copy(mesh.faces)
    # mesh.faces = mesh.faces[ProjFlags==0,:]
    # ProjFlags = np.ones((faces.shape[0]))

    # TotalDisp[:,1,:] *= -1
    # mesh.points[:,1] *= -1
    
    # np.ones(lmesh.faces.shape[0])
    # post_process.HighOrderCurvedPatchPlot(lmesh,TotalSol, ProjectionFlags=ProjectionCriteriaF6Faces(lmesh),
    #     InterpolationDegree=9,figure=figure, color=(73/255.,89/255.,133/255.), show_plot=False)

    post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,ProjectionFlags=ProjectionCriteriaF6Faces(mesh),
        InterpolationDegree=4, color=(73/255.,89/255.,133/255.), figure=figure,show_plot=False)

    # TotalDisp = TotalDisp[:,:,None]
    TotalDisp[:,1] *= -1
    mesh.points[:,1] *= -1

    post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,ProjectionFlags=ProjFlags,
        InterpolationDegree=4,figure=figure,show_plot=False)

    # print mesh.Bounds
    # exit()

    svpoints = mesh.points + TotalDisp#[:,:,-1]
    # svpoints = mesh.points + TotalSol[:,:,-1]
    # faces = mesh.faces[ProjFlags.flatten()==1,:]

    # svpoints = lmesh.points + TotalSol[:,:,-1]
    faces = mesh.faces[ProjFlags.flatten()==1,:]
    svpoints = svpoints[np.unique(faces),:]
    # # svpoints = svpoints[svpoints[:,1]<=0,:]
    # # svpoints = svpoints[svpoints[:,1]>-2,:]
    # # svpoints = mesh.points + TotalDisp[:,:,-1]

    TotalDisp[:,1] *= -1
    mesh.points[:,1] *= -1

    svpoints = svpoints[svpoints[:,0]<-10,:]
    # svpoints = svpoints[svpoints[:,1]>-9.2,:]
    # svpoints = svpoints[svpoints[:,1]<-4.7,:]
    # svpoints = svpoints[svpoints[:,1]<-12,:]
    # svpoints = svpoints[svpoints[:,1]>-15.5,:]
    mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=0.005)

    mlab.show()




if __name__ == '__main__':

    # directory = "/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Almond3D/"
    directory = "/media/MATLAB/RESULTS_DIR/Falcon3D/"
    # directory = "/media/MATLAB/RESULTS_DIR/F6/"
    # directory = "/media/MATLAB/Layers/"
    # directory = "/home/roman/Dropbox/"
    # directory = "/home/roman/Falcon3D/"

    # filename = directory+"Almond3D_P2.mat"
    # filename = directory+"Almond3D_P3.mat"
    # filename = directory+"Almond3D_P4.mat"
    # filename = directory+"Almond3D_H2_P4.mat"
    # filename = directory+"Sphere.mat"
    # filename = directory+"Falcon3DIso_P2_Old.mat"
    # filename = directory+"Falcon3DIso_P2.mat"
    # filename = directory+"Falcon3DIso_P3.mat"
    # filename = directory+"Falcon3DIso_P4.mat"
    # filename = directory+"Falcon3DIso_P5.mat"
    # filename = directory+"Falcon3DBig_P3.mat"

    # filename = directory+"F6_P2.mat"
    # filename = directory+"F6_P4.mat"

    # filename = directory+"Drill_P3.mat"
    # filename = directory+"Drill_P5.mat"
    # filename = directory+"Valve_P5.mat"
    # filename = directory+"MechanicalComponent3D_P2.mat"
    # filename = directory+"MechanicalComponent3D_P2_New.mat"
    # filename = directory+"MechanicalComponent3D_P2_New_200.mat"
    # filename = directory+"MechanicalComponent3D_P3.mat"
    # filename = directory+"F6Iso_P2.mat"
    # filename = directory+"F6Iso_P4.mat"
    # filename = os.path.join(directory,"f6BL_P3_SOLUTION_avg.mat")
    # filename = os.path.join("/media/MATLAB/Layers_P4/","f6BL_P4_SOLUTION_min.mat")
    # filename = "/media/MATLAB/f6BL_Layer_dd_Sol_P3.mat"
    # filename = os.path.join("/home/roman/LayerSolution/Layers_P4/","f6BL_P4_SOLUTION_min.mat")
    # filename = os.path.join("/media/MATLAB/Layers_P3/","f6BL_P3_SOLUTION_min.mat")
    # filename = "/media/MATLAB//Layers_P3/f6BL_Layer_4_Sol_P3.mat"
    # filename = os.path.join("/media/MATLAB/f6BL_P3.mat")
    # filename = "/media/MATLAB/Layers_P3/f6BL_P3_SOLUTION_None_RESCALED_FIXED.mat" #
    # filename = "/media/MATLAB/Layers_P4/f6BL_P4_SOLUTION_None_RESCALED_FIXED.mat"

    # filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso_P2.mat"
    # filename = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/Falcon3DIso.mat"

    filename = "/media/MATLAB/Repository/Almond3D_P6.mat"

    RunCurvedPlotter(filename)
    # RunCurvedPlotterFalcon3D(filename)
    # RunCurvedPlotterF6(filename)

    exit()

    


    ############################################################################
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/bracket_log").astype(np.int64)

    mesh = Mesh()
    # mesh.ReadHDF5("/home/roman/f6BL_P2.mat")
    # mesh.ReadHDF5("/media/MATLAB/f6BL_P2.mat")
    # print mesh.faces
    # del mesh
    # mesh = Mesh()
    # mesh.ReadGIDMesh("/media/MATLAB/f6BL.dat","tet")
    # print mesh.faces
    # exit()

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Tests/bracketH0.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/bracket_log").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/f6_dir2").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6_iso.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/f6_iso_log2").astype(np.int64)

    mesh.ReadGIDMesh("/media/MATLAB/f6BL.dat","tet")
    dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/f6BL_log3").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/mechanicalComplex.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/mech3d").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/almond_H1.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/almond_log").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Drill/drill.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/drill_log2").astype(np.int64)

    # mesh.ReadGIDMesh("/home/roman/Dropbox/Florence/Examples/FiniteElements/Valve/valve.dat","tet")
    # dirichlet_faces_int = np.loadtxt("/home/roman/Dropbox/valve_log2").astype(np.int64)

    # mesh.SimplePlot()


    ProjFlags = ProjectionCriteriaF6(mesh).flatten()
    # ProjFlags = ProjectionCriteriaAlmond(mesh).flatten()
    # ProjFlags = np.ones(mesh.faces.shape[0],dtype=np.int64)


    # dirichlet_faces_int = np.concatenate((mesh.faces,dirichlet_faces_int),axis=1).astype(np.int64)
    dirichlet_faces_ext = np.concatenate((mesh.faces,mesh.face_to_surface[:,None]),axis=1).astype(np.int64)
    dirichlet_faces_ext = dirichlet_faces_ext[ProjFlags==1,:]
    # print dirichlet_faces_ext

    # print dirichlet_faces_int[:10,:]
    # print 
    # print dirichlet_faces_ext[:10,:]
    # exit()
    
    ################################

    # from Core.Supplementary.Tensors import shuffle_along_axis
    # dirichlet_faces_int[:,:3].sort()
    # dirichlet_faces_ext[:,:3].sort()

    # mapper = shuffle_along_axis(dirichlet_faces_ext[:,:3],dirichlet_faces_int[:,:3],consider_sort=True)
    # dirichlet_faces_ext = dirichlet_faces_ext[mapper,:]

    # print dirichlet_faces_int
    # print 
    # print dirichlet_faces_ext
    # print mesh.faces
    # exit()

    # x1 = np.where(dirichlet_faces_int==8029)[0]
    # x2 = np.where(dirichlet_faces_int==8546)[0]
    # x3 = np.where(dirichlet_faces_int==8279)[0]

    # yy = np.intersect1d(np.intersect1d(x1,x2),x3)[0]
    # # print dirichlet_faces_int[yy,:]
    # print yy
    # exit()


    dirichlet_faces_int2 = np.copy(dirichlet_faces_int)
    unique_flags_ext = np.unique(dirichlet_faces_ext[:,-1])

    from Core.Supplementary.Tensors import itemfreq_py
    xx = []; yy = []
    for i in unique_flags_ext:
        poss = np.where(dirichlet_faces_ext[:,-1]==i)[0]
        # print dirichlet_faces_ext[poss,-1]
        xx =  itemfreq_py(dirichlet_faces_int[poss,-1])
        # print xx
        dirichlet_faces_int[poss,-1] = xx[xx[:,1].argmax(),0]
        # xx.append(dirichlet_faces_ext[poss,-1])
        # yy.append( dirichlet_faces_int[poss,-1])

    yy = np.zeros(mesh.faces.shape[0])
    yy[ProjFlags==0] = 10000
    yy[ProjFlags==1] = dirichlet_faces_int[:,-1]
    # print np.where(yy==10000)
    # exit()
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6_iso_face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6BL_face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Drill/face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Valve/face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/face_to_surface_mapped.dat",yy)
    # np.savetxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/Almond3D/face_to_surface_mapped.dat",yy)
    # print (dirichlet_faces_int - dirichlet_faces_int2)[:,-1][5000:6000]
    # print yy[0],xx[0]

    # print mesh.faces
    print dirichlet_faces_int
    # print dirichlet_faces_int[14703,:]
    # print yy
    # print np.loadtxt("/home/roman/Dropbox/Florence/Examples/FiniteElements/F6/f6_iso_face_to_surface_mapped.dat")



    # print np.unique(dirichlet_faces_ext[:,-1])
    # lmesh = Mesh()
    # lmesh.ReadGIDMesh("/media/MATLAB/f6BL.dat","tet")
    # print np.unique(lmesh.face_to_surface)
    # print itemfreq_py(lmesh.face_to_surface)
    # print np.unique(dirichlet_faces_ext[:,-1])
    # print itemfreq_py(dirichlet_faces_int[:,-1])



    exit()
    ############################################################################


    #############################################################################
    DictOutput = loadmat(filename)

    import os
    os.environ['ETS_TOOLKIT'] = 'qt4'
    from mayavi import mlab

    # ProjFlags = DictOutput['ProjFlags']
    mesh = Mesh()
    mesh.faces = DictOutput['faces']
    mesh.points = DictOutput['points']
    mesh.elements = DictOutput['elements']
    mesh.nelem = mesh.elements.shape[1]
    mesh.element_type = "tet"
    ProjFlags =  ProjectionCriteriaFalcon(mesh).flatten()

    # print mesh.faces.shape
    mesh.faces = mesh.faces[ProjFlags==1,:]

    mlab.triangular_mesh(mesh.points[:,0],mesh.points[:,1],mesh.points[:,2],mesh.faces[:,:3],scalars=np.random.rand(mesh.points.shape[0]))
    mlab.show()
    # print mesh.faces
    # print DictOutput.keys()

    exit()
    #############################################################################

    #############################################################################
    # from Core import Mesh
    # from scipy.spatial import Delaunay

    # DictOutput = loadmat("/home/roman/Dropbox/deformed_configuration.mat",squeeze_me=True, struct_as_record=False)
    # # print DictOutput.keys()
    # mesh = Mesh()
    # mesh.points = DictOutput['GEOM'].x0.T
    # # mesh.points = DictOutput['GEOM'].x.T
    # mesh.elements = DictOutput['FEM'].mesh.connectivity.T - 1
    # mesh.nelem = mesh.elements.shape[0]
    # mesh.element_type = "hex"
    # # print mesh.points.shape
    # cauchy = DictOutput['str'].postproc.cauchy.T
    # ones = np.ones_like(cauchy[:,-2])
    # # print cauchy[:,-2]
    # # print mesh.elements.min()
    # # mesh.WriteVTK("/home/roman/Dropbox/dd.vtu",pdata=cauchy[:,-2])
    # mesh.WriteVTK("/home/roman/Dropbox/dd.vtu",pdata=ones)
    # # print repr(DictOutput)


    ###############################
    # import os
    # os.environ['ETS_TOOLKIT'] = 'qt4'
    # from mayavi import mlab

    # TrianglesFunc = Delaunay(mesh.points)
    # mesh.elements = TrianglesFunc.simplices.copy()
    # mesh.element_type = "tet"
    # mesh.nelem = mesh.elements.shape[0]
    # mesh.GetBoundaryFacesTet()
    # # print Triangles
    # trimesh_h = mlab.triangular_mesh(mesh.points[:,0], 
    #             mesh.points[:,1], mesh.points[:,2], mesh.faces[:,:3],
    #             line_width=0.1,tube_radius=0.1,color=(0,0.6,0.4),
    #             representation='wireframe') 
    # mlab.show()
    exit()
    #############################################################################



    #########################################################################
    # fpath = "/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles/Wing2D_Stretch25.mat"
    # fpath = "/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles/Wing2D_Stretch1600.mat"
    fpath = "/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles/Wing2D_Nonlinear.mat"

    Results = loadmat(fpath)
    print Results.keys()
    exit()
    mesh = Mesh()
    approach = "NeoHookean_2"
    approach = "Linear_NeoHookean_2"
    # approach = "Linear_IncrementalLinearElastic"

    mesh.elements = Results['MeshElements'] - 1
    mesh.element_type = "tri"
    mesh.nelem = mesh.elements.shape[0]
    mesh.points = Results['MeshPoints']

    TotalDisp = Results['TotalDisplacement']
    ScaledJacobian = Results['ScaledJacobian']

    # mesh.points = mesh.points - TotalDisp[:,:,-1]
    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
            InterpolationDegree=10)
    plt.show()

    exit()



    Results = loadmat(fpath)
    print Results.keys()
    p = 3
    mesh = Mesh()
    approach = "NeoHookean_2"
    approach = "Linear_NeoHookean_2"
    # approach = "Linear_IncrementalLinearElastic"

    mesh.elements = Results['MeshElements_P'+str(p)] - 1
    mesh.element_type = "tri"
    mesh.nelem = mesh.elements.shape[0]
    mesh.points = Results['MeshPoints_P'+str(p)]

    TotalDisp = Results['TotalDisplacement_P'+str(p)+'_'+approach]
    ScaledJacobian = Results['ScaledJacobian_P'+str(p)+'_'+approach]
    ProjFlags = np.zeros(mesh.elements.shape[0])

    # print TotalDisp.max()
    print TotalDisp[:,:,-1].max()
    exit()
    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp,QuantityToPlot=ScaledJacobian.flatten(),
            InterpolationDegree=10,ProjectionFlags=ProjFlags)
    plt.show()





    exit()
    #########################################################################



    # DictOutput = loadmat("/home/roman/Dropbox/check_falcon.mat")
    DictOutput = loadmat(filename)
    ScaledJacobian = DictOutput['ScaledJacobian'].flatten()
    # print ScaledJacobian.shape[0]
    x, y = np.histogram(ScaledJacobian)
    # x, y = np.histogram(ScaledJacobian)

    # x = x.astype(np.float64)
    # x *= 1./ScaledJacobian.shape[0] 
    # print x
    # print y
    # print x.shape, y.shape
    # print ScaledJacobian[ScaledJacobian<0.1]
    # exit()

    import matplotlib
    import matplotlib.pyplot as plt
    plt.hist(ScaledJacobian.flatten(),color="#FFBBAA",bins=10,
        weights=np.zeros_like(ScaledJacobian.flatten()) + 100./ScaledJacobian.flatten().shape[0])
    # plt.bar(x,y)
    plt.show()
    exit()

    #############################################################################

    mesh = Mesh()
    mesh.points = np.array(
        [[ 12.83473098,   1.27472899,   0.17391122],
         [ 12.91923171,   1.06567849,   0.14300067],
         [ 13.301     ,   1.18309293,   0.19702503],
         [ 12.62500975,   1.21870554,   0.15335219],
         [ 12.87698134,   1.17020374,   0.15845595],
         [ 13.06786549,   1.22891096,   0.18546813],
         [ 13.11011585,   1.12438571,   0.17001285],
         [ 12.72987036,   1.24671727,   0.16363171],
         [ 12.77212073,   1.14219201,   0.14817643],
         [ 12.96300487,   1.20089924,   0.17518861]]
        )

    mesh.elements = np.arange(10).reshape(1,10)
    mesh.element_type = "tet"
    mesh.nelem = 1
    mesh.GetBoundaryFacesTet()
    mesh.GetBoundaryEdgesTet()
    ScaledJacobian = np.array([1])
    TotalDisp = np.zeros_like(mesh.points)
    ProjFlags = np.ones(4)

    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalDisp[:,:,None],QuantityToPlot=ScaledJacobian.flatten(),
            ProjectionFlags=ProjFlags,InterpolationDegree=50)

    exit()
    ############################################################################


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
