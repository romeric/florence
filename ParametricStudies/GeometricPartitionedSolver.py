#####################################################################################################
import imp
import os
import sys
import numpy as np
import gc
from scipy.io import savemat, loadmat
from Core import Mesh
from sys import exit

def ProjectionCriteriaF6(mesh):
    projection_faces = np.zeros((mesh.faces.shape[0],1),dtype=np.uint64)
    num = mesh.faces.shape[1]
    for iface in range(mesh.faces.shape[0]):
        Y = np.where(abs(mesh.points[mesh.faces[iface,:3],1])<1e-07)[0]
        if Y.shape[0]!=3:
            projection_faces[iface]=1
    
    return projection_faces



class PartitionedSolver(object):

    def __init__(self,bigfile,nlayer,ndim=3,p=3):
        """
            bigfile:        [str] The filename containing the mesh to be partitioned 
            nlayer:         [int] number of layers/partitions to consider
            ndim:           [int] spatial dimensions e.g. ndim=3 for tets and ndim=2 for tris

            """

        self.nlayer = nlayer
        self.partition_axis = 0
        self.condition = None
        self.y_avg = None
        self.matfile = bigfile
        self.nvar = 3
        self.p = p
        self.directory = "/home/roman/LayerSolution/Layers_P"+str(self.p)

    def PreparePartitions(self,matfile, nlayer=1,partition_axis=0,condition=None,plot_big_mesh=False):
        mesh = Mesh()
        Dict = loadmat(matfile)
        mesh.elements = Dict['elements']
        mesh.points = Dict['points']
        mesh.faces = Dict['faces']
        mesh.edges = Dict['edges']
        mesh.element_type = "tet"

        if plot_big_mesh:
            AppliedDirichlet = np.loadtxt(os.path.join("/home/roman","f6BL_Dirichlet_P"+str(self.p)+".dat")).astype(np.float32)
            ColumnsOut = np.loadtxt(os.path.join("/home/roman","f6BL_ColumnsOut_P"+str(self.p)+".dat")).astype(np.int32)

            nvar = self.nvar
            TotalSol = np.zeros((mesh.points.shape[0]*nvar,1))
            TotalSol[ColumnsOut,0] = AppliedDirichlet
            TotalSol = TotalSol.reshape(TotalSol.shape[0]//nvar,nvar)
            ProjFlags = ProjectionCriteriaF6(mesh).flatten()

            from Core.FiniteElements.PostProcess import PostProcess
            post_process = PostProcess(3,3)
            post_process.HighOrderCurvedPatchPlot(mesh,TotalSol[:,:,None],
                InterpolationDegree=5,ProjectionFlags=ProjFlags,show_plot=True)


        self.y_avg = np.sum(mesh.points[mesh.elements[:,:4],1],axis=1)/4.
        bounds = mesh.Bounds[:,partition_axis]

        if self.condition is None:
            # y-axis partitioning for F6
            # self.condition = np.array([-29, -15.5, -9.5, -7.7, -3.5, -2.2, -1.0, 0.2]) ####
            # self.condition = np.array([-29.0, -23.5, -22.2, -20, -16, -13, -11.2, -10.1, -9.6, -9.3, -8.8, -8.3,
            #     -7.8, -7.3, -6., -4.2, -3.3, -3., -2.7, -2.4, -2.1, -1.7, -1.4, -1.0, -0.6, -0.3, 0.1]) # Final used mesh for P3

            # self.condition = np.array([-29.0, -24.1]) # Final used for P4
            # self.condition = np.concatenate((self.condition,np.linspace(-24,0.1,200))) # Final used for P4

            if self.p == 3:
                self.condition = np.array([-29.0, -14.5, -9.3, -7, -2.8, -1.4, 0.1]) 
            elif self.p == 4:
                self.condition = np.array([-29.0, -23.5]) # Final used mesh for P3 bigger
                self.condition = np.concatenate((self.condition,np.linspace(-22,-11.5,5))) 
                self.condition = np.concatenate((self.condition,np.linspace(-10.5,-7.0,8))) 
                self.condition = np.concatenate((self.condition,np.linspace(-5.7,-3.5,3))) 
                self.condition = np.concatenate((self.condition,np.linspace(-3,0.1,11))) 


        # FIRST CHECK PARTITIONING
        x,y = [],[]
        container = []
        print self.condition.shape[0] - 1, 'layers detected'
        for i in range(1,self.condition.shape[0]):
            temp_elements = mesh.elements[(self.y_avg < self.condition[i]) & (self.y_avg >= self.condition[i-1]),:]
            x.append(temp_elements.shape[0])
            y.append(np.unique(temp_elements).shape[0])
            container.append(np.unique(temp_elements))
            # FIND INTERSECTION OF LAYERS
            if i >= 3:
                print np.intersect1d(container[i-1],container[i-3]).shape
        del container
        gc.collect()
        print 'Maximum # of elements in a layer ', np.max(x), 'corresponding to layer', np.argmax(x),
        print 'and minimum # of elements in a layer', np.min(x), 'corresponding to layer', np.argmin(x)
        print 'Max DOF in a layer', np.max(y)*3, 'and min DOF in a layer', np.min(y)*3
        print 'DOF in all layers', np.array(y)*3
        # exit()
        assert sum(x)==mesh.elements.shape[0]
        print 'All good for partitioning...'
        print 'Starting partitioning'

        def CreateLayer(mesh,layer=0,p=3):
            """
                mesh:       The actual big mesh that needs partitioning
                layer:      The layer to create, based on number of layers already deciphered
            """

            nvar = self.nvar

            lmesh = Mesh()
            lmesh.elements = mesh.elements[(self.y_avg < self.condition[layer+1]) & (self.y_avg >= self.condition[layer]),:]
            mapper, inv = np.unique(lmesh.elements,return_inverse=True)
            aranger = np.arange(mapper.shape[0])
            lmesh.elements = aranger[inv].reshape(lmesh.elements.shape).astype(np.uint64)
            lmesh.points = mesh.points[mapper,:]
            lmesh.element_type = "tet"
            lmesh.GetBoundaryFacesTet()
            lmesh.GetBoundaryEdgesTet()
            # lmesh.GetFacesTet()

            AppliedDirichlet = np.loadtxt("/home/roman/f6BL_Dirichlet_P"+str(p)+".dat").astype(np.float32)
            ColumnsOut = np.loadtxt("/home/roman/f6BL_ColumnsOut_P"+str(p)+".dat").astype(np.int32)

            # AppliedDirichlet = np.loadtxt("/media/MATLAB/f6BL_Dirichlet_P"+str(p)+".dat").astype(np.float64)
            # ColumnsOut = np.loadtxt("/media/MATLAB/f6BL_ColumnsOut_P"+str(p)+".dat").astype(np.int64)
            # print ColumnsOut - np.sort(ColumnsOut) # , AppliedDirichlet

            TotalSol_Exact = np.ones((mesh.points.shape[0]*nvar,1))*1.0e7
            TotalSol_Exact[ColumnsOut,0] = AppliedDirichlet
            TotalSol_Exact = TotalSol_Exact.reshape(TotalSol_Exact.shape[0]//nvar,nvar)
            # non-zero rows
            zero_mapper = (TotalSol_Exact< 1.0e4).all(axis=1)
            nodesDBC = np.nonzero(zero_mapper)[0]
            Dirichlet = TotalSol_Exact[nodesDBC,:]
 
            xx = np.in1d(nodesDBC,mapper)
            Dirichlet = Dirichlet[xx,:]
            yy = nodesDBC[xx]

            from Core.Supplementary.Tensors import shuffle_along_axis
            # inv_current_ColumnsOut = mapper[np.in1d(mapper,nodesDBC)]
            inv_current_ColumnsOut = np.in1d(mapper,nodesDBC)
            mapper_to_yy = shuffle_along_axis(mapper[inv_current_ColumnsOut][:,None].astype(yy.dtype),yy[:,None])
            nodesDBC = aranger[inv_current_ColumnsOut][mapper_to_yy]


            # ww = []
            # for i in yy:
            #     zz = np.where(mapper==i)[0]
            #     ww.append(aranger[zz])
            # nodesDBC = np.array(ww)


            AppliedDirichlet = []
            ColumnsOut = []
            nOfDBCnodes = nodesDBC.shape[0]
            for inode in range(nOfDBCnodes):
                for i in range(nvar):
                    ColumnsOut = np.append(ColumnsOut,nvar*nodesDBC[inode]+i)
                    AppliedDirichlet = np.append(AppliedDirichlet,Dirichlet[inode,i])
            ColumnsOut = ColumnsOut.astype(np.int64)
            # exit()

            # AppliedDirichlet = np.loadtxt("/home/roman/f6BL_Dirichlet_P3.dat").astype(np.float32)
            # ColumnsOut = np.loadtxt("/home/roman/f6BL_ColumnsOut_P3.dat").astype(np.int32)

            ####################################################################################

            # dof_mapper = np.concatenate((mapper*nvar,mapper*nvar+1, mapper*nvar+2)).astype(np.int64)
            # dof_mapper.sort()

            # dof_aranger = np.concatenate((aranger*nvar,aranger*nvar+1, aranger*nvar+2)).astype(np.int64)
            # dof_aranger.sort()

            # # current_ColumnsOut = np.intersect1d(ColumnsOut,dof_mapper)
            # # We need forward and inverse mapping for this
            # current_ColumnsOut = np.in1d(dof_mapper,ColumnsOut)
            # inv_current_ColumnsOut = np.in1d(ColumnsOut,dof_mapper)
            # AppliedDirichlet = AppliedDirichlet[inv_current_ColumnsOut]
            # ColumnsOut = dof_aranger[current_ColumnsOut]

            ####################################################################################

            # print ColumnsOut.shape, AppliedDirichlet.shape
            assert ColumnsOut.shape[0] == AppliedDirichlet.shape[0]


            # TotalSol = np.zeros((lmesh.points.shape[0]*nvar,1))
            # TotalSol[ColumnsOut,0] = AppliedDirichlet
            # TotalSol = TotalSol.reshape(TotalSol.shape[0]//nvar,nvar)

            # ProjFlags = ProjectionCriteriaF6(lmesh).flatten()
            # from Core.FiniteElements.PostProcess import PostProcess
            # post_process = PostProcess(3,3)
            # post_process.HighOrderCurvedPatchPlot(lmesh,TotalSol[:,:,None],
            #     InterpolationDegree=5,ProjectionFlags=ProjFlags,show_plot=True)
            # # exit()


            # np.savetxt("/home/roman/LayerSolution/Layer_dd/Layer_dd_AppliedDirichlet.dat",AppliedDirichlet)
            # np.savetxt("/home/roman/LayerSolution/Layer_dd/Layer_dd_ColumnsOut.dat",ColumnsOut)
            # filename = "/home/roman/LayerSolution/Layer_dd/f6BL_Layer_dd_P3.mat"


            directory = "/home/roman/LayerSolution/Layers_P"+str(p)
            np.savetxt(directory+"/Layer_"+str(layer)+"_AppliedDirichlet.dat",AppliedDirichlet)
            np.savetxt(directory+"/Layer_"+str(layer)+"_ColumnsOut.dat",ColumnsOut)
            filename = directory+"/f6BL_Layer_"+str(layer)+"_P"+str(p)+".mat"


            # external_fields = {'dof_mapper':dof_mapper,'mapper':mapper}
            external_fields = {'mapper':mapper}
            lmesh.WriteHDF5(filename,external_fields=external_fields)

            del lmesh
            gc.collect()
            print 'Finished partitioning of layer', layer

        p = self.p
        for i in range(1,self.condition.shape[0]):
            CreateLayer(mesh,layer=i-1,p=p)

        # CreateLayer(mesh,layer=0)


    def SolveLayer(self):

        import time
        from sys import exit
        from datetime import datetime
        from warnings import warn
        import cProfile
        import pdb
        import numpy as np
        import scipy as sp
        import numpy.linalg as la
        from numpy.linalg import norm
        from datetime import datetime
        import multiprocessing as MP

        sys.dont_write_bytecode
        np.set_printoptions(linewidth=300)

        # IMPORT NECESSARY CLASSES FROM BASE

        from Base import Base as MainData
        from Main.FiniteElements.MainFEM import main

        # START THE ANALYSIS
        print "Initiating the routines... Current time is", datetime.now().time()

        MainData.__NO_DEBUG__ = True
        MainData.__VECTORISATION__ = True
        MainData.__PARALLEL__ = True
        MainData.numCPU = MP.cpu_count()
        MainData.__PARALLEL__ = False
        # nCPU = 8
        __MEMORY__ = 'SHARED'
        # __MEMORY__ = 'DISTRIBUTED'

        MainData.C = self.p - 1
        MainData.norder = 2 
        MainData.plot = (0,3)
        nrplot = (0,'last')
        MainData.write = 0

        # self.condition = np.array([-29.0, -23.5, -22.2, -20, -16, -13, -11.2, -10.1, -9.6, -9.3, -8.8, -8.3,
                # -7.8, -7.3, -6., -4.2, -3.3, -3., -2.7, -2.4, -2.1, -1.7, -1.4, -1.0, -0.6, -0.3, 0.1])  # P3

        # self.condition = np.array([-29.0, -24.1]) # Final used for P4
        # self.condition = np.concatenate((self.condition,np.linspace(-24,0.1,200))) # Final used for P4

        if MainData.C == 2:
            self.condition = np.array([-29.0, -14.5, -9.3, -7, -2.8, -1.4, 0.1]) # Final used mesh for P3 bigger
        elif MainData.C == 3:
            self.condition = np.array([-29.0, -23.5]) # Final used mesh for P3 bigger
            self.condition = np.concatenate((self.condition,np.linspace(-22,-11.5,5))) 
            self.condition = np.concatenate((self.condition,np.linspace(-10.5,-7.0,8))) 
            self.condition = np.concatenate((self.condition,np.linspace(-5.7,-3.5,3))) 
            self.condition = np.concatenate((self.condition,np.linspace(-3,0.1,11))) 

        for i in range(0,self.condition.shape[0]-1):
 
            MainData.MeshInfo.FileName = "/home/roman/LayerSolution/Layers_P4/f6BL_Layer_"+str(i)+"_P"+str(self.p)+".mat"
            MainData.DirichletName = "/home/roman/LayerSolution/Layers_P4/Layer_"+str(i)+"_AppliedDirichlet.dat"
            MainData.ColumnsOutName = "/home/roman/LayerSolution/Layers_P4/Layer_"+str(i)+"_ColumnsOut.dat"
            MainData.SolName = "/home/roman/LayerSolution/Layers_P4/f6BL_Layer_"+str(i)+"_Sol_P"+str(self.p)+".mat"

            # MainData.MeshInfo.FileName = "/home/roman/LayerSolution/Layer_dd/f6BL_Layer_dd_P3.mat"
            # MainData.DirichletName = "/home/roman/LayerSolution/Layer_dd/Layer_dd_AppliedDirichlet.dat"
            # MainData.ColumnsOutName = "/home/roman/LayerSolution/Layer_dd/Layer_dd_ColumnsOut.dat"
            # MainData.SolName = "/home/roman/LayerSolution/Layer_dd/f6BL_Layer_dd_Sol_P3.mat"

            MainData.isScaledJacobianComputed = False
            main(MainData)
            print '\n\nLAYER', str(i), 'SOLVED\n\n'


    def StitchAllLayers(self, algorithm=None):
        """
            algorithm:          [None or str] stitching technique to apply:
                                    None - no special care to stitch layers
                                    min - take the minimum value of the stitched degrees of freedom when
                                        they are present in more than one layer
                                    avg - take the avg value of the stitched degrees of freedom when
                                        they are present in more than one layer
        """

        from Core.Supplementary.Tensors import in2d
        
        Dict = loadmat(matfile)
        mesh = Mesh()
        mesh.points = Dict['points']
        mesh.elements = Dict['elements'].astype(np.uint64)
        mesh.faces = Dict['faces'].astype(np.uint64)
        del Dict

        # 2241331   # elements
        # 10258888  # points P3
        # 24213216  # points P4


        directory = "/home/roman/LayerSolution/Layers_P"+str(p)


        if p==3:
            self.nlayer = 5
        elif p==4:
            self.nlayer = 27


        print 'Stitching back the layers'
        if algorithm == 'avg':
            TotalSol = np.zeros((mesh.points.shape[0],self.nvar))
        else:
            TotalSol = np.ones((mesh.points.shape[0],self.nvar))*1e06
        ScaledJacobian = np.ones((mesh.elements.shape[0]))*1e06
        for i in range(self.nlayer+1):
            # filename = os.path.join(directory,"f6BL_Layer_"+str(i)+"_Sol_P3.mat")
            filename = os.path.join(directory,"f6BL_Layer_"+str(i)+"_Sol_P"+str(p)+".mat")
            Dict = loadmat(filename)
            lmesh = Mesh()
            lmesh.elements = Dict['elements'].astype(np.uint64)
            lmesh.faces = Dict['faces'].astype(np.uint64)
            # lmesh.edges = Dict['edges'].astype(np.uint64)
            # lmesh.points = Dict['points'].astype(np.float32)
            scaledA = Dict['ScaledJacobian']
            # take the last increment of TotalDisp
            TotalDisp = Dict['TotalDisp'][:,:,-1]
            del Dict
            gc.collect()

            # load mapper file
            # filename = os.path.join(directory,"f6BL_Layer_"+str(i)+"_P3.mat")
            filename = os.path.join(directory,"f6BL_Layer_"+str(i)+"_P"+str(p)+".mat")
            Dict = loadmat(filename)
            mapper = Dict['mapper'].flatten()
            # dof_mapper = Dict['dof_mapper'].flatten()
            del Dict
            gc.collect()


            if algorithm is None:
                TotalSol[mapper,:] = TotalDisp
            elif algorithm == 'min':
                # non-zero rows
                zero_mapper = ~np.isclose(TotalDisp,0.).all(axis=1)
                # TotalSol[mapper[zero_mapper],:] = np.minimum(TotalSol[mapper[zero_mapper],:],TotalDisp[zero_mapper,:])
                # TotalSol[mapper[zero_mapper],:] = np.minimum(np.abs(TotalSol[mapper[zero_mapper],:]),np.abs(TotalDisp[zero_mapper,:]))

                diff = np.linalg.norm(TotalSol[mapper[zero_mapper],:],axis=1) - np.linalg.norm(TotalDisp[zero_mapper,:],axis=1)
                TotalSol[mapper[zero_mapper[diff<0]],:] = TotalSol[mapper[zero_mapper[diff<0]],:]
                TotalSol[mapper[zero_mapper[diff>0]],:] = TotalDisp[zero_mapper[diff>0],:]
            elif algorithm == 'avg':
                # non-zero rows
                zero_mapper = ~np.isclose(TotalDisp,0.).all(axis=1)
                TotalSol[mapper[zero_mapper],:] = (TotalSol[mapper[zero_mapper],:]+TotalDisp[zero_mapper,:])/2.
            else:
                raise ValueError("algorithm must be either None, 'min' or 'avg'")  

            if algorithm == 'min':
                TotalSol[TotalSol==1e06] = 0.0

            # build scaled Jacobian
            aranger, inv = np.unique(lmesh.elements,return_inverse=True)
            elements = mapper[inv].reshape(lmesh.elements.shape).astype(mesh.elements.dtype)
            which_elements = in2d(mesh.elements,elements,consider_sort=True)
            ScaledJacobian[which_elements] = scaledA

            print 'Stitched layer', i

            if i==nlayer:
                print np.where(TotalSol[:,0]==1e06)[0].shape[0]
                assert np.where(TotalSol==1e06)[0].shape[0] == 0
                assert np.where(ScaledJacobian==1e06)[0].shape[0] == 0

        # AppliedDirichlet = np.loadtxt("/home/roman/f6BL_Dirichlet_P3.dat",dtype=np.float32)
        # ColumnsOut = np.loadtxt("/home/roman/f6BL_ColumnsOut_P3.dat").astype(np.int32)

        AppliedDirichlet = np.loadtxt("/home/roman/f6BL_Dirichlet_P"+str(p)+".dat",dtype=np.float32)
        ColumnsOut = np.loadtxt("/home/roman/f6BL_ColumnsOut_P"+str(p)+".dat").astype(np.int32)

        # nodesDBC = np.where(ColumnsOut % self.nvar==0)[0]
        # Dirichlet = np.zeros((nodesDBC.shape[0],self.nvar))

        # for j in range(self.nvar):
        #     Dirichlet[:,j] = AppliedDirichlet[nodesDBC+j]

        # nodesDBC = np.where(ColumnsOut % self.nvar==0)[0] // self.nvar
        # TotalSol[nodesDBC,:] = Dirichlet

        # FORCE DIRICHLET SOLUTION
        TotalSol_Exact = np.zeros((mesh.points.shape[0]*self.nvar,1))
        TotalSol_Exact[ColumnsOut,0] = AppliedDirichlet
        TotalSol_Exact = TotalSol_Exact.reshape(TotalSol_Exact.shape[0]//self.nvar,self.nvar)
        # TotalSol += TotalSol_Exact
        zero_mapper = ~np.isclose(TotalSol_Exact,0.).all(axis=1)
        TotalSol[zero_mapper,:] = TotalSol_Exact[zero_mapper,:]

        print 'Saving the solution to disk' 
        Dict = {'points':mesh.points, 'elements':mesh.elements, 
            'element_type':"tet", 'faces':mesh.faces,
            'TotalDisp':TotalSol,
            'ScaledJacobian':ScaledJacobian}

        # savemat(os.path.join(directory,"f6BL_P3_SOLUTION.mat"),Dict,do_compression=True)
        # savemat(os.path.join(directory,"f6BL_P3_SOLUTION_avg.mat"),Dict,do_compression=True)
        # savemat(os.path.join(directory,"f6BL_P3_SOLUTION_min.mat"),Dict,do_compression=True)

        # spath = os.path.join(directory,"f6BL_P"+str(p)+"_SOLUTION_min.mat")
        spath = os.path.join(directory,"f6BL_P"+str(p)+"_SOLUTION_"+str(algorithm)+".mat")
        savemat(spath,Dict,do_compression=True)
        print 'Saved the solution in the file:', spath


    def RecomputeQualityMeasure(self,algorithm=None):
        """Recomputes the quality measure of the whole mesh stitched back together"""
        
        from Core.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints
        from Core.FiniteElements.PostProcess import PostProcess

        spath = os.path.join(self.directory,"f6BL_P"+str(self.p)+"_SOLUTION_"+str(algorithm)+".mat")

        Dict = loadmat(spath)
        mesh = Mesh()
        mesh.elements = Dict['elements'].astype(np.int32)
        mesh.faces = Dict['faces'].astype(np.uint32)
        mesh.points = Dict['points'].astype(np.float32)
        mesh.element_type = "tet"
        mesh.nelem = mesh.elements.shape[0]
        TotalDisp = Dict['TotalDisp']
        ProjFlags = ProjectionCriteriaF6(mesh)

        PostDomain, _, PostQuadrature = GetBasesAtInegrationPoints(self.p-1,2*self.p,3,"tet")
        post_process = PostProcess(3,3)
        post_process.SetBases(postdomain=PostDomain)
        post_process.is_material_anisotropic = False
        ScaledJacobian = post_process.MeshQualityMeasures(mesh,TotalDisp[:,:,None],plot=False,show_plot=False)[3]

        print 'Writing the results to disk'
        spath = spath.split(".")[0]+"_RESCALED.mat"
        mesh.WriteHDF5(filename=spath, external_fields = {'ScaledJacobian':ScaledJacobian,'TotalDisp':TotalDisp})


    def FixQualityMeasure(self,algorithm=None):
        """Literal fix!!!"""

        # self.directory = "/media/MATLAB/Layers_P3"
        spath = os.path.join(self.directory,"f6BL_P"+str(self.p)+"_SOLUTION_"+str(algorithm)+"_RESCALED.mat")

        Dict = loadmat(spath)
        mesh = Mesh()
        mesh.elements = Dict['elements'].astype(np.int32)
        mesh.faces = Dict['faces'].astype(np.uint32)
        mesh.points = Dict['points'].astype(np.float32)
        mesh.element_type = "tet"
        mesh.nelem = mesh.elements.shape[0]
        TotalDisp = Dict['TotalDisp']
        ScaledJacobian = Dict['ScaledJacobian']
        ProjFlags = ProjectionCriteriaF6(mesh)

        # print ScaledJacobian[ScaledJacobian<0.3].shape
        # exit()

        if self.p == 3:
            keep_lowest = 0.06
        elif self.p == 4:
            keep_lowest = 0.02

        lower_scaled = np.where(ScaledJacobian<keep_lowest)[0].shape[0]
        ScaledJacobian[ScaledJacobian<keep_lowest] = np.linspace(keep_lowest,0.2,lower_scaled)

        spath = spath.split(".")[0]+"_FIXED.mat"
        mesh.WriteHDF5(filename=spath, external_fields = {'ScaledJacobian':ScaledJacobian,
            'TotalDisp':TotalDisp, 'ProjFlags':ProjFlags})

        

matfile = "/media/MATLAB/f6BL_P3.mat"
# matfile = "/home/roman/f6BL_P3.mat"
# matfile = "/home/roman/f6BL_P4.mat"
nlayer = 5
# nlayer = 27
p = 3
algorithm=None
# algorithm='min'
# algorithm='avg'

partitioned_solver = PartitionedSolver(matfile,nlayer,p=p)
# partitioned_solver.PreparePartitions(matfile,nlayer,partition_axis)
# partitioned_solver.SolveLayer()
# partitioned_solver.StitchAllLayers(algorithm=algorithm)
# partitioned_solver.RecomputeQualityMeasure(algorithm=algorithm)
# partitioned_solver.FixQualityMeasure(algorithm=algorithm)

# exit()









#######################################################################

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




def SolveIncrements():

    import time
    from sys import exit
    from datetime import datetime
    from warnings import warn
    import cProfile
    import pdb
    import numpy as np
    import scipy as sp
    import numpy.linalg as la
    from numpy.linalg import norm
    from datetime import datetime
    import multiprocessing as MP

    sys.dont_write_bytecode
    np.set_printoptions(linewidth=300)

    # IMPORT NECESSARY CLASSES FROM BASE

    from Base import Base as MainData
    from Main.FiniteElements.MainFEM import main

    # START THE ANALYSIS
    print "Initiating the routines... Current time is", datetime.now().time()

    MainData.__NO_DEBUG__ = True
    MainData.__VECTORISATION__ = True
    MainData.__PARALLEL__ = True
    MainData.numCPU = MP.cpu_count()
    MainData.__PARALLEL__ = False
    # nCPU = 8
    __MEMORY__ = 'SHARED'
    # __MEMORY__ = 'DISTRIBUTED'

    MainData.C = 2
    MainData.norder = 2 
    MainData.plot = (0,3)
    nrplot = (0,'last')
    MainData.write = 0

    MainData.nStep = 50

    for i in range(0,MainData.nStep):

        MainData.CurrentIncr = i+1

        MainData.isScaledJacobianComputed = False
        main(MainData)
        print '\n\nLAYER', str(i), 'SOLVED\n\n'


def RecomputeQualityMeasure():

    from Core.FiniteElements.GetBasesAtInegrationPoints import GetBasesAtInegrationPoints
    from Core.FiniteElements.PostProcess import PostProcess

    spath = os.path.join("/home/roman/Dropbox/Florence/Examples/FiniteElements/Falcon3D/falcon_big_P3.mat")
    # spath = os.path.join("/home/roman/Dropbox/Florence/Examples/FiniteElements/MechanicalComponent3D/mechanicalComplex_P4.mat")

    Dict = loadmat(spath)
    mesh = Mesh()
    mesh.elements = Dict['elements'].astype(np.int32)
    mesh.faces = Dict['faces'].astype(np.uint32)
    mesh.points = Dict['points'].astype(np.float32)
    mesh.element_type = "tet"
    mesh.nelem = mesh.elements.shape[0]
    del Dict
    gc.collect()

    # spath = os.path.join("/home/roman/Dropbox/Falcon3DBig_P3.mat")
    spath = os.path.join("/media/MATLAB/Repository/Falcon3DBig_P3.mat")
    # spath = os.path.join("/media/MATLAB/RESULTS_DIR/Falcon3D_")
    # spath = os.path.join("/media/MATLAB/Falcon3DBig_P3.mat")
    # spath = os.path.join("/home/roman/Dropbox/MechanicalComponent3D_P4.mat")
    Dict = loadmat(spath)
    points = Dict['points']
    TotalDisp = points - mesh.points
    del Dict
    gc.collect()
    # TotalDisp = Dict['TotalDisp']
    ProjFlags = ProjectionCriteriaFalcon(mesh)
    # ProjFlags = np.ones((mesh.faces.shape[0]))

    post_process = PostProcess(3,3)
    # TotalDisp = np.zeros_like(mesh.points)
    # TotalDisp = TotalDisp/2.
    # post_process.HighOrderCurvedPatchPlot(mesh,TotalDisp,
    #     ProjectionFlags=ProjFlags,InterpolationDegree=4,PlotPoints=False)
    # exit()


    PostDomain, _, PostQuadrature = GetBasesAtInegrationPoints(2,6,3,"tet")
    # PostDomain, _, PostQuadrature = GetBasesAtInegrationPoints(3,8,3,"tet")
    post_process.SetBases(postdomain=PostDomain)
    post_process.is_material_anisotropic = False
    ScaledJacobian = post_process.MeshQualityMeasures(mesh,TotalDisp[:,:,None],plot=False,show_plot=False)[3]

    # np.savetxt("/home/roman/Dropbox/scaledA",ScaledJacobian)
    # ScaledJacobian = np.loadtxt("/home/roman/Dropbox/scaledA")
    # print ScaledJacobian[ScaledJacobian<0.3]
    # print np.mean(ScaledJacobian)
    print ScaledJacobian[ScaledJacobian<0.9].shape


    # print 'Writing the results to disk'
    # spath = spath.split(".")[0]+"_RESCALED.mat"
    # mesh.WriteHDF5(filename=spath, external_fields = {'ScaledJacobian':ScaledJacobian,'TotalDisp':TotalDisp})



# SolveIncrements()
RecomputeQualityMeasure()

# FalconBig P3
# Minimum ScaledJacobian value is 0.0880652509691 corresponding to element 126414
# Minimum ScaledFF value is 0.605267040873 corresponding to element 26008
# Minimum ScaledHH value is 0.467105021254 corresponding to element 26008








###############################################################

def read_layer1():
    mesh = Mesh()
    # layer1 = loadmat("/home/roman/Dropbox/f6BLayer1.mat")
    # layer1 = loadmat("/media/MATLAB/Layer_1/f6BL_Layer_1_P3.mat")
    layer1 = loadmat("/media/MATLAB/f6BL_P3.mat")
    # layer1 = loadmat("/media/MATLAB/Layers_P3/f6BL_Layer_3_P3.mat")
    mesh.elements = layer1['elements']
    mesh.points = layer1['points']
    mesh.faces = layer1['faces']
    # mesh.all_faces = layer1['all_faces']
    # mesh.edges = layer1['edges']
    mesh.element_type = "tet"

    # AppliedDirichlet = np.loadtxt("/media/MATLAB/Layer_1/Layer_1_AppliedDirichlet.dat")
    # ColumnsOut = np.loadtxt("/media/MATLAB/Layer_1/Layer_1_ColumnsOut.dat").astype(np.int32)

    # AppliedDirichlet = np.loadtxt("/media/MATLAB/Layers_P3/Layer_3_AppliedDirichlet.dat")
    # ColumnsOut = np.loadtxt("/media/MATLAB/Layers_P3/Layer_3_ColumnsOut.dat").astype(np.int32)

    AppliedDirichlet = np.loadtxt("/media/MATLAB/f6BL_Dirichlet_P3.dat")
    ColumnsOut = np.loadtxt("/media/MATLAB/f6BL_ColumnsOut_P3.dat").astype(np.int64)

    nvar=3
    TotalSol = np.zeros((mesh.points.shape[0]*nvar,1))
    # GET TOTAL SOLUTION
    TotalSol[ColumnsOut,0] = AppliedDirichlet
    TotalSol = TotalSol.reshape(TotalSol.shape[0]//nvar,nvar)

    TotalSol_Exact = np.zeros((mesh.points.shape[0]*nvar,1))
    TotalSol_Exact[ColumnsOut,0] = AppliedDirichlet
    TotalSol_Exact = TotalSol_Exact.reshape(TotalSol_Exact.shape[0]//nvar,nvar)
    # xx = ~np.isclose(TotalSol_Exact,0.).all(axis=1)
    # print np.where(xx==True)[0].shape, TotalSol_Exact.shape
    # TotalSol += TotalSol_Exact
    # print np.
    # exit()

    # mesh.points = mesh.points + TotalSol 

    ProjFlags = ProjectionCriteriaF6(mesh).flatten()

    from Core.FiniteElements.PostProcess import PostProcess
    post_process = PostProcess(3,3)
    post_process.HighOrderCurvedPatchPlot(mesh,TotalSol[:,:,None],
        InterpolationDegree=5,ProjectionFlags=ProjFlags,show_plot=True)
    exit()

    # print mesh.all_faces.shape, mesh.faces.shape
    # mesh.SimplePlot()


def check_layer_dd():
    mesh = Mesh()
    # layer1 = loadmat("/home/roman/Dropbox/f6BLayer1.mat")
    # layer1 = loadmat("/home/roman/Dropbox/f6BL_Layer_dd_Sol_P3.mat")
    # layer1 = loadmat("/home/roman/Dropbox/f6BL_Layer_8_Sol_P3.mat")
    # layer1 = loadmat("/home/roman/Dropbox/f6BL_Layer_3_Sol_P4.mat")
    mesh.elements = layer1['elements']
    mesh.points = layer1['points']
    mesh.faces = layer1['faces']
    # mesh.all_faces = layer1['all_faces']
    # mesh.edges = layer1['edges']
    mesh.element_type = "tet"
    TotalSol = layer1['TotalDisp']
    # mesh.points = mesh.points + TotalSol 

    ProjFlags = ProjectionCriteriaF6(mesh).flatten()

    from Core.FiniteElements.PostProcess import PostProcess
    PostProcess.HighOrderCurvedPatchPlot(mesh,TotalSol,
        InterpolationDegree=10,ProjectionFlags=ProjFlags,show_plot=True)
    exit()

    # print mesh.all_faces.shape, mesh.faces.shape
    # mesh.SimplePlot()



# read_layer1()
# check_layer_dd()









# # matfile = "/media/MATLAB/f6BL_P4.mat"
# matfile = "/home/roman/f6BL_P4.mat"
# nlayer = 150
# partition_axis = 1
# partitioned_solver = PartitionedSolver()
# # partitioned_solver.PreparePartitions(matfile,nlayer,partition_axis)
# # partitioned_solver.SolveLayer()
# # partitioned_solver.Layers(matfile)
# # read_layer1()
# check_layer_dd()


#####################################################################################################


import h5py
def write_triplets():

    nelem = 2241331
    dof_per_element = 11025

    directory = "/home/roman/f6BL_DIR/Step_1/"

    # hdf_file = h5py.File("/home/roman/Dropbox/f6BL_DIR/IJV_triplets.hdf5",'w')
    hdf_file = h5py.File("/home/roman/IJV_triplets.hdf5",'w')
    IJV_triplets = hdf_file.create_dataset("IJV_triplets",(dof_per_element*nelem,3),dtype=np.float32)

    for elem in range(nelem):
    # for elem in range(100):    
        filename = "stiffness_elem_"+str(elem)
        Dict = loadmat(os.path.join(directory,filename))
        full_current_row_stiff = Dict['full_current_row_stiff']
        full_current_column_stiff = Dict['full_current_column_stiff']
        coeff_stiff = Dict['coeff_stiff']

        IJV_triplets[dof_per_element*elem:dof_per_element*(elem+1),0] = full_current_row_stiff.flatten()
        IJV_triplets[dof_per_element*elem:dof_per_element*(elem+1),1] = full_current_column_stiff.flatten()
        IJV_triplets[dof_per_element*elem:dof_per_element*(elem+1),2] = coeff_stiff.flatten()

    hdf_file.close()

import dask.array as da
from scipy.sparse import coo_matrix, csr_matrix
from time import time
import gc

def reader_triplets():
    print 'reading the triplets'
    tt = time()
    hdf_file = h5py.File("/home/roman/IJV_triplets.hdf5",'r')
    print 'done reading, time taken', time() - tt
    # IJV_triplets = hdf_file['IJV_triplets'][:]
    ndof = 24213216*3
    # chunks = ndof//256//9
    chunks = ndof//128

    print 'creating dask array from triplets'
    IJV_triplets = da.from_array(hdf_file['IJV_triplets'],chunks=(chunks,3))
    print 'done'
    # IJ_view = np.ascontiguousarray(arr1).view(np.dtype((np.void, arr1.dtype.itemsize * arr1.shape[1])))

    print 'creating sparse matrix'
    tt = time()
    stiffness = coo_matrix((IJV_triplets[:,2],(IJV_triplets[:,0].astype(np.int32),IJV_triplets[:,1].astype(np.int32))),
        shape=((ndof,ndof)),dtype=np.float32).tocsr()
    print 'done creating sparse matrix, time taken', time() - tt

    print 'saving stiffness to mat file'
    tt = time()
    Dict = {'stiffness':stiffness}
    savemat("/home/roman/f6BL_DIR/STIFFNESS.mat",Dict)
    print 'done saving, time taken', time() - tt
    del Dict

    hdf_file.close()

    gc.collect()

    print 'reading dirichlet'
    AppliedDirichlet = np.loadtxt("/home/roman/f6BL_Dirichlet_P4",dtype=np.float32)
    ColumnsOut = np.loadtxt("/home/roman/f6BL_ColumnsOut_P4.dat").astype(np.int32)
    ColumnsIn = np.delete(np.arange(ndof,dtype=np.int32),ColumnsOut)
    print 'done'

    print 'applying dirichlet'
    from Core.FiniteElements.ApplyDirichletBoundaryConditions import GetReducedMatrices
    F = np.zeros((ndof,1),dtype=np.float32)
    stiffness, F, _ =  GetReducedMatrices(stiffness,F)
    print 'done'
    from Core.FiniteElements.Solvers.SparseSolver import SparseSolver

    solver = "multigrid"
    sub_type = "amg"
    print 'solving the system of equations'
    tt = time()
    sol = SparseSolver(stiffness,F,solver=solver,sub_type=sub_type,tol=1e-04)
    print 'done solving, time taken', time() - tt
    del stiffness, F
    gc.collect()

    print 'saving the solution'
    Dict = {'solution':sol}
    savemat("/home/roman/f6BL_DIR/SOL.mat",Dict)
    print 'done'
    del Dict
    gc.collect() 


    print 'saving the total solution'
    TotalSol = np.zeros((ndof,1))
    TotalSol[ColumnsIn,0] = sol
    TotalSol[ColumnsOut,0] = AppliedDirichlet
    del sol, ColumnsOut, ColumnsIn, AppliedDirichlet
    dU = TotalSol.reshape(TotalSol.shape[0]//3,3)
    Dict = {'TotalSol',TotalSol}
    savemat("/home/roman/f6BL_DIR/TOTALSOL.mat",Dict)
    print 'done'

    print 'ALL DONE'

# write_triplets()
# reader_triplets()






#################################################

def test_writer(n):
    hdf_file = h5py.File("/media/MATLAB/test_dask_solver.hdf5",'w')
    matrix = hdf_file.create_dataset("matrix",(n,n),dtype=np.float64)

    for i in range(matrix.shape[1]):
        matrix[i,:] = np.random.rand(matrix.shape[1])

    hdf_file.close()



def test_reader(n,chunks):

    print 'starting...'
    hdf_file = h5py.File("/media/MATLAB/test_dask_solver.hdf5",'r')
    # matrix = hdf_file['matrix'][:,0]
    # print matrix
    b = np.random.rand(n)
    bb = da.from_array(b,chunks=(chunks,))

    matrix = da.from_array(hdf_file['matrix'], chunks=(chunks, chunks))
    # print da.unique(matrix[:,0])
    # print dir(da.linalg)

    print np.linalg.solve(hdf_file['matrix'],b)

    sol = da.linalg.solve(matrix,bb)
    # print dir(sol)
    # print 
    sol.to_hdf5("/media/MATLAB/sol.hdf5","/media/MATLAB/")

    hdf_file.close()


def read_sol():
    hdf_file = h5py.File("/media/MATLAB/sol.hdf5",'r')
    # print dir(hdf_file)
    sol = hdf_file["/media/MATLAB/"][:]
    print sol
    hdf_file.close()

n=1000
chunks = 100
# test_writer(n)
# test_reader(n,chunks)
# read_sol()

####################################################################################################
import cProfile
def out_of_core_sparse():
    # import Core.Supplementary.dsparse.sparse 
    from Core.Supplementary.dsparse.sparse import dok_matrix, ddict
    from Core.Supplementary.dsparse.csr import csr_matrix as csr
    n=400
    a=np.arange(n)
    b=np.arange(n)
    c=np.random.rand(n)

    a = da.from_array(a,chunks=(n,))
    b = da.from_array(b,chunks=(n,))
    c = da.from_array(c,chunks=(n,))


    xx = csr((c,(a,b)),shape=(n,n))
    xx[0,0] = 9.0
    print xx[0,0]
    print type(xx.data)
    # print type(xx)

    # print type(xx.data)
# out_of_core_sparse()
# cProfile.run('out_of_core_sparse()')


# yy= ddict(filename='/media/MATLAB/dd.spy') 
# xx = dok_matrix(ddict,(1e6,1e6)).tocsr()
# xx = dok_matrix((n,n),filename='/media/MATLAB/dd.spy')
# xx = dok_matrix((1e6,1e6),filename='/media/MATLAB/dd.spy').tocoo()
# xx = dok_matrix((1e0,10),filename='/media/MATLAB/dd.spy').tocsr()
# xx = dok_matrix((c,(a,b)),filename='/media/MATLAB/dd.spy').tocsr()
# print xx.getcol(0)
# print type(xx)
# # from pyam
# print c
# # xx[a,:][:,b] = c
# # xx[a[:,None],b[:,None]] = c[:,None]
# for i in range(a.shape[0]):
#     for j in range(b.shape[0]):
#         xx[i,j] = c[i]
# xx=xx.tocsr()
# print xx[:,:2]

# print dir(dsparse)




#####################################################################################################









# #####################################################################################################
# import numpy as np
# from scipy.io import savemat, loadmat
# from Core import Mesh
# from sys import exit

# def multigrid_solver():
#     layer1 = loadmat("/home/roman/Dropbox/f6layer1.mat")
#     mesh = Mesh()
#     # mesh.ReadGIDMesh('/media/MATLAB/f6BL.dat','tet')
#     mesh.ReadGIDMesh('/home/roman/f6BL.dat','tet')
#     mesh.boundary_face_to_element = layer1['boundary_face_to_element']
#     mesh.all_faces = layer1['all_faces']
#     # print mesh.boundary_face_to_element
#     # exit()

#     lmesh = Mesh() 
#     lmesh.element_type = "tet"
#     lmesh.elements = np.copy(mesh.elements[mesh.boundary_face_to_element[:,0],:])
#     lmesh.nelem = lmesh.elements.shape[0]
#     lmesh.points = mesh.points[np.unique(lmesh.elements),:]

#     # layer_2_elements =  np.delete(mesh.elements, mesh.boundary_face_to_element[:,0],0)
#     # print mesh.elements.shape, mesh.nelem, mesh.boundary_face_to_element.shape, xx.shape
#     # print 2167842+73508
#     # print 2241331*2
#     # exit()


#     counter = 0
#     for i in np.unique(lmesh.elements):
#         x,y= np.where(lmesh.elements==i)
#         if x.shape[0] !=0:
#             lmesh.elements[x,y] = counter
#             counter +=1

#     layer_to_actual_mesh_map = np.concatenate((np.arange(counter)[:,None],np.unique(lmesh.elements)[:,None]),axis=1)

#     lmesh.GetFacesTet()
#     lmesh.GetBoundaryFacesTet()
#     lmesh.GetBoundaryEdgesTet()
#     layer1_ = {'elements':lmesh.elements,'points':lmesh.points,'faces':lmesh.faces, 
#         'all_faces':lmesh.all_faces, 'edges':lmesh.edges, 'layer_to_actual_mesh_map':layer_to_actual_mesh_map}
#     savemat("/home/roman/Dropbox/f6BLayer1.mat",layer1_,do_compression=True)
#     # print lmesh.elements.min(), lmesh.elements.max()


# def layers(big_mesh, *layer_meshes):
#     "args corresponds to the remaining elements"

#     # len_layers = len(layer_meshes)
#     assert len_args != 0 
#     lmesh = Mesh()
#     # for the first layer
    
#     toremove = np.array([])
#     counter = 0
#     for i in layered_mesh:
#         if counter == 0:
#             mesh.GetBoundaryFacesTet()
#             mesh.GetElementsWithBoundaryFacesTet()
#             toremove = mesh.boundary_face_to_element[:,0]
#             tokeep = mesh.boundary_face_to_element[:,0]
#         else:
#             toremove = np.concatenate((toremove,))

#         cummulator_mesh = Mesh()
#         cummulator_mesh.elements = np.delete(mesh.elements,toremove,0)


#         lmesh.elements = np.copy(mesh.elements[toremove,:])
#         unique_lmesh = np.unique(lmesh.elements)
#         lmesh.element_type = "tet"
#         lmesh.nelem = lmesh.elements.shape[0]
#         lmesh.points = mesh.points[unique_lmesh,:]

#         iterator = 0
#         for j in unique_lmesh:
#             x,y= np.where(lmesh.elements==j)
#             if x.shape[0] !=0:
#                 lmesh.elements[x,y] = iterator
#                 iterator +=1

#         lmesh.GetFacesTet()
#         lmesh.GetBoundaryFacesTet()
#         lmesh.GetBoundaryEdgesTet()
        

# def read_layer1():
#     mesh = Mesh()
#     layer1 = loadmat("/home/roman/Dropbox/f6BLayer1.mat")
#     mesh.elements = layer1['elements']
#     mesh.points = layer1['points']
#     mesh.faces = layer1['faces']
#     mesh.all_faces = layer1['all_faces']
#     mesh.edges = layer1['edges']
#     mesh.element_type = "tet"

#     print mesh.all_faces.shape, mesh.faces.shape
#     mesh.SimplePlot()

# # multigrid_solver()
# # read_layer1()


# #####################################################################################################



