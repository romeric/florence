#!/usr/bin/env python
""" Parameteric studeis"""


import imp, os, sys, time
from sys import exit
from datetime import datetime
import cProfile, pdb 
import numpy as np
import scipy as sp
import numpy.linalg as la
from numpy.linalg import norm
from datetime import datetime
import multiprocessing as MP
# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode

# IMPORT NECESSARY CLASSES FROM BASE
from Base import Base as MainData

from Main.FiniteElements.MainFEM import main

from scipy.io import savemat, loadmat


def runMain():

    # START THE ANALYSIS
    print "Initiating the routines... Current time is", datetime.now().time()

    MainData.__NO_DEBUG__ = True
    MainData.__VECTORISATION__ = True
    MainData.__PARALLEL__ = True
    MainData.numCPU = MP.cpu_count()
    # MainData.__PARALLEL__ = False
    # nCPU = 8
    __MEMORY__ = 'SHARED'
    # __MEMORY__ = 'DISTRIBUTED'

    MainData.C = 1
    MainData.norder = 2 
    MainData.plot = (0,3)
    nrplot = (0,'last')
    MainData.write = 0

    t_FEM = time.time()
    # nu = np.linspace(0.001,0.495,10)
    # nu = np.linspace(0.001,0.495,1)
    nu = np.linspace(0.35,0.495,1)
    E = 1e05
    E_A = 2.5*E
    G_A = E/2.

    ProblemPath = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/"

    ProblemDataFile = ["sd7003_Stretch25","sd7003_Stretch50",
        "sd7003_Stretch100","sd7003_Stretch200","sd7003_Stretch400",
        "sd7003_Stretch800","sd7003_Stretch1600"]
    nStep= [1,2,5,10,25,50]

    ProblemDataFile = ["sd7003_Stretch25"]
    nStep=[1]

    

    Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

    condA=np.zeros((3,len(nStep),len(ProblemDataFile),nu.shape[0]))
    scaledA = np.copy(condA)
    scaledAFF = np.copy(condA)
    scaledAHH = np.copy(condA)
    whole_scaledA = np.zeros((3,len(nStep),len(ProblemDataFile),nu.shape[0],4000))
    # LOOP OVER FORMULATIONS
    for m in range(2,3):

        if m==0:
            MainData.AnalysisType = "Linear"
            MainData.MaterialArgs.Type = "IncrementalLinearElastic"
        elif m==1:
            MainData.AnalysisType = "Linear"
            MainData.MaterialArgs.Type = "NeoHookean_2"
        elif m==2:
            MainData.AnalysisType = "Nonlinear"
            MainData.MaterialArgs.Type = "NeoHookean_2"

        # LOOP OVER INCREMENTS
        for k in range(len(nStep)):
            MainData.LoadIncrement = nStep[k]
            # LOOP OVER DATA FILES
            for i in range(len(ProblemDataFile)):
                MainData.MeshInfo.FileName = ProblemPath+ProblemDataFile[i]

                if MainData.C==5 and MainData.AnalysisType == "Nonlinear" \
                    and nStep[k]>6:
                    continue

                # LOOP OVER POISSON'S RATIOS
                for j in range(nu.shape[0]):
                    MainData.MaterialArgs.nu = nu[j]
                    MainData.MaterialArgs.E = E
                    MainData.MaterialArgs.E_A = E_A
                    MainData.MaterialArgs.G_A = G_A

                    print 'Poisson ratio is:', MainData.MaterialArgs.nu
                    print MainData.AnalysisType, MainData.MaterialArgs.Type, "LoadIncrements", MainData.LoadIncrement, 
                    print "FileName", MainData.MeshInfo.FileName.split("/")[-1]

                    
                    MainData.isScaledJacobianComputed = False
                    main(MainData,Results)  
                    print 
                    CondExists = getattr(MainData.solve,'condA',None)
                    # ScaledExists = getattr(MainData.solve,'scaledA',None)
                    scaledA[m,k,i,j] = np.min(MainData.ScaledJacobian)
                    scaledAFF[m,k,i,j] = np.min(MainData.ScaledFF)
                    scaledAHH[m,k,i,j] = np.min(MainData.ScaledHH)
                    whole_scaledA[m,k,i,j,:MainData.ScaledJacobian.shape[0]] = MainData.ScaledJacobian
                    if CondExists is not None:
                        condA[m,k,i,j] = MainData.solve.condA
                    else:
                        condA[m,k,i,j] = np.NAN

    Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
    Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
    Results['WholeScaledJacobian'] = whole_scaledA
    Results['ScaledFF'] = scaledAFF
    Results['ScaledHH'] = scaledAHH

    
    fname = "P"+str(MainData.C+1)+".mat"
    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'

    # print fpath+fname
    # savemat(fpath+fname,Results)

    t_FEM = time.time()-t_FEM
    print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
    # np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])

if __name__ == '__main__':

    Run = 0
    if Run:
        runMain()

    if not Run:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.mlab import griddata

        # marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
        marker = itertools.cycle(('o', 's', 'x'))

        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
        ## for Palatino and other serif fonts use:
        rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})


        p=2
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
        fname = "P"+str(p)+".mat"
        

        DictOutput =  loadmat(fpath+fname)   
        # print DictOutput
        # exit(0)
        scaledA = DictOutput['ScaledJacobian']
        condA = DictOutput['ConditionNumber']
        # nu = DictOutput['PoissonsRatios'][0]
        nu = np.linspace(0.001,0.5,10)*100
        p = DictOutput['PolynomialDegrees'][0]


        xmin = 0
        xmax = 50
        ymin = nu[0]
        ymax = nu[-1]

        # print DictOutput
        # for key, value in DictOutput.iteritems():
            # print key

        print np.min(DictOutput['WholeScaledJacobian'][0,1,0,0][:2171])
        # print scaledA.shape
        nStep= [1,2,5,10,25,50]

        scaledA  = np.zeros((6,10))
        for i in range(len(nStep)):
            for j in range(10):
                scaledA[i,j] = np.min(DictOutput['WholeScaledJacobian'][0,i,0,j][:2171])
        # print scaledA[:,0]
        # scaledA = scaledA.T
        # Z = scaledA
        # print scaledA
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False, shade=False)

        # scaledA = np.min(scaledA,axis=)
        # plt.plot(DictOutput['WholeScaledJacobian'][2,0,0,0])
        # plt.show()
        # exit()
        


        # scaledA = DictOutput['ScaledJacobian'][2,:,0,:]
        # scaledA = np.fliplr(scaledA)

        # print scaledA
        scaledA = np.flipud(scaledA)

        print scaledA
        # exit()
        # print scaledA.shape
        # scaledA = scaledA[2,:,0,:]
        # exit()
        # x = np.linspace(0.0,0.5,10)
        # y = np.array([1,2,5,10,25,50])
        # y = np.array([1,2,5])
        # Z = scaledA[0,:,0,:]
        # Z1 = scaledA[0,:,6,:]
        # print x.shape, y.shape, z.shape
        # exit()

        # X,Y,Z = np.meshgrid(x,y,z)

        # X,Y = np.meshgrid(x,y)

        # print X.shape, Z.shape, Y.shape
        # print Z.shape
        # exit()
        # 
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False, shade=False)
        # surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False, shade=True)
        # surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)

        # ax.view_init(90,0)
        # plt.axis('off')


        # ax.set_ylabel(r'$Increments$',fontsize=18)
        # ax.set_xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=18)
        # ax.set_zlabel(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=18)

        # 

        # tick_locs = [1.0,2.0,5.0,10.0,24.0,49.0]
        # tick_lbls = [1, 2, 5, 10, 25, 50]
        tick_locs = [0,10,20,30,40,50]
        tick_lbls = [1,10,20,30,40,50]
        plt.yticks(tick_locs, tick_lbls)
        tick_locs = np.array([0,1,2,3,4,5])*10
        tick_lbls = [0,0.1,0.2,0.3,0.4,0.5]
        plt.xticks(tick_locs, tick_lbls)

        plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bicubic', cmap=cm.viridis)
        # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bilinear', cmap=cm.viridis)
        # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='nearest', cmap=cm.viridis)
        # plt.imshow(scaledA)

        # tick_locs = [2,2.8,3.6,4.4,5.2]
        # tick_lbls = [2, 3, 4, 5, 6]
        # plt.yticks(tick_locs, tick_lbls)
        # tick_locs = [0,1,2,3,4,5]
        # tick_lbls = [0,0.1,0.2,0.3,0.4,0.5]
        # plt.xticks(tick_locs, tick_lbls)
        plt.ylabel(r'$Number\;of\;Increments$',fontsize=18)
        plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=18)
        # plt.zlabel(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=18)
        plt.title(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=18)

        ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.8)
        cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis,
                           norm=mpl.colors.Normalize(vmin=-0, vmax=1))
        # cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis)
        cbar.set_clim(0, 1)
        plt.clim(0,1)

        plt.show()








