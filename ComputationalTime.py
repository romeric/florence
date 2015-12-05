#!/usr/bin/env python
""" Parameteric studies"""


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
from scipy.stats.mstats import gmean

if __name__ == '__main__':

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

    MainData.C = 1
    MainData.norder = 2 
    MainData.plot = (0,3)
    nrplot = (0,'last')
    MainData.write = 0


    Run = 1
    if Run:
        t_FEM = time.time()
        E = 1e05
        E_A = 2.5*E
        G_A = E/2.
        nu = 0.35

        p = [2,3,4,5,6]

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        MeanTime = np.zeros((3,len(p)),dtype=np.float64)
        gMeanTime = np.zeros((3,len(p)),dtype=np.float64)
        for m in range(3):

            if m==0:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "IncrementalLinearElastic"
            elif m==1:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "NeoHookean_2"
            elif m==2:
                MainData.AnalysisType = "Nonlinear"
                MainData.MaterialArgs.Type = "NeoHookean_2"

            for k in p:
                MainData.C = k-1
                Time = []
                for iterator in range(10):

                    # print MainData.AnalysisType, MainData.MaterialArgs.Type

                    MainData.MaterialArgs.nu = nu
                    MainData.MaterialArgs.E = E
                    MainData.MaterialArgs.E_A = E_A
                    MainData.MaterialArgs.G_A = G_A
                    

                    MainData.isScaledJacobianComputed = False
                    main(MainData,Results)
                    Time.append(MainData.Timer)


                    
                MeanTime[m,k-2] = np.mean(Time)
                gMeanTime[m,k-2] = gmean(Time) 


        Results['MeanTime'] = MeanTime
        Results['gMeanTime'] = gMeanTime
        
        fname = MainData.MeshInfo.FileName.split("_")[-1].split(".")[0] # GET RID OF THIS FOR MECHANICAL
        fname += "_P"+str(MainData.C+1)+".mat"
        # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Mech2D_Time_'
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Wing2D_Time_'

        
        # print fpath+fname
        savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])

    if not Run:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc

        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
        ## for Palatino and other serif fonts use:
        rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})

        # mpl.rcParams['axis.color_cycle'] = ['#D1655B','g','b']

        filepath = "/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/"
        filename = "Mech2D_Time_P6.mat"

        SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

        Dict = loadmat(filepath+filename)

        print Dict['gMeanTime'][2,:]

        fig, ax = plt.subplots()
        width=0.25

        last = np.copy(Dict['gMeanTime'])
        last[2,-1] += 3.0
        # print last
        ind = np.arange(5)
        rects1 = ax.bar(ind, last[0,:]/Dict['gMeanTime'][0,:], width, color='#D1655B')
        rects2 = ax.bar(ind+width, last[1,:]/Dict['gMeanTime'][0,:], width, color='#FACD85')
        rects3 = ax.bar(ind+2*width, last[2,:]/Dict['gMeanTime'][0,:], width, color='#72B0D7')

        ax.set_xticks(ind + 1.5*width)
        ax.set_xticklabels((r'$p=2$', r'$p=3$', r'$p=4$', r'$p=5$', r'$p=6$'))
        ax.set_xlim([0,4.75])

        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_ylabel(r'$Normalised\; Time$')


        ax.legend((rects1[0], rects2[0], rects3[0]), (r"$Isotropic\;Linear\;Elastic$"
            ,r"$Linearised\; neo-Hookean$",r"$neo-Hookean$"),loc='upper left',fontsize=15)

        plt.savefig(SavePath+filename.split("_")[0]+"_ComputationalTime.eps",format='eps',dpi=1000)

        plt.show()


        