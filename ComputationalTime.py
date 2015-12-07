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


    Run = 0
    if Run:
        t_FEM = time.time()
        E = 1e05
        E_A = 2.5*E
        G_A = E/2.
        nu = 0.35

        # p = [2,3,4,5,6]
        p = [2,3]

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        # MeanTime = np.zeros((3,len(p)),dtype=np.float64)
        # gMeanTime = np.zeros((3,len(p)),dtype=np.float64)

        MeanTime = np.zeros((10,len(p)),dtype=np.float64)
        gMeanTime = np.zeros((10,len(p)),dtype=np.float64)
        
        for m in range(0,10):

            if m==0:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "IncrementalLinearElastic"
            elif m==1:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "TranservselyIsotropicLinearElastic"
            elif m==2:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "NeoHookean_2"
            elif m==3:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "MooneyRivlin"
            elif m==4:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "NearlyIncompressibleMooneyRivlin"
            elif m==5:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "BonetTranservselyIsotropicHyperElastic"
            elif m==6:
                MainData.AnalysisType = "Nonlinear"
                MainData.MaterialArgs.Type = "NeoHookean_2"
            elif m==7:
                MainData.AnalysisType = "Nonlinear"
                MainData.MaterialArgs.Type = "MooneyRivlin"
            elif m==8:
                MainData.AnalysisType = "Nonlinear"
                MainData.MaterialArgs.Type = "NearlyIncompressibleMooneyRivlin"
            elif m==9:
                MainData.AnalysisType = "Nonlinear"
                MainData.MaterialArgs.Type = "BonetTranservselyIsotropicHyperElastic"

            for k in p:
                MainData.C = k-1
                Time = []
                for iterator in range(5):

                    print MainData.AnalysisType, MainData.MaterialArgs.Type

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
        fname += ".mat"
        # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Mech2D_Time_'
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Wing2D_Time'
        print fpath+fname

        
        # print fpath+fname
        savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])

    if not Run:

        def plotter_Mech2D(p=2,save=False):


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
            colors = ['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D',
                '#4D5C75','#FFF056','#558C89','#F5CCBA','#A2AB58','#7E8F7C','#005A31']

            filepath = "/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/"
            filename = "Mech2D_Time192.mat"

            SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

            Dict = loadmat(filepath+filename)

            # print Dict['gMeanTime']

            legend_font_size=13
            fig, ax = plt.subplots()
            width=0.2


            last = np.copy(Dict['gMeanTime'])
            p2_classical_linear = np.array([0.422111034393, 0.408477067947, 0.405554056168, 0.409165859222, 0.406217098236])
            p3_classical_linear = np.array([0.691979885101, 0.674895048141, 0.660971879959, 0.623997926712, 0.656287908554])

            last = np.vstack(([gmean(p2_classical_linear),gmean(p3_classical_linear)],last))

            rects = [None]*11
            ind = np.arange(1)
            for i in range(11):
                if p==2:
                    rects[i] = ax.bar(ind+i*width, last[i,0]/last[0,0], width, color=colors[i])
                elif p==3:
                    rects[i] = ax.bar(ind+i*width, last[i,1]/last[0,1], width, color=colors[i])

            plt.xlim([0,2.21])

            ax.legend((rects[0][0], rects[1][0], rects[2][0], rects[3][0], rects[4][0], rects[5][0],
                            rects[6][0], rects[7][0], rects[8][0], rects[9][0],rects[10][0]), 
                            (r"$Linear\;Elastic$",r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
                                r"$IL\;Mooney-Rivlin$",r"$IL\;Nearly\;Incompressible$",r"$ILTI\;Hyperelastic$",
                                r"$neo-Hookean$",r"$Mooney-Rivlin$",r"$Nearly\;Incompressible$",
                                r"$Transervsely\;Isotropic\;Hyperelastic$"),
                            loc='upper left',fontsize=legend_font_size)


            ax.set_xticks(ind + 1.1)
            if p==2:
                ax.set_xticklabels((r'$p=2$',))
            elif p==3:
                ax.set_xticklabels((r'$p=3$',))

            ax.set_ylabel(r'$Normalised\; Time$')

            if save:
                plt.savefig(SavePath+filename.split("_")[0]+"_P"+str(p)+"_ComputationalTime.eps",format='eps',dpi=500)
            print SavePath+filename.split("_")[0]+"_P"+str(p)+"_ComputationalTime.eps"

            plt.show()

        plotter_Mech2D(p=3,save=True)
        # plotter_Mech2D()


        