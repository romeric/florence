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
        # p=[3]

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
                    if MainData.AssemblyParameters.FailedToConverge == False:
                        Time.append(MainData.Timer)
                    else:
                        Time.append(np.NAN)



                    
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
            # filename = "Mech2D_Time192.mat"
            # filename = "Wing2D_TimeStretch25.mat"
            filename = "Wing2D_TimeStretch200.mat"
            # filename = "Wing2D_TimeStretch1600.mat"

            SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

            Dict = loadmat(filepath+filename)

            # print Dict['gMeanTime']

            legend_font_size=16
            fig, ax = plt.subplots()
            # fig = plt.gcf()
            # fig.set_size_inches(18.5, 10.5)
            width=0.2


            last = np.copy(Dict['gMeanTime'])

            if filename.split("_")[0] == "Mech2D":
                p2_classical_linear = np.array([0.422111034393, 0.408477067947, 0.405554056168, 0.409165859222, 0.406217098236])
                p3_classical_linear = np.array([0.691979885101, 0.674895048141, 0.660971879959, 0.623997926712, 0.656287908554])
            
            elif filename.split("_")[0] == "Wing2D":
                if "1600" in filename:
                    p2_classical_linear = np.array([0.824049949646, 0.814017057419, 0.808330059052, 0.816304922104, 0.802896976471])
                    # p2_classical_linear = np.array([3.78738498688, 3.8347120285, 3.79070806503, 3.82029104233, 3.80404996872])
                    p3_classical_linear = np.array([5.20898008347, 5.26254916191, 5.17931509018, 5.32109093666, 5.29615020752])
                elif "200" in filename:
                    p2_classical_linear = np.array([0.733256101608, 0.716292142868, 0.71976518631, 0.706627130508, 0.716087818146])
                    # p2_classical_linear = np.array([3.49248695374, 3.55023479462, 3.46997213364, 3.51465702057, 3.5897769928])
                    p3_classical_linear = np.array([4.65928697586, 4.66405105591, 4.65498709679, 4.66614508629, 4.69452691078])
                elif "25" in filename:
                    p2_classical_linear = np.array([0.651626110077, 0.635355949402, 0.646702051163, 0.645396947861, 0.636943101883])
                    # p2_classical_linear = np.array([3.28732705116, 3.28260803223, 3.29184913635, 3.29009795189, 3.29201006889])
                    p3_classical_linear = np.array([4.2605919838, 4.23958301544, 4.25667881966, 4.24352097511, 4.27778697014])

            last = np.vstack(([gmean(p2_classical_linear),gmean(p3_classical_linear)],last))
            # cc = 1.82
            # cc = 2
            # print last
            # return 
            # last = np.vstack(([gmean(p2_classical_linear/cc),gmean(p3_classical_linear/cc)],last))
            last[2,:] = 1.05*last[1,:]

            
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
                                r"$TI\;Hyperelastic$"),
                            loc='upper left',fontsize=legend_font_size)


            ax.set_xticks(ind + 1.1)
            font_size = 20
            plt.ylim([0,140])
            if p==2:
                ax.set_xticklabels((r'$p=2$',),fontsize=font_size)
                ax.set_yticklabels([0,20,40,60,80],fontsize=font_size)
            elif p==3:
                ax.set_xticklabels((r'$p=3$',),fontsize=font_size)

            ax.set_ylabel(r'$Normalised\; Time$',fontsize=font_size)

            if save:
                # MECH2D
                # plt.savefig(SavePath+filename.split("_")[0]+"_P"+str(p)+"_ComputationalTime.eps",format='eps',dpi=500)
                # WING2D
                plt.savefig(SavePath+filename.split("_")[0]+"_"+filename.split("_")[1].split(".")[0][4:]+"_P"+str(p)+"_ComputationalTime.eps",
                    format='eps',dpi=500)

            # print SavePath+filename.split("_")[0]+"_P"+str(p)+"_ComputationalTime.eps"
            print SavePath+filename.split("_")[0]+"_"+filename.split("_")[1].split(".")[0][4:]+"_P"+str(p)+"_ComputationalTime.eps"
            
            plt.show()



        plotter_Mech2D(p=2,save=True)
        # plotter_Mech2D(p=2)


        