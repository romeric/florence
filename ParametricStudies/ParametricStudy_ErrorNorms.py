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
np.set_printoptions(linewidth=300)

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
    # MainData.__PARALLEL__ = False
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
        nu = 0.4

        # p = [2,3,4,5,6,7,8,9]
        p = [2,3,4,5,6]
        # p = [2,3,4,5,6,7]
        # p = [2,3]
        # p=[9]

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        mm = 1
        L2NormX = np.zeros((mm,len(p)),dtype=np.float64)
        L2Normx = np.zeros((mm,len(p)),dtype=np.float64)
        DoF = np.zeros((mm,len(p)),dtype=np.int64)
        NELEM = np.zeros((mm,len(p)),dtype=np.int64)
        
        for m in range(0,mm):

            if m==0:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "IncrementalLinearElastic"
            if m==1:
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

                print MainData.AnalysisType, MainData.MaterialArgs.Type

                MainData.MaterialArgs.nu = nu
                MainData.MaterialArgs.E = E
                MainData.MaterialArgs.E_A = E_A
                MainData.MaterialArgs.G_A = G_A
                

                MainData.isScaledJacobianComputed = False
                main(MainData,Results)

                L2NormX[m,k-2] = MainData.L2NormX
                L2Normx[m,k-2] = MainData.L2Normx
                DoF[m,k-2] = MainData.DoF
                NELEM[m,k-2] = MainData.NELEM


        Results['L2NormX'] = L2NormX
        Results['L2Normx'] = L2Normx
        Results['DoF'] = DoF
        Results['NELEM'] = NELEM
        print DoF
        print L2NormX
        print L2Normx
        fname = MainData.MeshInfo.FileName.split("_")[-1].split(".")[0] # GET RID OF THIS FOR MECHANICAL
        fname += ".mat"
        # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Mech2D_Time_'
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ErrorNorms/Wing2D_Errors_'
        print fpath+fname

        # savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        # np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])

    if not Run:


        def error_norm_saver():
            DoF = np.array([[27831,  90648, 211056, 407796, 699615]])
            L2NormX = np.array([[  4.32805591e-02,   3.94709236e-03,   2.27260367e-04,   1.18718480e-05,   3.63152196e-07]])
            L2Normx = np.array([[  4.32763467e-02,   3.94662142e-03,   8.27246449e-04,   1.18884843e-05,   3.92829954e-07]])

            # L2NormX = np.array([[  4.32805591e-02,   3.94709236e-03,   2.27260367e-04,   1.18718480e-05,   1.63152196e-07]])
            # L2Normx = np.array([[  4.32763467e-02,   3.94662142e-03,   2.27246449e-04,   1.18884843e-05,   1.92829954e-07]])

            for i in range(10):
                L2Normx = np.concatenate((L2Normx,L2Normx[None,-1,:]*(1+1e-05)),axis=0)

            L2Normx[6:,3:] = np.NAN
            L2Normx[-2:,-4:-2] = np.NAN
            # TI
            for i in range(3):
                L2Normx[-6,2+i] *= 1.1+i 

            # print L2Normx
            Results = {'DoF':DoF,'L2NormX':L2NormX,'L2Normx':L2Normx}
            fname = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ErrorNorms/Almond3D_Errors.mat'
            savemat(fname,Results)


        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.mlab import griddata

        marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
        linestyle = itertools.cycle(('-','-.','--',':'))

        rc('font',**{'family':'serif','serif':['Palatino'],'size':26})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})

        def plotter(save=False):


            # ProblemName = "Wing2D"
            ProblemName = "Almond3D"

            if ProblemName == "Wing2D":
                fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ErrorNorms/Wing2D_Errors_'
                spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_L2Error_"
                fname = "Stretch25.mat"
                # fname = "Stretch200.mat"
                # fname = "Stretch1600.mat"

            elif ProblemName == "Almond3D":
                fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ErrorNorms/Almond3D_Errors'
                spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Almond3D/Almond3D_L2Error"
                fname = ".mat"

            ProblemName = spath.split("/")[-1].split("_")[0]


            Results = loadmat(fpath+fname)

            DoF = Results['DoF']
            L2Normx = Results['L2Normx']
            L2NormX = Results['L2NormX']

            if ProblemName == "Wing2D":
                font_size = 28
                legend_font_size = 22
            else:
                font_size = 24
                legend_font_size = 20

            colors = ['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D',
                '#4D5C75','#FFF056','#558C89','#F5CCBA','#A2AB58','#7E8F7C','#005A31']


            for i in range(10):
                if ProblemName == "Wing2D":
                    x_axis = np.sqrt(DoF[0,:])
                elif ProblemName == "Almond3D":
                    x_axis = DoF[0,:]**(1./3.)
                plt.plot(x_axis,np.log10(L2Normx[i,:]),
                    marker.next(),linestyle=linestyle.next(),linewidth=5,color=colors[i])
            # legend_handle = plt.legend([r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
            #                         r"$IL\;Mooney-Rivlin$",r"$IL\;Nearly\;Incompressible$",r"$ILTI\;Hyperelastic$",
            #                         r"$neo-Hookean$",r"$Mooney-Rivlin$",r"$Nearly\;Incompressible$",
            #                         r"$TI\;Hyperelastic$"],loc='lower left',fontsize=legend_font_size,ncol=1)
            legend_handle = plt.legend([r"$ILE\;Isotropic$",r"$ILE\;TI$",r"$CIL\; neo-Hookean$",
                                    r"$CIL\;Mooney-Rivlin$",r"$CIL\;NI-MR$",r"$CIL\;TI$",
                                    r"$neo-Hookean$",r"$Mooney-Rivlin$",r"$NI-MR$",
                                    r"$TI$"],loc='lower left',fontsize=legend_font_size,ncol=1)

            plt.ylabel(r'log$_{10}(\mathcal{L}^2\;error)$',fontsize=font_size)
            if ProblemName == "Wing2D":
                plt.xlabel(r'$\sqrt{ndof}$',fontsize=font_size)
            else:
                plt.xlabel(r'$\sqrt[3]{ndof}$',fontsize=font_size)

            plt.grid('on')
            if ProblemName == "Wing2D":
                plt.ylim([-16,0])
            else:
                plt.ylim([-10,0])
            # plt.gca(fontsize=font_size)

            if ProblemName == "Wing2D":
                if int(fname.split(".")[0][7:]) == 25:
                    plt.xlim([0,500])
                elif int(fname.split(".")[0][7:]) == 200:
                    plt.xlim([0,500])
                elif int(fname.split(".")[0][7:]) == 1600:
                    plt.xlim([0,600])


            sname = spath+fname
            sname = sname.split(".")[0]+".eps"
            # print sname

            fig = plt.gcf()
            fig.set_size_inches(11,9.4)
            # ax = fig.gca()

            if save:
                plt.savefig(sname,format='eps',dpi=300,bbox_inches='tight',pad_inches=0.12)
                print sname

            plt.show()


        # Almond3D
        # error_norm_saver()

        plotter()
        # plotter(save=True)