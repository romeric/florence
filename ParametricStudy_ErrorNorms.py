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
        p = [2,3,4,5,6,7]
        # p = [2,3]

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        mm = 10
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

        savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        # np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])

    if not Run:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.mlab import griddata

        marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
        linestyle = itertools.cycle(('-','-.','--',':'))

        rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})

        

        # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ComputationalTime/Mech2D_Time_'
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/ErrorNorms/Wing2D_Errors_'
        fname = "Stretch25.mat"
        # fname = "Stretch1600.mat"

        print fpath+fname

        Results = loadmat(fpath+fname)

        DoF = Results['DoF']
        L2Normx = Results['L2Normx']
        L2NormX = Results['L2NormX']

        print DoF.shape, L2Normx.shape
        # plt.loglog(DoF[0,:],L2NormX[0,:],'-bs')
        # plt.loglog(DoF[0,:],L2Normx[0,:],'-ro')
        # L2NormX = np.concatenate((np.zeros((1,2)),L2NormX),axis=1)
        # print L2NormX

        # plt.plot(np.sqrt(DoF[0,:]),np.log10(L2NormX[0,:]),'-bs')
        # plt.plot(np.sqrt(DoF[0,:]),np.log10(L2Normx[0,:]),'-.ro')

        for i in range(10):
            plt.plot(np.sqrt(DoF[0,:]),np.log10(L2Normx[i,:]),marker.next(),linestyle=linestyle.next(),linewidth=2)
        # plt.plot(np.log10(np.sqrt(DoF[0,:])),np.log10(L2Normx[0,:]),'-ro')
        # plt.loglog(DoF[0,:],L2Normx[0,:],'-ro')
        # plt.legend([r'$Undeformed\; mesh$',r'$Deformed\; mesh\;using\;Linear\; Elasticity$'],loc='best',fontsize=16)
        plt.legend([r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
                                r"$IL\;Mooney-Rivlin$",r"$IL\;Nearly\;Incompressible$",r"$ILTI\;Hyperelastic$",
                                r"$neo-Hookean$",r"$Mooney-Rivlin$",r"$Nearly\;Incompressible$",
                                r"$TI\;Hyperelastic$"],loc='best',fontsize=16)
        plt.ylabel(r'log$_{10}(L^2\;error)$')
        plt.xlabel(r'$\sqrt{ndof}$')
        plt.grid('on')


        sname = fpath+fname
        sname = sname.split(".")[0]+".eps"
        print sname
        # plt.savefig(fpath+fname,format='png',dpi=100)
        # plt.savefig("/home/roman/Dropbox/Wing2D_Errors_Stretch25.eps",format='eps',dpi=100)

        plt.show()