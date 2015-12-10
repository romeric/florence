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


    Run = 1
    if Run:
        t_FEM = time.time()
        E = 1e05
        E_A = 2.5*E
        G_A = E/2.
        nu = 0.35

        # p = [2,3,4,5,6,7,8,9]
        p = [2,3,4,5,6,7]
        # p = [2,3]

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        mm = 2
        L2NormX = np.zeros((mm,len(p)),dtype=np.float64)
        L2Normx = np.zeros((mm,len(p)),dtype=np.float64)
        DoF = np.zeros((mm,len(p)),dtype=np.int64)
        NELEM = np.zeros((mm,len(p)),dtype=np.int64)
        
        for m in range(0,mm):

            if m==0:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "IncrementalLinearElastic"
            elif m==1:
                MainData.AnalysisType = "Linear"
                MainData.MaterialArgs.Type = "NeoHookean_2"
            # elif m==2:
            #     MainData.AnalysisType = "Linear"
            #     MainData.MaterialArgs.Type = "MooneyRivlin"
            # elif m==4:
            #     MainData.AnalysisType = "Linear"
            #     MainData.MaterialArgs.Type = "NearlyIncompressibleMooneyRivlin"
            # elif m==5:
            #     MainData.AnalysisType = "Linear"
            #     MainData.MaterialArgs.Type = "BonetTranservselyIsotropicHyperElastic"
            # elif m==6:
            #     MainData.AnalysisType = "Nonlinear"
            #     MainData.MaterialArgs.Type = "NeoHookean_2"
            # elif m==7:
            #     MainData.AnalysisType = "Nonlinear"
            #     MainData.MaterialArgs.Type = "MooneyRivlin"
            # elif m==8:
            #     MainData.AnalysisType = "Nonlinear"
            #     MainData.MaterialArgs.Type = "NearlyIncompressibleMooneyRivlin"
            # elif m==9:
            #     MainData.AnalysisType = "Nonlinear"
            #     MainData.MaterialArgs.Type = "BonetTranservselyIsotropicHyperElastic"

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
        pass