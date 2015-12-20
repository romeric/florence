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
        nu = 0.4

        p = [2,3,4,5,6]
        # p = [2,3]
        # p=[2]

        Results = {'PolynomialDegrees':p,'PoissonsRatio':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

        mm=2
        for m in range(0,mm):

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

                print MainData.AnalysisType, MainData.MaterialArgs.Type

                MainData.MaterialArgs.nu = nu
                MainData.MaterialArgs.E = E
                MainData.MaterialArgs.E_A = E_A
                MainData.MaterialArgs.G_A = G_A
                
                MainData.isScaledJacobianComputed = False
                main(MainData,Results)


        spath = "/home/roman/Dropbox/2015_HighOrderMeshing/Paper_CompMech2015_CurvedMeshFiles"
        sname = "/Mech2D.mat"
        # For Wing2D
        # sname = "/Wing2D_"
        # sname += MainData.MeshInfo.FileName.split(".")[0].split("_")[-1]+".mat"

        print spath+sname

        savemat(spath+sname,Results)