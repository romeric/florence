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
np.set_printoptions(linewidth=400)

# IMPORT NECESSARY CLASSES FROM BASE
from Base import Base as MainData

from Main.FiniteElements.MainFEM import main
from scipy.io import savemat, loadmat



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
        nu = np.linspace(0.001,0.495,10)
        # nu = np.linspace(0.01,0.495,2)
        E = np.array([1e05])
        p = [2,3,4,5,6]
        p = [2]
         

        Results = {'PolynomialDegrees':p,'PoissonsRatios':nu,'Youngs_Modulus':E}
            # 'MeshPoints':None,'MeshElements':None,
            # 'MeshEdges':None, 'MeshFaces':None,'TotalDisplacement':None}

        condA=np.zeros((len(p),nu.shape[0]))
        scaledA = np.copy(condA)
        scaledAFF = np.copy(condA)
        scaledAHH = np.copy(condA)
        scaledAFNFN = np.copy(condA)
        scaledACNCN = np.copy(condA)
        for i in range(len(p)):
            MainData.C = p[i]-1
            for j in range(nu.shape[0]):
                MainData.MaterialArgs.nu = nu[j]
                MainData.MaterialArgs.E = E[0]
                MainData.isScaledJacobianComputed = False
                main(MainData,Results)  
                CondExists = getattr(MainData.solve,'condA',None)
                # ScaledExists = getattr(MainData.solve,'scaledA',None)
                scaledA[i,j] = np.min(MainData.ScaledJacobian)
                scaledAFF[i,j] = np.min(MainData.ScaledFF)
                scaledAHH[i,j] = np.min(MainData.ScaledHH)
                # scaledAFNFN[i,j] = np.min(MainData.ScaledFNFN)
                # scaledACNCN[i,j] = np.min(MainData.ScaledCNCN)
                if CondExists is not None:
                    condA[i,j] = MainData.solve.condA
                else:
                    condA[i,j] = np.NAN

        Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
        Results['ScaledFF'] = scaledAFF # one given row contains all values of nu for a fixed p
        Results['ScaledHH'] = scaledAHH # one given row contains all values of nu for a fixed p
        # Results['ScaledFNFN'] = scaledAFNFN # one given row contains all values of nu for a fixed p
        # Results['ScaledCNCN'] = scaledACNCN # one given row contains all values of nu for a fixed p
        Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
        Results['MaterialModel'] = MainData.MaterialArgs.Type
        # print Results['ScaledJacobian']

        fname = MainData.MaterialArgs.Type
        if MainData.MaterialArgs.Type != "IncrementalLinearElastic" and MainData.MaterialArgs.Type != "LinearModel":
            if MainData.AnalysisType == "Linear":
                fname = "IncrementallyLinearised"+MainData.MaterialArgs.Type
        
        fname = fname + "_" + MainData.BoundaryData.ProjectionType+".mat"
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'

        # print fname
        print repr(scaledA)
        print repr(condA)
        exit()
        # exit(0)
        # savemat(fpath+fname,Results)


        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        # np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/DONE', [t_FEM])

    if not Run:


        def plotter(which_formulation=0,projection_type=0, save=False):
            """ 
                which_formulation           0 for linear
                                            1 for linearised
                                            2 for nonlinear

                projection_type             0 for arc length
                                            1 for orthogonal
            """

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib import rc

            ## for Palatino and other serif fonts use:
            rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
            rc('text', usetex=True)
            params = {'text.latex.preamble' : [r'\usepackage{amssymb}',r'\usepackage{mathtools}']}


            # ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/'
            ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D/'
            SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

            if which_formulation == 0:
                if projection_type == 0:
                    ResultsFile = 'Mech2D_P_vs_Nu_IncrementalLinearElastic_arc_length'
                elif projection_type == 1:
                    ResultsFile = 'Mech2D_P_vs_Nu_IncrementalLinearElastic_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')

            elif which_formulation == 1:
                if projection_type == 0:
                    ResultsFile = 'Mech2D_P_vs_Nu_IncrementallyLinearisedNeoHookean_2_arc_length'
                elif projection_type == 1:
                    ResultsFile = 'Mech2D_P_vs_Nu_IncrementallyLinearisedNeoHookean_2_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')

            elif which_formulation == 2:
                if projection_type == 0:
                    ResultsFile = 'Mech2D_P_vs_Nu_NeoHookean_2_arc_length'
                elif projection_type == 1:
                    ResultsFile = 'Mech2D_P_vs_Nu_NeoHookean_2_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')


            DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   
            
            scaledA = DictOutput['ScaledJacobian']
            condA = DictOutput['ConditionNumber']
            # nu = DictOutput['PoissonsRatios'][0]
            nu = np.linspace(0.001,0.5,100)*10
            p = DictOutput['PolynomialDegrees'][0]



            xmin = p[0]
            xmax = p[-1]
            ymin = nu[0]
            ymax = nu[-1]



            # imshow is a direct matrix-to-pixel transformation
            # so flip the matrix upside down
            scaledA = scaledA[::-1,:]
            # print scaledA
            # exit()
            condA = condA[::-1,:]

            font_size = 22
            plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bicubic', cmap=cm.viridis)

            tick_locs = [2,2.8,3.6,4.4,5.2]
            tick_lbls = [2, 3, 4, 5, 6]
            plt.yticks(tick_locs, tick_lbls)
            tick_locs = [0,1,2,3,4,5]
            tick_lbls = [0,0.1,0.2,0.3,0.4,0.5]
            plt.xticks(tick_locs, tick_lbls)
            plt.ylabel(r'$Polynomial\, Degree\,\, (p)$',fontsize=font_size)
            plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)
            # plt.title(r"$Mesh\, Quality\,-\, min(Q_3)$",fontsize=18) # DONT PUT TITLE

            # USE COLORBAR ONLY FOR THE NONLINEAR - LAST COLUMN IN THE PAPER
            # if which_formulation == 2:
            ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.8)
            cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis,
                           norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            cbar.set_clim(0, 1)
            # Scale the image
            plt.clim(0,1)

            # fig = plt.gcf()
            # fig.set_size_inches(8, 7, forward=True)

            if save:
                # plt.savefig(SavePath+ResultsFile+'.eps',format='eps',dpi=1000) # high resolution
                # plt.savefig(SavePath+ResultsFile+'.eps',format='eps',dpi=300)
                # plt.savefig('/home/roman/Dropbox/dddd.svg', format='svg',dpi=1200)
                plt.savefig(SavePath+ResultsFile+'.png',bbox='tight',format='png',dpi=100)

                # plt.savefig('/home/roman/Dropbox/dddd.eps', format='eps',dpi=100)

            plt.show()


        # plotter(2,1,True)
        plotter(which_formulation=0,projection_type=0, save=False) 










