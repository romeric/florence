#!/usr/bin/env python
""" Parameteric studeis"""

import os, sys
sys.path.insert(1,'/home/roman/Dropbox/florence')


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

    

    Run = 0
    if Run:
        t_FEM = time.time()
        # nu = np.linspace(0.001,0.495,10)
        nu = np.linspace(0.001,0.495,6)
        # nu = np.linspace(0.01,0.495,2)
        E = np.array([1e05])
        p = [2,3,4,5,6]
        # p = [5,6]
        # p = [2]
         

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
                print 

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
        # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Almond3D_P_vs_Nu_'

        # print fname
        print repr(scaledA)
        print repr(condA)

        savemat(fpath+fname,Results)


        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        # np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/DONE', [t_FEM])

    if not Run:


        def plotter(which_formulation=0,projection_type=0, which_quality = 3, interpolate=False, save=False):
            """ 
                which_formulation           0 for linear
                                            1 for linearised
                                            2 for nonlinear

                projection_type             0 for arc length
                                            1 for orthogonal


                which_quality               1 F:F
                                            2 H:H
                                            3 J**2
            """


            if which_quality < 1:
                raise ValueError("which_quality can only be 1, 2 or 3")

            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib import rc


            # after review - change colormap
            colmap = cm.jet
            # colmap = cm.viridis

            ## for Palatino and other serif fonts use:
            rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
            rc('text', usetex=True)
            params = {'text.latex.preamble' : [r'\usepackage{amssymb}',r'\usepackage{mathtools}']}


            # ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/'

            # ProblemName = "Mech2D"
            ProblemName = "Almond3D"
            ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/'+ProblemName+"/"
            # SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/"+ProblemName+"/"
            SavePath = "/media/MATLAB/Paper_CompMech2015_After_Review/figures/"+ProblemName+"/"

            if which_formulation == 0:
                if projection_type == 0:
                    ResultsFile = ProblemName+'_P_vs_Nu_IncrementalLinearElastic_arc_length'
                elif projection_type == 1:
                    ResultsFile = ProblemName+'_P_vs_Nu_IncrementalLinearElastic_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')

            elif which_formulation == 1:
                if projection_type == 0:
                    ResultsFile = ProblemName+'_P_vs_Nu_IncrementallyLinearisedNeoHookean_2_arc_length'
                elif projection_type == 1:
                    ResultsFile = ProblemName+'_P_vs_Nu_IncrementallyLinearisedNeoHookean_2_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')

            elif which_formulation == 2:
                if projection_type == 0:
                    ResultsFile = ProblemName+'_P_vs_Nu_NeoHookean_2_arc_length'
                elif projection_type == 1:
                    ResultsFile = ProblemName+'_P_vs_Nu_NeoHookean_2_orthogonal'
                else:
                    raise ValueError('ProjectionType not understood')


            DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   
            
            if which_quality == 3:
                scaledA = DictOutput['ScaledJacobian']
            elif which_quality == 1:
                scaledA = DictOutput['ScaledFF']
                ResultsFile += "_Q1"
            elif which_quality == 2:
                scaledA = DictOutput['ScaledHH']
                ResultsFile += "_Q2"

            # condA = DictOutput['ConditionNumber']
            # nu = DictOutput['PoissonsRatios'][0]
            nu = np.linspace(0.001,0.5,100)*10
            p = np.array(DictOutput['PolynomialDegrees'][0])



            xmin = p[0]
            xmax = p[-1]
            ymin = nu[0]
            ymax = nu[-1]

            if scaledA.shape == (3,6):
                scaledA = np.concatenate((scaledA,np.zeros((2,6))+np.NAN),axis=0)
                xmax = 6

            # imshow is a direct matrix-to-pixel transformation
            # so flip the matrix upside down
            scaledA = scaledA[::-1,:]
            # condA = condA[::-1,:]

            #-----------------------------------
            # Manual interpolation
            if interpolate is True:
                from scipy.interpolate import interp1d
                cols = 20
                new_scaledA = np.zeros(((xmax-1)*cols,scaledA.shape[1]))+np.NAN
                for i in range(scaledA.shape[1]):
                    if which_formulation==2:
                        upto = np.min(np.where(np.isnan(np.sort(scaledA[:,i])))[0])
                    else:
                        upto = 6
                    if upto==0:
                        upto = 5
                    to_interpolate_y = scaledA[:,i][-upto:]
                    to_interpolate_x = p[:upto]
                    if to_interpolate_y.shape[0] > 1:
                        func = interp1d(to_interpolate_x,to_interpolate_y,kind='linear')
                        x_interp_data = np.linspace(2,to_interpolate_x[-1],cols*(to_interpolate_x.shape[0]))
                        y_interp_data = func(x_interp_data)
                        # print cols*(to_interpolate_x[0]-2),cols*(to_interpolate_x[-1]-1)
                        new_scaledA[cols*(to_interpolate_x[0]-2):cols*(to_interpolate_x[-1]-1),i] = y_interp_data[::-1]
                scaledA = new_scaledA[::-1,:]
            #-----------------------------------

            if ProblemName == "Almond3D":
                ninterp = 50
                if which_formulation != 1:
                    xp = np.linspace(0.001,0.5,6)
                else:
                    xp = np.linspace(0.001,0.5,10)
                cols = scaledA.shape[1]*ninterp
                new_scaledA = np.zeros((scaledA.shape[0],cols))
                for i in range(scaledA.shape[0]):
                    yp = scaledA[i,:]
                    new_scaledA[i,:] = np.interp(np.linspace(0.001,0.5,cols),xp,yp)
                scaledA = new_scaledA
            #-----------------------------------

            font_size = 22
            if ProblemName == "Almond3D" and which_formulation >22:
                plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bicubic', cmap=colmap)
                # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bilinear', cmap=colmap)
            else:
                # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bilinear', cmap=colmap)
                plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='nearest', cmap=colmap)

            if interpolate is True:
                tick_locs = [2,2.8,3.6,4.4,5.2]
                tick_lbls = [2, 3, 4, 5, 6]
                plt.yticks(tick_locs, tick_lbls)
            else:
                tick_locs = [2.4,3.2,4.0,4.8,5.6]
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
            # ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=1.0)
            # cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis,
                           # norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            # after review
            cbar = mpl.colorbar.ColorbarBase(ax, cmap=colmap,
                           norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            cbar.set_clim(0, 1)
            # Scale the image
            plt.clim(0,1)

            # fig = plt.gcf()
            # fig.set_size_inches(8, 7, forward=True)


            if save:
                # plt.savefig(SavePath+ResultsFile+'.eps',format='eps',dpi=1000) # high resolution
                # plt.savefig(SavePath+ResultsFile+'.eps',format='eps',dpi=300)
                plt.savefig(SavePath+ResultsFile+'.png',bbox_inches='tight',format='png',dpi=100,pad_inches=0.01)
                print SavePath+ResultsFile+'.png'

            plt.show()
            plt.close()


        plotter(2,0)
        # plotter(which_formulation=1,projection_type=0, save=True) 
        # Alomnd3D
        # plotter(which_formulation=0,projection_type=1, which_quality=3, save=False)
        # for i in range(3):
            # for j in range(1,4):
                # plotter(which_formulation=i,projection_type=1, which_quality=j, save=False) 










