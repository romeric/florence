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

    MainData.C = 2
    MainData.norder = 2 
    MainData.plot = (0,3)
    nrplot = (0,'last')
    MainData.write = 0




    Run = 0
    if Run:
        t_FEM = time.time()
        nu = np.linspace(0.001,0.495,100)
        # nu = np.linspace(0.01,0.495,2)
        E = 1e05
        E_A = 2.5*E
        G_A = E/2.
        AnalysisTypes = ["Linear","Nonlinear"]
        MaterialModels = ["IncrementalLinearElastic","TranservselyIsotropicLinearElastic","NeoHookean_2",
            "MooneyRivlin","NearlyIncompressibleMooneyRivlin","BonetTranservselyIsotropicHyperElastic"]

        # MaterialModels = ["IncrementalLinearElastic"]
        # MaterialModels = ["NearlyIncompressibleMooneyRivlin"]; AnalysisTypes = ["Nonlinear"]
        

        Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}
            # 'MeshPoints':None,'MeshElements':None,
            # 'MeshEdges':None, 'MeshFaces':None,'TotalDisplacement':None}

        condA=np.zeros((len(AnalysisTypes),len(MaterialModels),nu.shape[0]))
        scaledA = np.copy(condA)
        for k in range(len(AnalysisTypes)):
            MainData.AnalysisType = AnalysisTypes[k]
            for i in range(len(MaterialModels)):
                MainData.MaterialArgs.Type = MaterialModels[i]

                if (MaterialModels[i] == "IncrementalLinearElastic" or \
                    MaterialModels[i] == "TranservselyIsotropicLinearElastic") and \
                    AnalysisTypes[k] == "Nonlinear":
                    condA[k,i,j] = np.NAN
                    scaledA[k,i,j] = np.NAN
                    continue
                    # pass

                for j in range(nu.shape[0]):
                    MainData.MaterialArgs.nu = nu[j]
                    MainData.MaterialArgs.E = E
                    MainData.MaterialArgs.E_A = E_A
                    MainData.MaterialArgs.G_A = G_A

                    MainData.isScaledJacobianComputed = False
                    main(MainData,Results)  
                    CondExists = getattr(MainData.solve,'condA',None)
                    # ScaledExists = getattr(MainData.solve,'scaledA',None)
                    scaledA[k,i,j] = np.min(MainData.ScaledJacobian)
                    if CondExists is not None:
                        condA[k,i,j] = MainData.solve.condA
                    else:
                        condA[k,i,j] = np.NAN

        Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
        Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
        Results['MaterialModels'] = MaterialModels
        Results['AnalysisTypes'] = AnalysisTypes
        # print Results['ScaledJacobian']

        
        fname = "P"+str(MainData.C+1)+".mat"
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/Mech2D_MaterialFormulation_vs_Nu_'

        # print fpath+fname
        # savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/DONE_Materials', [t_FEM])

    if not Run:

        # MaterialModels = ["Isotropic Linear Elastic","Anisotropic Linear Elastic","Linearised NeoHookean",
            # "Linearised Mooney-Rivlin","Linearised Nearly Incompressible Material","Linearised Anisotropic Hyperelastic"]
        pp = 0
        mm = 5
        if mm==3:
            MaterialModels = [r"$Isotropic\; Linear\; Elastic$",r"$Linearised\; Mooney-Rivlin$",r"$Mooney-Rivlin$"]
        elif mm==2:
            MaterialModels = [r"$Isotropic\; Linear\; Elastic$",r"$Linearised\; NeoHookean$",r"$NeoHookean$"]
        elif mm==4:
            MaterialModels = [r"$Isotropic\; Linear\; Elastic$",r"$Linearised\;Nearly\;Incompressible Material$",r"$Nearly\;Incompressible Material$"]
        elif mm==5:
            MaterialModels = [r"$Transervsely\;Isotropic\; Linear\; Elastic$",r"$Linearised\; Transervsely\;Isotropic\;Hyperelastic$",
            r"$Transervsely\;Isotropic\;Hyperelastic$"]
            pp = 1

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


        # import h5py as hpy 
        ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/'
        # ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P2_orthogonal"
        ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P4"

        SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

        DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   
        
        scaledA = DictOutput['ScaledJacobian']
        condA = DictOutput['ConditionNumber']
        # nu = DictOutput['PoissonsRatios'][0]
        nu = np.linspace(0.001,0.5,100)*1
        p = DictOutput['PolynomialDegrees'][0]

        # plt.plot(nu,scaledA[0,:,:].T,'-o')
        # plt.plot(condA[0,:,:-5].T,'-o')

        nn = -5
        # nn = -1
        # func = scaledA
        func = condA

        font_size = 18
        plt.plot(nu[:nn],func[0,pp,:nn],linewidth=2)
        plt.plot(nu[:nn],func[0,mm,:nn],linewidth=2)
        plt.plot(nu[:nn],func[1,mm,:nn],linewidth=2)
        plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)
        # plt.ylabel(r"$Mesh\, Quality\,\, (Q_1)$",fontsize=font_size)
        plt.ylabel(r"$\kappa(A)$",fontsize=font_size)

        plt.legend(MaterialModels,loc='upper left',fontsize=font_size-2)
        # plt.legend(MaterialModels,loc='lower left',fontsize=font_size-2)
        # print np.mean(scaledA[0,5,:nn])
        # print np.std(scaledA[0,5,:nn],dtype=np.float64)
        # print np.min(scaledA[0,3,:nn]), np.max(scaledA[0,3,:nn])
        # print scaledA[0,3,:]
        # plt.show()

        # import itertools
        # marker = itertools.cycle(('o', 's', '+', '.', 'o', '*','x')) 

        # nn=10
        # for mm in range(6):
        #     zdata = np.polyfit(nu[:-nn],condA[0,mm,:-nn],5)
        #     poly = np.poly1d(zdata)
        #     xx=[]
        #     for i in nu[:-nn]:
        #         xx.append(poly(i))
        #     # print condA[0,0,:]
        #     # print xx

        #     plt.plot(nu[:-nn],xx,marker.next(),linestyle='-',linewidth=2)
        #     plt.legend(MaterialModels,loc='upper left')

        # # plt.plot(nu[:-nn],condA[0,mm,:-nn],'-ro')
        # plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=18)

        # if mm==3:
        #     plt.savefig(SavePath+ResultsFile+"_MooneyRivlin.eps",format='eps',dpi=1000)
        # elif mm==2:
        #     plt.savefig(SavePath+ResultsFile+"_Neo-Hookean.eps",format='eps',dpi=1000)
        # elif mm==4:
        #     plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial.eps",format='eps',dpi=1000)
        # elif mm==5:
        #     plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial.eps",format='eps',dpi=1000)


        # if mm==3:
        #     plt.savefig(SavePath+ResultsFile+"_MooneyRivlin_CondA.eps",format='eps',dpi=1000)
        # elif mm==2:
        #     plt.savefig(SavePath+ResultsFile+"_Neo-Hookean_CondA.eps",format='eps',dpi=1000)
        # elif mm==4:
        #     plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial_CondA.eps",format='eps',dpi=1000)
        # elif mm==5:
        #     plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial_CondA.eps",format='eps',dpi=1000)
        # print SavePath+ResultsFile+".mat"
        plt.show()








