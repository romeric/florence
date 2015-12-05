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

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools


        def plotter(degree = 2, which_func=0, save = False):
            """
                degree          2,3 or 4 every degree is a mat file for p

                which_func = 0 for scaledA
                which_func = 1 for condA

                save            to save or not

            """


            # marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
            marker = itertools.cycle(('o', 's', 'x'))

            # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
            ## for Palatino and other serif fonts use:
            rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
            rc('text', usetex=True)

            # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
            rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
            # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})


            pp = 0
            for mm in range(2,6):
                if mm==3:
                    MaterialModels = [r"$Isotropic\;Linear\;Elastic$",r"$Linearised\; Mooney-Rivlin$",r"$Mooney-Rivlin$"]
                elif mm==2:
                    MaterialModels = [r"$Isotropic\;Linear\;Elastic$",r"$Linearised\; neo-Hookean$",r"$neo-Hookean$"]
                elif mm==4:
                    MaterialModels = [r"$Isotropic\;Linear\;Elastic$",r"$Linearised\;Nearly\;Incompressible\;Material$",r"$Nearly\;Incompressible\;Material$"]
                elif mm==5:
                    MaterialModels = [r"$Transervsely\;Isotropic\; Linear\; Elastic$",r"$Linearised\; Transervsely\;Isotropic\;Hyperelastic$",
                    r"$Transervsely\;Isotropic\;Hyperelastic$"]
                    pp = 1

            

                ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/'
                # ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P2_orthogonal"
                ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P"+str(degree)
                SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

                DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   
                
                scaledA = DictOutput['ScaledJacobian']
                condA = DictOutput['ConditionNumber']
                # nu = DictOutput['PoissonsRatios'][0]
                nu = np.linspace(0.001,0.5,100)*1
                p = DictOutput['PolynomialDegrees'][0]

                font_size = 18

                plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)
                

                # print np.mean(scaledA[0,5,:nn])
                # print np.std(scaledA[0,5,:nn],dtype=np.float64)
                # print np.min(scaledA[0,3,:nn]), np.max(scaledA[0,3,:nn])
                # print scaledA[0,3,:]

                if which_func == 1:
                    func = scaledA
                    nn=-1
                    plt.plot(nu[:nn],func[0,pp,:nn],marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:nn],func[0,mm,:nn],marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:nn],func[1,mm,:nn],marker.next(),linestyle='-',linewidth=2)


                    plt.ylabel(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=font_size)


                    plt.legend(MaterialModels,loc='lower left',fontsize=font_size-2)
                    if degree == 2 and mm == 5:
                        plt.legend(MaterialModels,loc='upper left',fontsize=font_size-2)
                    if degree > 2:
                        plt.legend(MaterialModels,loc='upper left',fontsize=font_size-2)
                        if degree == 4 and mm==5:
                            plt.legend(MaterialModels,loc='lower left',fontsize=font_size-2)



                    if save:
                        if mm==3:
                            plt.savefig(SavePath+ResultsFile+"_MooneyRivlin.eps",format='eps',dpi=1000)
                        elif mm==2:
                            plt.savefig(SavePath+ResultsFile+"_Neo-Hookean.eps",format='eps',dpi=1000)
                        elif mm==4:
                            plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial.eps",format='eps',dpi=1000)
                        elif mm==5:
                            plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial.eps",format='eps',dpi=1000)


                
                


                if which_func == 0:

                    func = condA

                    nn=1
                    pdegree = 8
                    # for mm in range(6):
                    zdata = np.polyfit(nu[:-nn],condA[0,pp,:-nn],pdegree)
                    poly = np.poly1d(zdata)
                    x1=[]
                    for i in nu[:-nn]:
                        x1.append(poly(i))

                    zdata = np.polyfit(nu[:-nn],condA[0,mm,:-nn],pdegree)
                    poly = np.poly1d(zdata)
                    x2=[]
                    for i in nu[:-nn]:
                        x2.append(poly(i))

                    zdata = np.polyfit(nu[:-nn],condA[1,mm,:-nn],pdegree)
                    poly = np.poly1d(zdata)
                    x3=[]
                    for i in nu[:-nn]:
                        x3.append(poly(i))

                    plt.plot(nu[:-nn],x1,marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:-nn],x2,marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:-nn],x3,marker.next(),linestyle='-',linewidth=2)

                    plt.legend(MaterialModels,loc='upper left',fontsize=font_size-2)
                    plt.ylabel(r"$\kappa(A)$",fontsize=font_size)

                    y_formatter = mpl.ticker.ScalarFormatter(useOffset=True)
                    y_formatter.set_powerlimits((-4,4))
                    ax = plt.gca()
                    ax.yaxis.set_major_formatter(y_formatter)

                    if save:
                        if mm==3:
                            plt.savefig(SavePath+ResultsFile+"_MooneyRivlin_CondA.eps",format='eps',dpi=1000)
                        elif mm==2:
                            plt.savefig(SavePath+ResultsFile+"_Neo-Hookean_CondA.eps",format='eps',dpi=1000)
                        elif mm==4:
                            plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial_CondA.eps",format='eps',dpi=1000)
                        elif mm==5:
                            plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial_CondA.eps",format='eps',dpi=1000)



                plt.show()


        plotter(degree=4,which_func=1,save=True)
        # plotter(which_func=1)








