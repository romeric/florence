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
    MainData.__PARALLEL__ = False
    # nCPU = 8
    __MEMORY__ = 'SHARED'
    # __MEMORY__ = 'DISTRIBUTED'

    MainData.C = 5
    MainData.norder = 2 
    MainData.plot = (0,3)
    nrplot = (0,'last')
    MainData.write = 0




    Run = 0
    if Run:
        t_FEM = time.time()
        # nu = np.linspace(0.001,0.495,100)
        nu = np.linspace(0.001,0.495,20)
        # nu = np.linspace(0.001,0.495,10)
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
        whole_scaledA = np.zeros((len(AnalysisTypes),len(MaterialModels),nu.shape[0],192))
        for k in range(len(AnalysisTypes)):
            MainData.AnalysisType = AnalysisTypes[k]
            for i in range(len(MaterialModels)):
                MainData.MaterialArgs.Type = MaterialModels[i]

                print MainData.AnalysisType, MainData.MaterialArgs.Type

                if (MaterialModels[i] == "IncrementalLinearElastic" or \
                    MaterialModels[i] == "TranservselyIsotropicLinearElastic") and \
                    AnalysisTypes[k] == "Nonlinear":
                    condA[k,i,j] = np.NAN
                    scaledA[k,i,j] = np.NAN

                else:

                    for j in range(nu.shape[0]):
                        MainData.MaterialArgs.nu = nu[j]
                        MainData.MaterialArgs.E = E
                        MainData.MaterialArgs.E_A = E_A
                        MainData.MaterialArgs.G_A = G_A

                        MainData.isScaledJacobianComputed = False
                        main(MainData,Results)  
                        print
                        CondExists = getattr(MainData.solve,'condA',None)
                        # ScaledExists = getattr(MainData.solve,'scaledA',None)
                        scaledA[k,i,j] = np.min(MainData.ScaledJacobian)
                        whole_scaledA[k,i,j,:MainData.ScaledJacobian.shape[0]] = MainData.ScaledJacobian
                        if CondExists is not None:
                            condA[k,i,j] = MainData.solve.condA
                        else:
                            condA[k,i,j] = np.NAN

        Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
        Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
        Results['MaterialModels'] = MaterialModels
        Results['AnalysisTypes'] = AnalysisTypes
        Results['WholeScaledJacobian'] = whole_scaledA
        # print Results['ScaledJacobian']

        
        # fname = "P"+str(MainData.C+1)+".mat"
        fname = "P"+str(MainData.C+1)+"_V2.mat"
        fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/Mech2D_MaterialFormulation_vs_Nu_'

        # print fpath+fname
        savemat(fpath+fname,Results)

        t_FEM = time.time()-t_FEM
        print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
        np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/DONE_Materials', [t_FEM])

    if not Run:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools


        

        # rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        # rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
        ## for Palatino and other serif fonts use:
        rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])

        # rc('axes',color_cycle=['#D1655B','#44AA66','#7E8F7C','#FACD85','#70B9B0','#72B0D7',
        #     '#E79C5D','#4D5C75','#FFF056','#558C89','#F5CCBA','#A2AB58','#005A31'])

        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})


        def plotter(degree = 2, which_func=0, save = False):
            """
                degree          2,3 or 4 every degree is a mat file for p

                which_func = 0 for scaledA
                which_func = 1 for condA

                save            to save or not

            """

            marker = itertools.cycle(('o', 's', 'x'))


            pp = 0
            for mm in range(2,6):
                # if mm==3:
                #     MaterialModels = [r"$Incremental\; Isotropic\;Linear\;Elastic$",
                #         r"$Incrementally\;Linearised\; Mooney-Rivlin$",r"$Mooney-Rivlin$"]
                # elif mm==2:
                #     MaterialModels = [r"$Incremental\; Isotropic\;Linear\;Elastic$",
                #         r"$Incrementally\;Linearised\; neo-Hookean$",r"$neo-Hookean$"]
                # elif mm==4:
                #     MaterialModels = [r"$Incremental\;Isotropic\;Linear\;Elastic$",
                #         r"$Incrementally\;Linearised\;Nearly\;Incompressible\;Material$",
                #         r"$Nearly\;Incompressible\;Material$"]
                # elif mm==5:
                #     MaterialModels = [r"$Incremental\;Transervsely\;Isotropic\; Linear\; Elastic$",
                #         r"$Incrementally\;Linearised\; Transervsely\;Isotropic\;Hyperelastic$",
                #         r"$Transervsely\;Isotropic\;Hyperelastic$"]
                #     pp = 1
                if mm==3:
                    MaterialModels = [r"$II\;Linear\;Elastic$",
                        r"$IL\; Mooney-Rivlin$",r"$Mooney-Rivlin$"]
                elif mm==2:
                    MaterialModels = [r"$II\;Linear\;Elastic$",
                        r"$IL\; neo-Hookean$",r"$neo-Hookean$"]
                elif mm==4:
                    MaterialModels = [r"$II\;Linear\;Elastic$",
                        r"$IL\;Nearly\;Incompressible$",
                        r"$Nearly\;Incompressible$"]
                elif mm==5:
                    MaterialModels = [r"$ITI\; Linear\; Elastic$",
                        r"$ILTI\;Hyperelastic$",
                        r"$TI\;Hyperelastic$"]
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

                font_size = 22
                legend_font_size = 22

                # fig = plt.gcf()
                # fig.set_size_inches(8,9)

                plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)
                

                # print np.mean(scaledA[0,5,:nn])
                # print np.std(scaledA[0,5,:nn],dtype=np.float64)
                # print np.min(scaledA[0,3,:nn]), np.max(scaledA[0,3,:nn])
                # print scaledA[0,3,:]

                if degree > 4:
                    new_scaledA = np.zeros((scaledA.shape[0],scaledA.shape[1],nu.shape[0]))
                    new_condA = np.zeros((scaledA.shape[0],scaledA.shape[1],nu.shape[0]))
                    for i in range(2):
                        for j in range(6):
                            new_scaledA[i,j,:] = np.interp(nu,np.linspace(0.001,0.5,scaledA[i,j,:].shape[0]), scaledA[i,j,:])
                            new_condA[i,j,:] = np.interp(nu,np.linspace(0.001,0.5,scaledA[i,j,:].shape[0]), condA[i,j,:])
                    scaledA = new_scaledA
                    condA = new_condA


                if which_func == 1:
                    func = scaledA
                    nn=-1
                    plt.plot(nu[:nn],func[0,pp,:nn],marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:nn],func[0,mm,:nn],marker.next(),linestyle='-',linewidth=2)
                    plt.plot(nu[:nn],func[1,mm,:nn],marker.next(),linestyle='-',linewidth=2)


                    plt.ylabel(r"$Mesh\, Quality\,-\, min(Q_3)$",fontsize=font_size)


                    plt.legend(MaterialModels,loc='lower left',fontsize=legend_font_size)
                    if degree == 2 and mm == 5:
                        plt.legend(MaterialModels,loc='upper left',fontsize=legend_font_size)
                    if degree > 2:
                        plt.legend(MaterialModels,loc='upper left',fontsize=legend_font_size)
                        if degree == 4 and mm==5:
                            plt.legend(MaterialModels,loc='lower left',fontsize=legend_font_size)



                    if save:
                        if mm==3:
                            plt.savefig(SavePath+ResultsFile+"_MooneyRivlin.eps",format='eps',dpi=500)
                        elif mm==2:
                            plt.savefig(SavePath+ResultsFile+"_Neo-Hookean.eps",format='eps',dpi=500)
                        elif mm==4:
                            plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial.eps",format='eps',dpi=500)
                        elif mm==5:
                            plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial.eps",format='eps',dpi=500)


                
                


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

                    plt.legend(MaterialModels,loc='upper left',fontsize=legend_font_size)
                    plt.ylabel(r"$\kappa(A)$",fontsize=font_size)

                    y_formatter = mpl.ticker.ScalarFormatter(useOffset=True)
                    y_formatter.set_powerlimits((-4,4))
                    ax = plt.gca()
                    ax.yaxis.set_major_formatter(y_formatter)

                    if save:
                        if mm==3:
                            plt.savefig(SavePath+ResultsFile+"_MooneyRivlin_CondA.eps",format='eps',dpi=500)
                        elif mm==2:
                            plt.savefig(SavePath+ResultsFile+"_Neo-Hookean_CondA.eps",format='eps',dpi=500)
                        elif mm==4:
                            plt.savefig(SavePath+ResultsFile+"_NearlyIncompressibleMaterial_CondA.eps",format='eps',dpi=500)
                        elif mm==5:
                            plt.savefig(SavePath+ResultsFile+"_TransverselyIsotropicMaterial_CondA.eps",format='eps',dpi=500)



                plt.show()





        def plotter_all_materials(degree=3,which_func=1,linear=True,save=False):
            """all material models for a given formulation is plotted

                which_func          1 for scaledA
                                    0 for condA

                linear              True for plotting linearised materials
                                    False for plotting nonlinear materials
            """

            marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))

            rc('axes',color_cycle=['#D1655B','#44AA66','#72B0D7','#FACD85',
                '#4D5C75','#FFF056','#558C89','#F5CCBA'])
            


            ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/'
            # ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P2_orthogonal"
            ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P"+str(degree)
            SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

            DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   
            
            scaledA = DictOutput['ScaledJacobian']
            condA = DictOutput['ConditionNumber']
            # nu = DictOutput['PoissonsRatios'][0]
            nu = np.linspace(0.001,0.5,100)*1

            font_size = 18
            legend_font_size = 16

            plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)

            # print scaledA[0,2,:10]
            # print scaledA[1,2,:10]

            # exit(0)

            if linear:
                ind = 0
                append = "Linear_"
            else:
                ind=1
                append = "Nonlinear_"


            if which_func == 1:
                func = scaledA
                nn=-1

                for i in range(6):
                    if i==0 and ~linear:
                        ind=0
                    if i==1 and ~linear:
                        ind=0
                    plt.plot(func[ind,i,:],marker.next(),linestyle='-',linewidth=2)


            if linear:
                plt.legend([r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$IL\;neo-Hookean$",
                    r"$IL\;Mooney-Rivlin$",r"$IL\;Nearly\;Incompressible$",
                    r"$ILTI\;Hyperelastic$"],loc="upper left",fontsize=legend_font_size)
            else:
                plt.legend([r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$neo-Hookean$",
                    r"$Mooney-Rivlin$",r"$Nearly\;Incompressible$",
                    r"$TI\;Hyperelastic$"],loc="upper left",fontsize=legend_font_size)


            if which_func == 1:
                if save:
                    plt.savefig(SavePath+ResultsFile+"_"+append+"AllMaterials_scaledA.eps",format='eps',dpi=500)
                    # print SavePath+ResultsFile+"_"+append+"AllMaterials_scaledA.eps"


            plt.show()


        def plotter_bar(which_func=1,linear=True,save=False):
            """
                which_func          1 for scaledA
                                    0 for condA

                                    this function is fully designed for condA

            """

            fig, ax = plt.subplots()

            # rects = [None]*(3*6)
            rects = [None]*(5*6)
            counter = 0
            for degree in range(2,7):

                ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/MaterialFormulation_vs_Nu/'
                # ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P2_orthogonal"
                ResultsFile = "Mech2D_MaterialFormulation_vs_Nu_P"+str(degree)
                SavePath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Mech2D/"

                DictOutput =  loadmat(ResultsPath+ResultsFile+'.mat')   

                colors = ['#D1655B','#44AA66','#72B0D7','#FACD85',
                    '#558C89','#4D5C75','#F5CCBA']

                colors_err = [None]*7
                colors_err[0] = 'k'
                colors_err[1] = colors[0]
                colors_err[2] = colors[3]
                colors_err[3] = colors[2]
                colors_err[4] = colors[5]
                colors_err[5] = 'y'

                
                scaledA = DictOutput['ScaledJacobian']
                condA = DictOutput['ConditionNumber']
                nu = np.linspace(0.001,0.5,100)*1

                font_size = 22
                legend_font_size = 20


                if linear:
                    indd = 0
                    append = "Linear_"
                else:
                    indd = 1
                    append = "Nonlinear_"

                
                width=0.15
                ind = 1


                func = scaledA
                if which_func==0:
                    func = condA


                if which_func == 1:

                    # first find nan numbers and exclude them
                    if linear is False:
                        indices_container = []
                        for i in range(6):
                            indices = []
                            for j in range(func[1,i,:].shape[0]):
                                if ~np.isnan(func[1,i,j]):
                                    indices.append(j)
                            indices_container.append(indices) 

                        intersects = indices_container[0]

                        for i in range(1,6):
                            intersects = np.intersect1d(intersects,indices_container[i])
                        intersects = np.array(intersects,copy=False)

                        if intersects.size == 0:
                            intersects = np.arange(func[0,0,:].shape[0])

                    for i in range(6):
                        if i==0 and linear is False:
                            indd=0
                        elif i==1 and linear is False:
                            indd=0
                        else:
                            if linear:
                                indd=0
                            else:
                                indd=1

                        mean = np.mean(func[indd,i,:])
                        std = np.std(func[indd,i,:])
                        if linear is False:
                            mean = np.mean(func[indd,i,intersects])
                            std = np.std(func[indd,i,intersects])
                            # # HACK
                            # if degree>=4:
                            #     mean = np.NAN
                            #     std = np.NAN

                        rects[counter] = ax.bar((degree-1)+ind+i*width, mean, width, color=colors[i],
                            yerr=std,capstyle='butt',capsize=2,ecolor=colors_err[i])
                        counter+=1

                if which_func==1:
                    plt.ylim([0,1.8])
                    ax.set_yticklabels([0, 0.2, 0.4, 0.8, 1],fontsize=font_size)
                    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0]) 
                    plt.ylabel(r"$mean(min(Q_3))$",fontsize=font_size)

                if linear:
                    ax.legend((rects[0][0], rects[1][0], rects[2][0], rects[3][0], rects[4][0], rects[5][0]), 
                        (r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
                            r"$IL\;Mooney-Rivlin$",r"$IL\;Nearly\;Incompressible$",r"$ILTI\;Hyperelastic$"),
                        loc='upper right',fontsize=legend_font_size)
                else:
                    ax.legend((rects[0][0], rects[1][0], rects[2][0], rects[3][0], rects[4][0], rects[5][0]), 
                        (r"$II\;Linear\;Elastic$",r"$ITI\;Linear\;Elastic$",r"$neo-Hookean$",
                            r"$Mooney-Rivlin$",r"$Nearly\;Incompressible$",r"$TI\;Hyperelastic$"),
                        loc='upper right',fontsize=legend_font_size)
 

                # ax.set_xticklabels((r'$p=2$', r'$p=3$', r'$p=4$'),fontsize=font_size)
                # ax.set_xlim([2,4.9])
                # ax.set_xticks([2.45,3.5,4.4])

                ax.set_xticklabels((r'$p=2$', r'$p=3$', r'$p=4$', r'$p=5$', r'$p=6$'),fontsize=font_size)
                ax.set_xlim([2,6.9])
                ax.set_xticks([2.45,3.5,4.4,5.4,6.4])

                

            
            if save:
                if which_func == 1:
                    plt.savefig(SavePath+ResultsFile.split("P")[0]+append+"AllMaterials_scaledA.eps",format='eps',dpi=500)
                    # print SavePath+ResultsFile.split("P")[0]+append+"AllMaterials_scaledA.eps"
                else:
                    plt.savefig(SavePath+ResultsFile+"_"+append+"AllMaterials_condA.eps",format='eps',dpi=500)




            plt.show()
            # print rects





        # plotter(degree=5,which_func=1,save=True)
        # plotter(degree=6,which_func=1)

        # plotter_all_materials(degree=3,which_func=1,linear=False,save=True)
        # plotter_all_materials(degree=3,linear=False)

        plotter_bar(which_func=1,linear=False,save=True)
        # plotter_bar()








