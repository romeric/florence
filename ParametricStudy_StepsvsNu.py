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


def runMain():

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

    t_FEM = time.time()
    nu = np.linspace(0.001,0.495,10)
    # nu = np.linspace(0.001,0.495,1)
    # nu = np.linspace(0.35,0.495,1)
    E = 1e05
    E_A = 2.5*E
    G_A = E/2.

    ProblemPath = "/home/roman/Dropbox/Florence/Examples/FiniteElements/Wing2D/"

    ProblemDataFile = ["sd7003_Stretch25","sd7003_Stretch50",
        "sd7003_Stretch100","sd7003_Stretch200","sd7003_Stretch400",
        "sd7003_Stretch800","sd7003_Stretch1600"]
    nStep = [1,2,5,10,25,50]
    # nStep = [1,2,5,10,25]
    # nStep = [1,2,5]

    # ProblemDataFile = ["sd7003_Stretch25"]
    # nStep=[10]

    

    Results = {'PolynomialDegrees':MainData.C+1,'PoissonsRatios':nu,'Youngs_Modulus':E,"E_A":E_A,"G_A":G_A}

    condA=np.zeros((3,len(nStep),len(ProblemDataFile),nu.shape[0]))
    scaledA = np.copy(condA)
    scaledAFF = np.copy(condA)
    scaledAHH = np.copy(condA)
    whole_scaledA = np.zeros((3,len(nStep),len(ProblemDataFile),nu.shape[0],4000))
    # LOOP OVER FORMULATIONS
    for m in range(0,3):

        if m==0:
            MainData.AnalysisType = "Linear"
            MainData.MaterialArgs.Type = "IncrementalLinearElastic"
        elif m==1:
            MainData.AnalysisType = "Linear"
            MainData.MaterialArgs.Type = "NeoHookean_2"
        elif m==2:
            MainData.AnalysisType = "Nonlinear"
            MainData.MaterialArgs.Type = "NeoHookean_2"

        # LOOP OVER INCREMENTS
        for k in range(len(nStep)):
            MainData.LoadIncrement = nStep[k]
            # LOOP OVER DATA FILES
            for i in range(len(ProblemDataFile)):
                MainData.MeshInfo.FileName = ProblemPath+ProblemDataFile[i]

                # if MainData.C==5 and MainData.AnalysisType == "Nonlinear" \
                #     and nStep[k]>6:
                #     continue
                # print int(MainData.MeshInfo.FileName.split("h")[-1])
                # if MainData.C==3 and MainData.AnalysisType == "Nonlinear" \
                #     and int(MainData.MeshInfo.FileName.split("h")[-1]) > 25 \
                #     and MainData.LoadIncrement > 6:
                    # continue

                # LOOP OVER POISSON'S RATIOS
                for j in range(nu.shape[0]):
                    MainData.MaterialArgs.nu = nu[j]
                    MainData.MaterialArgs.E = E
                    MainData.MaterialArgs.E_A = E_A
                    MainData.MaterialArgs.G_A = G_A

                    print 'Poisson ratio is:', MainData.MaterialArgs.nu, "Degree is:", MainData.C+1
                    print MainData.AnalysisType, MainData.MaterialArgs.Type, "LoadIncrements", MainData.LoadIncrement, 
                    print "FileName", MainData.MeshInfo.FileName.split("/")[-1]

                    # if MainData.C==3 and MainData.AnalysisType == "Nonlinear" \
                    #     and int(MainData.MeshInfo.FileName.split("h")[-1]) > 25 \
                    #     and MainData.LoadIncrement > 11:

                    #     MainData.isScaledJacobianComputed = False
                    #     print 
 
                    #     scaledA[m,k,i,j] = np.NAN
                    #     scaledAFF[m,k,i,j] = np.NAN
                    #     scaledAHH[m,k,i,j] = np.NAN
                    #     whole_scaledA[m,k,i,j,:] = np.NAN
                    #     condA[m,k,i,j] = np.NAN

                    # elif MainData.C==5 and MainData.AnalysisType == "Nonlinear" \
                    #     and MainData.LoadIncrement > 6:

                    #     MainData.isScaledJacobianComputed = False
                    #     print 
                    #     scaledA[m,k,i,j] = np.NAN
                    #     scaledAFF[m,k,i,j] = np.NAN
                    #     scaledAHH[m,k,i,j] = np.NAN
                    #     whole_scaledA[m,k,i,j,:] = np.NAN
                    #     condA[m,k,i,j] = np.NAN

                    # else:
                    #     MainData.isScaledJacobianComputed = False
                    #     main(MainData,Results)  
                    #     print 
                    #     CondExists = getattr(MainData.solve,'condA',None)
                    #     # ScaledExists = getattr(MainData.solve,'scaledA',None)
                    #     scaledA[m,k,i,j] = np.min(MainData.ScaledJacobian)
                    #     scaledAFF[m,k,i,j] = np.min(MainData.ScaledFF)
                    #     scaledAHH[m,k,i,j] = np.min(MainData.ScaledHH)
                    #     whole_scaledA[m,k,i,j,:MainData.ScaledJacobian.shape[0]] = MainData.ScaledJacobian
                    #     if CondExists is not None:
                    #         condA[m,k,i,j] = MainData.solve.condA
                    #     else:
                    #         condA[m,k,i,j] = np.NAN


                    # Direct for p==2
                    MainData.isScaledJacobianComputed = False
                    main(MainData,Results)  
                    print 
                    CondExists = getattr(MainData.solve,'condA',None)
                    # ScaledExists = getattr(MainData.solve,'scaledA',None)
                    scaledA[m,k,i,j] = np.min(MainData.ScaledJacobian)
                    scaledAFF[m,k,i,j] = np.min(MainData.ScaledFF)
                    scaledAHH[m,k,i,j] = np.min(MainData.ScaledHH)
                    whole_scaledA[m,k,i,j,:MainData.ScaledJacobian.shape[0]] = MainData.ScaledJacobian
                    if CondExists is not None:
                        condA[m,k,i,j] = MainData.solve.condA
                    else:
                        condA[m,k,i,j] = np.NAN

    Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
    Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
    Results['WholeScaledJacobian'] = whole_scaledA
    Results['ScaledFF'] = scaledAFF
    Results['ScaledHH'] = scaledAHH

    
    fname = "P"+str(MainData.C+1)+".mat"
    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'

    # print fpath+fname
    savemat(fpath+fname,Results)

    t_FEM = time.time()-t_FEM
    print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
    np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/DONE_P'+str(MainData.C+1), [t_FEM])


def plotter_surface():

    # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu_Old/Wing2D_Steps_vs_Nu_'
    fname = "P"+str(p)+".mat"
    

    DictOutput =  loadmat(fpath+fname)   

    scaledA = DictOutput['ScaledJacobian']
    condA = DictOutput['ConditionNumber']
    # nu = DictOutput['PoissonsRatios'][0]
    nu = np.linspace(0.001,0.5,10)*100
    p = DictOutput['PolynomialDegrees'][0]


    xmin = 0
    xmax = 50
    ymin = nu[0]
    ymax = nu[-1]

    # print np.min(DictOutput['WholeScaledJacobian'][0,1,0,0][:2171])
    nStep= [1,2,5,10,25,50]

    # scaledA  = np.zeros((6,10))
    # for i in range(len(nStep)):
    #     for j in range(10):
    #         scaledA[i,j] = np.min(DictOutput['WholeScaledJacobian'][0,i,0,j][:2171])
    


    scaledA = DictOutput['ScaledJacobian'][0,:,0,:]
    # scaledA = np.fliplr(scaledA)

    # imshow is a direct matrix-to-pixel visualtion so flip the 
    # matrix upside down
    scaledA = np.flipud(scaledA)

    # print scaledA
    # exit()
    # print scaledA.shape
    # scaledA = scaledA[2,:,0,:]
    # exit()
    # x = np.linspace(0.0,0.5,10)
    # y = np.array([1,2,5,10,25,50])
    # y = np.array([1,2,5])
    # Z = scaledA[0,:,0,:]
    # Z1 = scaledA[0,:,6,:]
    # print x.shape, y.shape, z.shape
    # exit()

    # X,Y,Z = np.meshgrid(x,y,z)

    # X,Y = np.meshgrid(x,y)

    # print X.shape, Y.shape, Z.shape
    # exit()
    # 
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False, shade=False)
    # surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.viridis, linewidth=0, antialiased=False, shade=True)
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=False)

    # ax.view_init(90,0)
    # plt.axis('off')


    # ax.set_ylabel(r'$Increments$',fontsize=18)
    # ax.set_xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=18)
    # ax.set_zlabel(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=18)



def plotter_individual_imshow(p=2,which_q=3,save=False):

    """ which_q:            3 for scaled Jacobian Q_3
                            1 for F:F Q_1
                            2 for H:H Q_2 

    """

    stretch = [25, 50, 100, 200, 400, 800, 1600]
    formulations = ["Linear","Linearised","Nonlinear"]

    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
    fname = "P"+str(p)+".mat"
    spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Steps_vs_Nu_"
    
    # print fpath+fname
    DictOutput =  loadmat(fpath+fname)   

    scaledA = DictOutput['ScaledJacobian']
    condA = DictOutput['ConditionNumber']
    nu = np.linspace(0.001,0.5,10)*100
    p = DictOutput['PolynomialDegrees'][0]


    xmin = 0
    xmax = 50
    ymin = nu[0]
    ymax = nu[-1]

    nStep= [1,2,5,10,25,50]


    # print DictOutput['ScaledJacobian'][0,:,0,:]
    # print
    # print DictOutput['ScaledJacobian'][0,:,1,:]

    # print DictOutput['ScaledJacobian'][0,2,0,:]
    # print
    # print DictOutput['ScaledJacobian'][0,2,1,:]

    # print DictOutput['ScaledJacobian'][0,0,1,-4], DictOutput['ScaledJacobian'][0,-1,1,-4]
    print DictOutput['ScaledJacobian'][2,2,0,:]
    print DictOutput['PoissonsRatios']
    exit()
    # which stretching 
    for lstr in range(0,7):
        # which formulation
        for m in range(0,3):
            if which_q == 3:
                scaledA = DictOutput['ScaledJacobian'][m,:,lstr,:]
                # scaledA =  np.zeros((6,10))
                # for i in range(6):
                #     for j in range(10):
                #         # print np.min(DictOutput['WholeScaledJacobian'][m,i,lstr,j][:2171])
                #         scaledA[i,j] = np.min(DictOutput['WholeScaledJacobian'][m,i,lstr,j][:2171])
            elif which_q == 1:
                scaledA = DictOutput['ScaledFF'][m,:,lstr,:]
            elif which_q == 2:
                scaledA = DictOutput['ScaledHH'][m,:,lstr,:]

            if p==6 and m==2:
                scaledA[3:,:] = np.NAN

            # imshow is a direct matrix-to-pixel visualtion so flip the 
            # matrix upside down
            scaledA = np.flipud(scaledA)
            # scaledA = scaledA[::-1,:] # or

            plt.figure()
            # tick_locs = [1.0,2.0,5.0,10.0,24.0,49.0]
            # tick_lbls = [1, 2, 5, 10, 25, 50]
            tick_locs = [0,10,20,30,40,50]
            tick_lbls = [1,10,20,30,40,50]
            plt.yticks(tick_locs, tick_lbls)
            tick_locs = np.array([0,1,2,3,4,5])*10
            tick_lbls = [0.0,0.1,0.2,0.3,0.4,0.5]
            plt.xticks(tick_locs, tick_lbls)

            plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bicubic', cmap=cm.viridis)
            # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bilinear', cmap=cm.viridis)
            # plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='nearest', cmap=cm.viridis)
            # plt.imshow(scaledA)

            font_size = 20
            plt.ylabel(r'$Number\;of\;Increments$',fontsize=font_size)
            plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=font_size)
            # plt.title(r"$Mesh\, Quality\,\, (Q_3)$",fontsize=18)

            ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.8)
            # cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis,
                               # norm=mpl.colors.Normalize(vmin=-0, vmax=1))
            cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis)
            cbar.set_clim(0, 1)
            cbar.set_ticks([0,0.1,0.2,0.3,0.4,0.5,.6,0.7,0.8,0.9,1])
            plt.clim(0,1)


            # fig = plt.gcf()
            # fig.set_size_inches(8,8)

            if which_q==3:
                # sname = spath+fname.split(".")[0]+"_"+str(formulations[m])+"_Stretch"+str(stretch[lstr])+'.eps'
                sname = spath+fname.split(".")[0]+"_"+str(formulations[m])+"_Stretch"+str(stretch[lstr])+'.png'
            elif which_q==1:
                # for scaledFF
                sname = spath+fname.split(".")[0]+"_"+str(formulations[m])+"_Stretch"+str(stretch[lstr])+'_Q1.png'
            elif which_q==2:
                sname = spath+fname.split(".")[0]+"_"+str(formulations[m])+"_Stretch"+str(stretch[lstr])+'_Q2.png'
            # print sname

            if save:
            #     # plt.savefig(sname,format='eps',dpi=300)
                plt.savefig(sname,format='png',dpi=100)

            plt.show()
            plt.close()



def plotter_quality_evolution(p):

    stretch = [25, 50, 100, 200, 400, 800, 1600]
    formulations = ["Linear","Linearised","Nonlinear"]

    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
    fname = "P"+str(p)+".mat"
    spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Quality_Improvement_"
    sname = "P"+str(p)
    
    DictOutput =  loadmat(fpath+fname)   

    scaledA = DictOutput['ScaledJacobian']
    # scaledA = DictOutput['ScaledFF']
    condA = DictOutput['ConditionNumber']
    nu = np.linspace(0.0,0.5,10)*100
    p = DictOutput['PolynomialDegrees'][0]

    colors = ['#D1655B','#44AA66','#72B0D7','#FACD85',
                    '#558C89','#4D5C75','#F5CCBA']


    fig, ax = plt.subplots()
    font_size = 20
    legend_font_size = 20
    rc('font',**{'family':'serif','serif':['Palatino'],'size':font_size})

    mm=-1
    print nu[mm]/100.
    # if p==6:
        # mm=-2
    # print scaledA[0,-1,2,mm], scaledA[0,0,2,mm]
    # print scaledA[1,-1,2,mm], scaledA[1,0,2,mm]
    percentage_improvement = np.zeros((7,3))
    for i in range(7):
        for j in range(3):
            percentage_improvement[i,j] =  (scaledA[j,-1,i,mm] - scaledA[j,0,i,mm])/scaledA[j,0,i,mm]*100
            # if i==2 and j==1:
                # print percentage_improvement[i,j]

    percentage_improvement[:,2] = 0
    # percentage_improvement = np.abs(percentage_improvement)
    # print percentage_improvement
    # percentage_improvement[:,:2] = np.log(percentage_improvement[:,:2])
    # percentage_improvement[:,2] = 0
    print percentage_improvement
    print scaledA[0,-1,6,mm], scaledA[0,0,6,mm]
    return

    

    nmax = np.max(percentage_improvement)
    print np.exp(nmax)
    for i in range(7):
        if p!=4:
            plt.plot([i+1.1,i+1.1],[percentage_improvement[i,0],nmax*1.4],'-k')
            tt1 = plt.text(i+1,nmax*1.9,np.around(scaledA[0,-1,i,mm],3),rotation=90,fontsize=font_size-2)

            # mmax = max(percentage_improvement[i,1],percentage_improvement[i,0])
            plt.plot([i+1.3,i+1.3],[percentage_improvement[i,1],nmax*2.2],'-k')
            tt2 = plt.text(i+1.2,nmax*2.7,np.around(scaledA[1,-1,i,mm],3),rotation=90,fontsize=font_size-2)

        elif p==4:
            plt.plot([i+1.1,i+1.1],[percentage_improvement[i,0],nmax*1.2],'-k')
            tt1 = plt.text(i+1,nmax*2.4,np.around(scaledA[0,-1,i,mm],3),rotation=90,fontsize=font_size-2)

            plt.plot([i+1.3,i+1.3],[percentage_improvement[i,1],nmax*2.2],'-k')
            tt2 = plt.text(i+1.2,nmax*3.8,np.around(scaledA[1,-1,i,mm],3),rotation=90,fontsize=font_size-2)
    
    # percentage_improvement = np.log(percentage_improvement)
    nmin = np.min(percentage_improvement)
    plt.ylim([2*nmin,5*nmax])

    width = 0.2
    rects = [None]*21
    ind = 1
    counter = 0
    for i in range(7):
        for j in range(3):
            rects[counter] = ax.bar(ind+i+j*width, percentage_improvement[i,j], width, color=colors[j])
            counter +=1

    ax.legend((rects[0][0], rects[1][0], rects[2][0]), 
                            (r"$II\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
                                r"$neo-Hookean$"),
                            loc='upper left',fontsize=legend_font_size)

    ax.set_xticklabels((r'$25$',r'$50$',r'$100$',r'$200$',r'$400$',r'$800$',r'$1600$'),fontsize=font_size)
    plt.xticks([1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2],rotation=45)
    if p==4:
        plt.yticks([-5, 0, 10, 20])
    elif p==6:
        plt.yticks([0, 50, 100])
    else:
        plt.yticks([0, 10, 20])
    # plt.ylabel(r'log($(Q_{3_{50}} - Q_{3_{1}})/Q_{3_{1}} \times 100)$',fontsize=font_size)
    plt.ylabel(r'$(Q_{3_{50}} - Q_{3_{1}})/Q_{3_{1}} \times 100$',fontsize=font_size)


    # print percentage_improvement
    plt.xlim([-0.01,8.3])
    # print spath+sname+"_"+str(np.around(nu[mm]/100.,3))
    plt.savefig(spath+sname+".eps", format="eps", dpi=300)
    
    plt.show()



def plotter_distribution_q3(p=2,save=False):

    from scipy.stats import norm

    stretch = [25, 50, 100, 200, 400, 800, 1600]
    nStep = [1,2,5,10,25,50]
    formulations = ["Linear","Linearised","Nonlinear"]

    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
    # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu_Old/Wing2D_Steps_vs_Nu_'
    fname = "P"+str(p)+".mat"
    spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Quality_Distribution_"
    
    # print fpath+fname
    DictOutput =  loadmat(fpath+fname)   

    scaledA = DictOutput['ScaledJacobian']
    condA = DictOutput['ConditionNumber']
    # nu = DictOutput['PoissonsRatios'][0]
    nu = np.linspace(0.001,0.5,10)*100

    colors = ['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D']

    nelems = [2171,2339,2511,2683,2855,3027,3199]
    lstr=4 # for fix level of stretching 
    incr = 0
    nus = [-8,-4,-2]


    for mm in nus:
        # print nu[mm]/100.
        for incr in [0,3,5]:
            bar_1 = DictOutput['WholeScaledJacobian'][0,incr,lstr,mm][:nelems[lstr]]
            bar_2 = DictOutput['WholeScaledJacobian'][1,incr,lstr,mm][:nelems[lstr]]
            bar_3 = DictOutput['WholeScaledJacobian'][2,incr,lstr,mm][:nelems[lstr]]

            all_nan = False
            if np.isnan(bar_3).all():
                all_nan = True
            bar_3 = np.nan_to_num(bar_3)

            hist_1 =  plt.hist(bar_1,range=(0.,1.),log=True)[0]
            hist_2 =  plt.hist(bar_2,range=(0.,1.),log=True)[0]
            hist_3 =  plt.hist(bar_3,range=(0.,1.),log=True)[0]
            if all_nan is True:
                hist_3 = np.zeros_like(hist_3)+1

            plt.close()
            fig = plt.figure()
            ax = fig.gca()
            width_1=0.09
            width_2=0.07
            width_3=0.05

            rects = [None]*3
            rects[0] = plt.bar(np.linspace(0,1,10)-width_1,hist_1,width=width_1,log=True,color=colors[0])
            rects[1] = plt.bar(np.linspace(0,1,10)-width_2*1.15,hist_2,width=width_2,log=True,color=colors[1])
            rects[2] = plt.bar(np.linspace(0,1,10)-width_3*1.4,hist_3,width=width_3,log=True,color=colors[2])

            plt.grid('on')
            plt.xlim([0,1])
            plt.ylim([0,10000])

            legend_font_size = 20
            font_size = 22
            plt.xlabel(r"$Scaled\;Jacobian\;(Q_3)$",fontsize=font_size)
            plt.ylabel(r"$Number\;of\;Elements$",fontsize=font_size)
            ax.legend((rects[0][0], rects[1][0], rects[2][0]), 
                                    (r"$II\;Linear\;Elastic$",r"$IL\; neo-Hookean$",
                                        r"$neo-Hookean$"),
                                    loc='upper left',fontsize=legend_font_size)

            # plt.hist(DictOutput['WholeScaledJacobian'][0,0,0,-4][:2171],log=True)
            spath = (spath+"Nu"+str(np.around(nu[mm]/100.,3))).split(".")
            spath = spath[0]+spath[1]+"_Inc"+str(nStep[incr])+"_P"+str(p)+".eps"
            print spath

            if save:
                plt.savefig(spath,format="eps",dpi=300)

            spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Quality_Distribution_"
            plt.show()



def plotter_quality_measures(p=2,save=False):

    stretch = [25, 50, 100, 200, 400, 800, 1600]
    nStep = [1,2,5,10,25,50]
    formulations = ["Linear","Linearised","Nonlinear"]

    fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu/Wing2D_Steps_vs_Nu_'
    # fpath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Steps_vs_Nu_Old/Wing2D_Steps_vs_Nu_'
    fname = "P"+str(p)+".mat"
    spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Quality_Measures_"
    
    # print fpath+fname
    DictOutput =  loadmat(fpath+fname)   

    scaledA = DictOutput['ScaledJacobian']
    scaledAFF = DictOutput['ScaledFF']
    scaledAHH = DictOutput['ScaledHH']
    # condA = DictOutput['ConditionNumber']
    # nu = DictOutput['PoissonsRatios'][0]
    nu = np.linspace(0.001,0.5,10)*100

    colors = ['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D']


    marker = itertools.cycle(('o', 's', 'x'))
    linestyle = itertools.cycle(('-', '--', '-.'))

    font_size = 24
    legend_font_size = 22


    mm=0
    incr = 2
    for lstr in [0,3,6]:
        for mm in range(3):
            plt.plot(nu/100.,scaledAFF[mm,incr,lstr,:],marker.next(),linestyle=linestyle.next(),color=colors[0],linewidth=3)
            plt.plot(nu/100.,scaledAHH[mm,incr,lstr,:],marker.next(),linestyle=linestyle.next(),color=colors[1],linewidth=3)
            plt.plot(nu/100.,scaledA[mm,incr,lstr,:],marker.next(),linestyle=linestyle.next(),color=colors[2],linewidth=3)

            # scaledAFF[mm,0,0,:]/3./scaledA[mm,0,0,:]
            # print scaledAFF[mm,0,0,:]**2/3./2**(3./2)

            plt.legend([r"$Q_1$",r"$Q_2$",r"$Q_3$"],loc="best",fontsize=legend_font_size)
            plt.xlabel(r"$Poisson's\;Ratio\;(\nu)$",fontsize=font_size)
            plt.ylabel(r"$Quality\;Measures$",fontsize=font_size)
            plt.grid('on')
            plt.xlim([0,0.5])
            plt.ylim([0.,1.0])

            if mm==0:
                appender = "IILinearElastic"
            elif mm==1:
                appender = "ILneoHookean"
            elif mm==2:
                appender = "neoHookean"

            spath = spath+appender+"_Stretch"+str(stretch[lstr])+"_P"+str(p)+".eps"
            print spath

            fig = plt.gcf()
            fig.set_size_inches(7,7)
            # plt.axis('equal')

            if save:
                plt.savefig(spath,format="eps",dpi=300)

            spath = "/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/figures/Wing2D/Wing2D_Quality_Measures_"
            # plt.show()
            plt.close()




if __name__ == '__main__':

    Run = 0
    if Run:
        runMain()

    if not Run:

        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from matplotlib import rc
        import itertools

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib.mlab import griddata

        # marker = itertools.cycle(('o', 's', 'x', '+', '*','.'))
        marker = itertools.cycle(('o', 's', 'x'))

        rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
        rc('text', usetex=True)

        # rc('axes',color_cycle=['#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        rc('axes',color_cycle=['#D1655B','#FACD85','#72B0D7','#E79C5D','#4D5C75','#E79C5D'])
        # rc('axes',**{'prop_cycle':['#D1655B','#FACD85','#70B9B0','#72B0D7','#E79C5D']})



        p=4
        plotter_individual_imshow(p)
        # plotter_individual_imshow(p,which_q=1,save=True)


        # plotter_quality_evolution(p)
        
        # plotter_distribution_q3(p)
        # plotter_distribution_q3(p,save=True)

        # plotter_quality_measures(p)
        # plotter_quality_measures(p,save=True)