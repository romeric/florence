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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat
from matplotlib import rc


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

	# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
	## for Palatino and other serif fonts use:
	rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
	rc('text', usetex=True)

	Run = 1
	if Run:
		t_FEM = time.time()
		nu = np.linspace(0.001,0.495,100)
		# nu = np.linspace(0.01,0.495,2)
		E = np.array([1e05])
		p = [2,3,4,5,6]
		# p = [6]
		 

		Results = {'PolynomialDegrees':p,'PoissonsRatios':nu,'Youngs_Modulus':E}
			# 'MeshPoints':None,'MeshElements':None,
			# 'MeshEdges':None, 'MeshFaces':None,'TotalDisplacement':None}

		condA=np.zeros((len(p),nu.shape[0]))
		scaledA = np.copy(condA)
		for i in range(len(p)):
			MainData.C = p[i]-1
			for j in range(nu.shape[0]):
				MainData.nu = nu[j]
				MainData.E = E
				MainData.isScaledJacobianComputed = False
				main(MainData,Results)	
				CondExists = getattr(MainData.solve,'condA',None)
				# ScaledExists = getattr(MainData.solve,'scaledA',None)
				scaledA[i,j] = np.min(MainData.ScaledJacobian)
				if CondExists is not None:
					condA[i,j] = MainData.solve.condA
				else:
					condA[i,j] = np.NAN

		Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
		Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
		Results['MaterialModel'] = MainData.MaterialArgs.Type
		# print Results['ScaledJacobian']

		# savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'+MainData.MaterialArgs.Type+'_FeketeSpacing.mat',Results)
		savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'+MainData.MaterialArgs.Type+'_Orthogonal.mat',Results)
		# savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'+MainData.MaterialArgs.Type+'_EqualSpacing.mat',Results)
		# savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'+MainData.MaterialArgs.Type+'.mat',Results)
		# savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu_'+MainData.MaterialArgs.Type+'2.mat',Results)
		t_FEM = time.time()-t_FEM
		print 'Time taken for the entire analysis was ', t_FEM, 'seconds'
		np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/DONE', [t_FEM])

	if not Run:
		# import h5py as hpy 
		ResultsPath = '/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/'

		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementalLinearElastic_EqualSpacing'
		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementalLinearElastic_FeketeSpacing'
		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementalLinearElastic_Orthogonal'

		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementallyLinearisedNeoHookean_EqualSpacing'
		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementallyLinearisedNeoHookean_FeketeSpacing'
		# ResultsFile = 'Mech2D_P_vs_Nu_IncrementallyLinearisedNeoHookean_Orthogonal'

		# ResultsFile = 'Mech2D_P_vs_Nu_NeoHookean_2_EqualSpacing'
		ResultsFile = 'Mech2D_P_vs_Nu_NeoHookean_2_FeketeSpacing'
		# ResultsFile = 'Mech2D_P_vs_Nu_NeoHookean_2_Orthogonal'

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

		# print xmin, xmax, ymin, ymax
		scaledA = scaledA[::-1,:]
		condA = condA[::-1,:]

		X,Y=np.meshgrid(p,nu)
		# print X
		# print scaledA



		plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='bicubic', cmap=cm.viridis)
		# plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),interpolation='nearest', cmap=cm.viridis)
		# # plt.colorbar()

		# # plt.axis('equal')
		# # plt.axis('off')
		# # tick_locs = [0, 1, 2, 3, 4]
		# tick_locs = [2, 3, 4, 5, 6]
		# tick_locs = np.linspace(2.5,6,6).tolist()
		tick_locs = [2.45,3.25,4.,4.82,5.55]
		tick_lbls = [2, 3, 4, 5, 6]
		plt.yticks(tick_locs, tick_lbls)
		tick_locs = [0,1,2,3,4,5]
		tick_lbls = [0,0.1,0.2,0.3,0.4,0.5]
		plt.xticks(tick_locs, tick_lbls)
		plt.ylabel(r'$Polynomial\, Degree\,\, (p)$',fontsize=18)
		plt.xlabel(r"$Poisson's\, Ratio\,\, (\nu)$",fontsize=18)
		plt.title(r"$Mesh\, Quality\,\, (Q_1)$",fontsize=18)

		ax, _ = mpl.colorbar.make_axes(plt.gca(), shrink=0.8)
		cbar = mpl.colorbar.ColorbarBase(ax, cmap=cm.viridis,
	                       norm=mpl.colors.Normalize(vmin=-0, vmax=1))
		cbar.set_clim(0, 1)

		# plt.xlim([0,5])
		# plt.ylim([2,6])
		ResultsPath+ResultsFile+'.mat'
		# plt.savefig(ResultsPath+'Mech2D/'+ResultsFile+'.eps',format='eps',dpi=1000)

		plt.show()










