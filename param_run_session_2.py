import imp, os, sys, time, cProfile	
from datetime import datetime
import numpy as np
import multiprocessing as MP
# AVOID WRITING .pyc OR .pyo FILES
sys.dont_write_bytecode

# START THE ANALYSIS
print 'Initiating the routines... Current time is', datetime.now().time()

# ALLOCATE/BUILD THE GENERAL DATA FOR THE ANALYSIS
class MainData(object):
 	"""General data such as directories, files, analysis session, etc that needs to be loaded a priori"""
 	pwd = os.path.dirname(os.path.realpath('__file__'))
 	session = 'FEM'
 	# session = 'BEM'
 	# session = 'BEM3D'
 	# session = 'Coupled'

 	C = 1									# ORDER OF BASIS FUNCTIONS (NOTE THAT C=P-1, WHERE P IS THE POLYNOMIAL DEGREE)
 	norder = 2  							# ORDER/NO OF QUADRATURE POINTS
 	plot = (0,3)							# PLOT FLAG FOR BEM 
 	nrplot = (1,'last')						# PLOT FLAG FOR NEWTON-RAPHSON CONVERGENCE
 	write = 0								# FLAG FOR WRITING THE RESULTS IN VTK/MAT/EPS/DAT ETC
 	Parallel = True 						# MULTI-PROCESSING 
 	nCPU = MP.cpu_count()					# CPU COUNT FOR MULTI-PROCESSING
 	Parallel = False
 	# nCPU = 8
 	__PARALLEL__ = True
	__NO_DEBUG__ = True
	__VECTORISATION__ = True
	numCPU = 16



from Main.FiniteElements.MainFEM import main
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat, loadmat
from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino'],'size':18})
rc('text', usetex=True)

if 0:
	t_FEM = time.time()
	# nu = np.linspace(0.001,0.495,20)
	nu = np.linspace(0.001,0.495,100)
	# nu = np.linspace(0.01,0.471,50)
	E = np.array([10])
	p = [2,3,4,5,6]
	# p = [2,3]
	 

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
			main(MainData,Results)	
			scaledA[i,j] = np.min(MainData.ScaledJacobian)
			condA[i,j] = MainData.solve.condA

	Results['ScaledJacobian'] = scaledA # one given row contains all values of nu for a fixed p
	Results['ConditionNumber'] = condA # one given row contains all values of nu for a fixed p
	Results['MaterialModel'] = MainData.MaterialArgs.Type
	# print Results['MaterialModel']

	savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu'+MainData.MaterialArgs.Type+'.mat',Results)
	t_FEM = time.time()-t_FEM
	np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/DONE2', [t_FEM])

if 1:
	# import h5py as hpy 
	DictOutput = {}
	# DictOutput = loadmat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_NuLinearModel.mat',DictOutput)
	DictOutput =  loadmat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_NuLinearModel.mat')
	scaledA = DictOutput['ScaledJacobian']
	condA = DictOutput['ConditionNumber']
	nu = DictOutput['PoissonsRatios'][0]
	nu = np.linspace(0.001,0.5,100)*10
	p = DictOutput['PolynomialDegrees'][0]
	# print nu



	xmin = p[0]
	xmax = p[-1]
	ymin = nu[0]
	ymax = nu[-1]

	# print xmin, xmax, ymin, ymax
	# print nu
	# print scaledA[4,:]
	# plt.plot(scaledA.T)
	# print scaledA.shape
	scaledA = scaledA[::-1,:]

	# scaledA = np.array([ [ 0.71740108,  0.54950547,  0.46659289,  0.45130078,  0.40435613],
	# 					 [ 0.72034474,  0.55847273,  0.48098432,  0.46758115,  0.41693345],
	# 					 [ 0.72383733,  0.56930343,  0.4983352 ,  0.48726023,  0.43181139],
	# 					 [ 0.71948871,  0.58265657,  0.51966925,  0.51154286,  0.44967149],
	# 					 [ 0.71411154,  0.5995507 ,  0.54654843,  0.54229378,  0.47141262],
	# 					 [ 0.70788926,  0.62165491,  0.58148482,  0.58257963,  0.49751104],
	# 					 [ 0.70057028,  0.64610798,  0.62880681,  0.62543501,  0.53082617],
	# 					 [ 0.69169948,  0.66548884,  0.68645197, 0.64263533 ,  0.57468561],
	# 					 [ 0.67997664,  0.70926285,  0.70469248,  0.66298808,  0.62766284],
	# 					 [ 0.63774012,  0.8030901 ,  0.67324841,  0.61970387,  0.59386199]
	#  ])

	# plt.imshow(scaledA, extent=(xmin, xmax, ymax, ymin),
	           # interpolation='nearest', cmap=cm.YlOrRd)
	# plt.imshow(scaledA, extent=(xmin, xmax, ymin, ymax),
	#            interpolation='nearest', cmap=cm.YlOrRd)
	# plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),
	           # interpolation='nearest', cmap=cm.Oranges) 

	plt.imshow(scaledA, extent=(ymin, ymax, xmin, xmax),
	           interpolation='bicubic', cmap=cm.viridis)
	# plt.pcolor(vmin=0)

	# plt.colorbar()

	# plt.imshow(scaledA, extent=(xmin, xmax, ymin, ymax),
	#            interpolation='bilinear', cmap=cm.viridis)
	# plt.imshow(scaledA, extent=(xmin, xmax, ymin, ymax))
	# plt.colorbar()

	# plt.axis('equal')
	# plt.axis('off')
	# tick_locs = [0, 1, 2, 3, 4]
	tick_locs = [2, 3, 4, 5, 6]
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
	plt.show()










