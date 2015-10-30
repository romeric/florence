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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import savemat




t_FEM = time.time()
# nu = np.linspace(0.001,0.495,20)
nu = np.linspace(0.001,0.495,2)
# nu = np.linspace(0.01,0.471,20)
E = np.array([10])
# p = [2,3,4,5,6]
p = [2,3]
 

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
		# print p[i]
		main(MainData,Results)	
		# scaledA[i,j] = np.min(MainData.ScaledJacobian)
		# condA[i,j] = MainData.solve.condA
	
savemat('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/RESULTS_DIR/Mech2D_P_vs_Nu.mat',Results)

# Results['MeshPoints_P'+str(p[0])] = 2
# print Results['MeshPoints_P2']
# print Results

# np.savetxt('/home/roman/Dropbox/MATLAB_MESHING_PLOTS/p_vs_nu.txt', scaledA.T)

# print scaledA.T
# plt.plot(nu,scaledA,'#F88379')
# plt.plot(nu,scaledA,'#F88379')
# plt.show()

xmin = p[0]
xmax = p[-1]
ymin = nu[0]
ymax = nu[-1]

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
           interpolation='nearest', cmap=cm.viridis)
plt.colorbar()
# plt.axis('equal')
# plt.axis('off')
# plt.show()



# x,y,temp = np.loadtxt('data.txt').T #Transposed for easier unpacking
# x = np.linspace(0,100,100)
# y = np.linspace(0,100,100)
# temp = np.random.rand(100*100)
# temp =  np.sort(temp)
# nrows, ncols = 100, 100
# grid = temp.reshape((nrows, ncols))

# plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()),
#            interpolation='nearest', cmap=cm.gist_rainbow)
# plt.show()











