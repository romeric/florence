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

 	C = 2									# ORDER OF BASIS FUNCTIONS (NOTE THAT C=P-1, WHERE P IS THE POLYNOMIAL DEGREE)
 	norder = 2  							# ORDER/NO OF QUADRATURE POINTS
 	plot = (0,3)							# PLOT FLAG FOR BEM 
 	nrplot = (1,'last')						# PLOT FLAG FOR NEWTON-RAPHSON CONVERGENCE
 	write = 0								# FLAG FOR WRITING THE RESULTS IN VTK/MAT/EPS/DAT ETC
 	Parallel = True 						# MULTI-PROCESSING 
 	nCPU = MP.cpu_count()					# CPU COUNT FOR MULTI-PROCESSING
 	Parallel = False
 	# nCPU = 8



from Main.FiniteElements.MainFEM import main
import matplotlib.pyplot as plt


t_FEM = time.time()
# nu = np.linspace(0.01,0.499,5)
nu = np.linspace(0.01,0.491,20)
nu = np.linspace(0.01,0.471,20)
# nu = np.linspace(0.472,0.495,1)
E = np.linspace(10,1e7,num=1) 


condA=np.zeros((nu.shape[0],E.shape[0]))
scaledA = np.copy(condA)
for i in range(nu.shape[0]):
	for j in range(E.shape[0]):
		MainData.nu = nu[i]
		MainData.E = E[j]
		main(MainData)	
		# print np.min(MainData.ScaledJacobian)
		# print MainData.solve.condA

		scaledA[i,j] = np.min(MainData.ScaledJacobian)
		condA[i,j] = MainData.solve.condA

# np.savetxt('/home/roman/Desktop/DumpReport_2/nu.txt', nu)
# np.savetxt('/home/roman/Desktop/DumpReport_2/E.txt', E)

# np.savetxt('/home/roman/Desktop/DumpReport_2/condA_'+MainData.MaterialArgs.Type+'.txt', condA)
# np.savetxt('/home/roman/Desktop/DumpReport_2/scaledA_'+MainData.MaterialArgs.Type+'.txt', scaledA)

# np.savetxt('/home/roman/Desktop/DumpReport_2/ani_condA_'+MainData.MaterialArgs.Type+'.txt', condA)
# np.savetxt('/home/roman/Desktop/DumpReport_2/ani_scaledA_'+MainData.MaterialArgs.Type+'.txt', scaledA)




plt.plot(nu,scaledA,'#F88379')
plt.show()

print 'Time taken for the entire analysis was', time.time()-t_FEM, 'seconds \n'
# MEMORY USAGE INFORMATION
# print 'Global sparse matrix needed', MainData.spmat, 'MB of memory with IJV indices requiring', MainData.ijv, 'MB'











