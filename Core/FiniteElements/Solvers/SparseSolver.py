import numpy as np
from scipy.sparse.linalg import spsolve, bicgstab
from scipy.io import savemat, loadmat
from subprocess import call
import os



def SparseSolver(A,b,solver='direct',sub_type='UMFPACK'):


	# sol = np.array([])
	if solver == 'direct':
		# CALL DIRECT SOLVER
		if sub_type=='UMFPACK':
			sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
		elif sub_type=='MUMPS':
			# CALL JULIA'S MUMPS WRAPPER
			pwd = os.path.dirname(os.path.realpath(__file__))

			A = A.tocoo()
			# SAVE I, J & V TO FILES
			np.savetxt(pwd+"/rowIndA",A.row+1)
			np.savetxt(pwd+"/colPtrA",A.col+1)
			np.savetxt(pwd+"/valuesA",A.data,fmt="%9.16f")
			np.savetxt(pwd+"/shapeA",A.shape)
			np.savetxt(pwd+"/rhs",b,fmt="%9.16f")

			del A, b

			call(["julia",pwd+"/JuliaMUMPS.jl"])
			# print np.fromfile(pwd+"/solution")
			sol = np.loadtxt(pwd+"/solution")

			# REMOVE THE FILES
			os.remove(pwd+"/rowIndA")
			os.remove(pwd+"/colPtrA")
			os.remove(pwd+"/valuesA")
			os.remove(pwd+"/shapeA")
			os.remove(pwd+"/rhs")
			os.remove(pwd+"/solution")
			# exit(0)
			# pass
		elif sub_type=='super_lu':
			sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
	else:
		# CALL ITERATIVE SOLVER
		sol = bicgstab(A,b,tol=MainData.solve.tol)[0]

	return sol