import numpy as np
from scipy.sparse.linalg import spsolve, bicgstab, gmres, cg
from scipy.io import savemat, loadmat
from subprocess import call
import os

def SparseSolver(A,b,solver='direct',sub_type='UMFPACK',tol=1e-05):


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
            JuliaDict = {'rowIndA':A.row.astype(np.int64)+1,
                        'colPtrA':A.col.astype(np.int64)+1,
                        'valuesA':A.data,'shapeA':A.shape,
                        'rhs':b}

            savemat(pwd+"/JuliaDict.mat",JuliaDict)

            del A, b

            call(["julia",pwd+"/JuliaMUMPS.jl"])
            sol = np.loadtxt(pwd+"/solution")
            # FromJulia = loadmat(pwd+"JuliaDict.mat")
            # sol = FromJulia["sol"]

            # REMOVE THE FILES
            os.remove(pwd+"/JuliaDict.mat")
            os.remove(pwd+"/solution")

        elif sub_type=='super_lu':
            sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
    else:
        # CALL ITERATIVE SOLVER
        # sol = bicgstab(A,b,tol=tol)[0]
        # sol = gmres(A,b,tol=tol)[0]
        sol = cg(A,b,tol=tol)[0]

    return sol

# def SparseSolver(A,b,solver='direct',sub_type='UMFPACK',tol=1e-05):


#   # sol = np.array([])
#   if solver == 'direct':
#       # CALL DIRECT SOLVER
#       if sub_type=='UMFPACK':
#           sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
#       elif sub_type=='MUMPS':
#           # CALL JULIA'S MUMPS WRAPPER
#           pwd = os.path.dirname(os.path.realpath(__file__))

#           A = A.tocoo()
#           # SAVE I, J & V TO FILES
#           np.savetxt(pwd+"/rowIndA",A.row+1)
#           np.savetxt(pwd+"/colPtrA",A.col+1)
#           np.savetxt(pwd+"/valuesA",A.data,fmt="%9.16f")
#           np.savetxt(pwd+"/shapeA",A.shape)
#           np.savetxt(pwd+"/rhs",b,fmt="%9.16f")

#           del A, b

#           call(["julia",pwd+"/JuliaMUMPS.jl"])
#           # print np.fromfile(pwd+"/solution")
#           sol = np.loadtxt(pwd+"/solution")

#           # REMOVE THE FILES
#           os.remove(pwd+"/rowIndA")
#           os.remove(pwd+"/colPtrA")
#           os.remove(pwd+"/valuesA")
#           os.remove(pwd+"/shapeA")
#           os.remove(pwd+"/rhs")
#           os.remove(pwd+"/solution")
#           # exit(0)
#           # pass
#       elif sub_type=='super_lu':
#           sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
#   else:
#       # CALL ITERATIVE SOLVER
#       sol = bicgstab(A,b,tol=tol)[0]

#   return sol