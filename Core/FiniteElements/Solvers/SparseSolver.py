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
        sol = cg(A,b,tol=1e-04)[0]

    return sol