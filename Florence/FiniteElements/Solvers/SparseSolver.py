# import numpy as np
# from scipy.sparse import isspmatrix_csc,  isspmatrix_csr
# from scipy.sparse.linalg import spsolve, bicgstab, gmres, lgmres, cg, spilu, LinearOperator
# from scipy.io import savemat, loadmat
# from subprocess import call
# import os
# try:
#     from pyamg import *
#     from pyamg.gallery import *
#     pyamg_imp = True
# except ImportError:
#     pyamg_imp = False

# # @profile
# def SparseSolver(A,b,solver='direct',sub_type='UMFPACK',tol=1e-06):

#     # sol = np.array([])
#     if solver == 'direct':
#         # CALL DIRECT SOLVER
#         if sub_type=='UMFPACK':
#             if A.dtype != np.float64:
#                 A = A.astype(np.float64)
#             sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
#         elif sub_type=='MUMPS':
#             # CALL JULIA'S MUMPS WRAPPER
#             pwd = os.path.dirname(os.path.realpath(__file__))

#             A = A.tocoo()
#             # SAVE I, J & V TO FILES
#             JuliaDict = {'rowIndA':A.row.astype(np.int64)+1,
#                         'colPtrA':A.col.astype(np.int64)+1,
#                         'valuesA':A.data,'shapeA':A.shape,
#                         'rhs':b}

#             savemat(pwd+"/JuliaDict.mat",JuliaDict)

#             del A, b

#             call(["julia",pwd+"/JuliaMUMPS.jl"])
#             sol = np.loadtxt(pwd+"/solution")
#             # FromJulia = loadmat(pwd+"JuliaDict.mat")
#             # sol = FromJulia["sol"]

#             # REMOVE THE FILES
#             os.remove(pwd+"/JuliaDict.mat")
#             os.remove(pwd+"/solution")

#         elif sub_type=='super_lu':
#             sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
#     elif solver == "iterative":
#         # CALL ITERATIVE SOLVER
#         # sol = bicgstab(A,b,tol=tol)[0]
#         # sol = gmres(A,b,tol=tol)[0]
#         sol = cg(A,b,tol=1e-04)[0]

#         # PRECONDITIONED ITERATIVE SOLVER - CHECK
#         # P = spilu(A, drop_tol=1e-5)
#         # M_x = lambda x: P.solve(x)
#         # n = A.shape[0]
#         # m = A.shape[1]
#         # M = LinearOperator((n * m, n * m), M_x)
#         # sol = lgmres(A, b, tol=1e-4, M=M)[0]

#     elif solver == "multigrid":
#         if pyamg_imp is False:
#             raise ImportError('A multigrid solver was not found')

#         if A.dtype != b.dtype:
#             # DOWN-CAST
#             b = b.astype(A.dtype)

#         if not isspmatrix_csr(A):
#             A = A.tocsr()
        
#         # AMG METHOD
#         ml = ruge_stuben_solver(A)
#         sol = ml.solve(b,tol=tol)

#         # EXPLICIT CALL TO KYROLOV SOLVERS WITH AMG PRECONDITIONER
#         # THIS IS TYPICALLY FASTER BUT THE TOLERANCE NEED TO BE SMALLER, TYPICALLY 1e-10
#         # ml = smoothed_aggregation_solver(A)
#         # M = ml.aspreconditioner()
#         # if tol > 1e-9:
#         #     tol = 1e-10
#         # sol, info = gmres(A, b, M=M, tol=tol)

#     return sol