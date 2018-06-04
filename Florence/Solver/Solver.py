from __future__ import print_function
import os, platform
from time import time
from warnings import warn
import numpy as np
import scipy as sp
from scipy.sparse import issparse, isspmatrix_coo, isspmatrix_csr, isspmatrix_csc
from scipy.sparse.linalg import spsolve, cg, cgs, bicgstab, gmres, lgmres, splu, spilu, LinearOperator, onenormest
from subprocess import call


__all__ = ["LinearSolver"]


class LinearSolver(object):
    """Base class for all linear sparse direct and iterative solvers"""

    def __init__(self,linear_solver="direct", linear_solver_type="umfpack",
        apply_preconditioner=False, preconditioner="smoothed_aggregation",
        iterative_solver_tolerance=1.0e-12, reduce_matrix_bandwidth=False,
        out_of_core=False, geometric_discretisation=None, dont_switch_solver=False):
        """

            input:
                linear_solver:          [str] type of solver either "direct",
                                        "iterative" or "multigrid"

                linear_solver_type      [str] type of direct or linear solver to
                                        use, for instance "umfpack", "superlu" or
                                        "mumps" for direct solvers, or "cg", "gmres"
                                        etc for iterative solvers or "amg" for algebraic
                                        multigrid solver. See WhichSolvers method for
                                        the complete set of available linear solvers

                preconditioner:         [str] either "amg_smoothed_aggregation",
                                        or "ruge_stuben" or "rootnode" for
                                        a preconditioner based on algebraic multigrid
                                        or "incomplete_lu" for scipy's spilu linear
                                        operator

                geometric_discretisation:
                                        [str] type of geometric discretisation used, for
                                        instance for FEM discretisations this would correspond
                                        to "tri", "quad", "tet", "hex" etc

                dont_switch_solver:     Do not switch between solvers automatically


        """

        self.is_sparse = True
        self.solver_type = linear_solver
        self.solver_subtype = linear_solver_type
        self.requires_cuthill_mckee = reduce_matrix_bandwidth
        self.iterative_solver_tolerance = iterative_solver_tolerance
        self.apply_preconditioner = apply_preconditioner
        self.preconditioner_type = preconditioner
        self.out_of_core = False
        self.geometric_discretisation = geometric_discretisation
        self.dont_switch_solver = dont_switch_solver
        self.reuse_factorisation = False
        self.solver_context_manager = None

        self.has_amg_solver = True
        if platform.python_implementation() == "PyPy":
            self.has_amg_solver = False
        else:
            try:
                import pyamg
            except ImportError:
                self.has_amg_solver = False

        self.has_umfpack = True
        try:
            from scikits.umfpack import spsolve
        except ImportError:
            self.has_umfpack = False

        self.has_mumps = False
        try:
            from mumps.mumps_context import MUMPSContext
            self.has_mumps = True
        except ImportError:
            self.has_mumps = False

        self.has_pardiso = False
        try:
            import pypardiso
            self.has_pardiso = True
        except ImportError:
            self.has_pardiso = False

        self.switcher_message = False

        # self.analysis_type = "static"
        # self.analysis_nature = "linear"

    def SetSolver(self,linear_solver="direct", linear_solver_type="umfpack",
        apply_preconditioner=False, preconditioner="amg_smoothed_aggregation",
        iterative_solver_tolerance=1.0e-12, reduce_matrix_bandwidth=False,
        geometric_discretisation=None):
        """

            input:
                linear_solver:          [str] type of solver either "direct",
                                        "iterative" or "multigrid"

                linear_solver_type      [str] type of direct or linear solver to
                                        use, for instance "umfpack", "superlu" or
                                        "mumps" for direct solvers, or "cg", "gmres"
                                        etc for iterative solvers or "amg" for algebraic
                                        multigrid solver. See WhichSolvers method for
                                        the complete set of available linear solvers

                preconditioner:         [str] either "amg_smoothed_aggregation" for
                                        a preconditioner based on algebraic multigrid
                                        or "incomplete_lu" for scipy's spilu linear
                                        operator

                geometric_discretisation:
                                        [str] type of geometric discretisation used, for
                                        instance for FEM discretisations this would correspond
                                        to "tri", "quad", "tet", "hex" etc

        """

        self.solver_type = linear_solver
        self.solver_subtype = "umfpack"
        self.iterative_solver_tolerance = iterative_solver_tolerance
        self.apply_preconditioner = apply_preconditioner
        self.requires_cuthill_mckee = reduce_matrix_bandwidth
        self.geometric_discretisation = geometric_discretisation


    @property
    def WhichLinearSolver(self):
        return self.solver_type, solver_subtype

    @property
    def WhichLinearSolvers(self):
        return {"direct":["superlu","umfpack","mumps"],
            "iterative":["cg","bicg","cgstab","bicgstab"
            "gmres","lgmres"],"multigrid":["amg"]}


    def GetPreconditioner(self,A, type="amg_smoothed_aggregation"):
        """Applies a suitable preconditioner to sparse matrix A
            based on algebraic multigrid of incomplete LU/Cholesky factorisation

            input:
                A:                      [csc_matrix or csc_matrix]
                type:                   [str] either "amg_smoothed_aggregation" for
                                        a preconditioner based on algebraic multigrid
                                        or "incomplete_lu" for scipy's spilu linear
                                        operator

            returns:                    A preconditioner that can be used in conjunction
                                        with scipy's sparse linear iterative solvers
                                        (the M keyword in scipy's iterative solver)
        """

        if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
            raise TypeError("Matrix must be in CSC or CSR sparse format for preconditioning")

        ml = smoothed_aggregation_solver(A)
        return ml.aspreconditioner()


    def GetCuthillMcKeePermutation(self,A):
        """Applies Cuthill-Mckee permutation to reduce the sparse matrix bandwidth

            input:
                A:                    [csc_matrix or csr_matrix]

            returns:
                perm:                 [1D array] of permutation such that A[perm,:][:,perm]
                                      has its non-zero elements closer to the diagonal
        """

        if not (isspmatrix_csc(A) or isspmatrix_csr(A)):
            raise TypeError("Matrix must be in CSC or CSR sparse format "
                "for Cuthill-McKee permutation")

        if int(sp.__version__.split('.')[1]) >= 15:
            from scipy.sparse.csgraph import reverse_cuthill_mckee
            perm = reverse_cuthill_mckee(A)
        else:
            from Florence.Tensor import symrcm
            perm = symrcm(A)

        return perm


    def SparsityPattern(self,A):
        import matplotlib.pyplot as plt
        plt.spy(A)
        plt.grid('on')
        plt.show()


    def Solve(self, A, b, reuse_factorisation=False):
        """Solves the linear system of equations"""

        self.reuse_factorisation = reuse_factorisation
        if self.solver_type != "direct" and self.reuse_factorisation is True:
            warn("Re-using factorisation for non-direct solvers is not possible. The pre-conditioner is going to be reused instead")

        # DECIDE IF THE SOLVER TYPE IS APPROPRIATE FOR THE PROBLEM
        if self.switcher_message is False and self.dont_switch_solver is False:
            # PREFER PARDISO OR MUMPS IF AVAILABLE
            if self.has_pardiso:
                self.solver_type = "direct"
                self.solver_subtype = "pardiso"
            elif self.has_mumps:
                self.solver_type = "direct"
                self.solver_subtype = "mumps"
            elif b.shape[0] > 100000 and self.has_amg_solver:
                self.solver_type = "multigrid"
                self.solver_subtype = "amg"
                print('Large system of equations. Switching to algebraic multigrid solver')
                self.switcher_message = True
            # elif mesh.points.shape[0]*MainData.nvar > 50000 and MainData.C < 4:
                # self.solver_type = "direct"
                # self.solver_subtype = "MUMPS"
                # print 'Large system of equations. Switching to MUMPS solver'
            elif b.shape[0] > 70000 and self.geometric_discretisation=="hex" and self.has_amg_solver:
                self.solver_type = "multigrid"
                self.solver_subtype = "amg"
                print('Large system of equations. Switching to algebraic multigrid solver')
                self.switcher_message = True
            else:
                self.solver_type = "direct"
                self.solver_subtype = "umfpack"


        if self.solver_type == 'direct':
            # CALL DIRECT SOLVER
            if self.solver_subtype=='umfpack' and self.has_umfpack:
                if A.dtype != np.float64:
                    A = A.astype(np.float64)

                if self.solver_context_manager is None:
                    if self.reuse_factorisation is False:
                        sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
                        # from scikits import umfpack
                        # sol = umfpack.spsolve(A, b)
                    else:
                        from scikits import umfpack
                        lu = umfpack.splu(A)
                        sol = lu.solve(b)
                        self.solver_context_manager = lu
                else:
                    sol = self.solver_context_manager.solve(b)


            elif self.solver_subtype=='mumps' and self.has_mumps:

                from mumps.mumps_context import MUMPSContext
                t_solve = time()
                A = A.tocoo()
                # False means non-symmetric - Do not change it to True. True means symmetric pos def
                # which is not the case for electromechanics
                if self.solver_context_manager is None:
                    context = MUMPSContext((A.shape[0], A.row, A.col, A.data, False), verbose=False)
                    context.analyze()
                    context.factorize()
                    sol = context.solve(rhs=b)

                    if self.reuse_factorisation:
                        self.solver_context_manager = context
                else:
                    sol = self.solver_context_manager.solve(rhs=b)

                print("MUMPS solver time is {}".format(time() - t_solve))

                return sol


            elif self.solver_subtype == "pardiso" and self.has_pardiso:
                # NOTE THAT THIS PARDISO SOLVER AUTOMATICALLY SAVES THE RIGHT FACTORISATION
                import pypardiso
                A = A.tocsr()
                t_solve = time()
                sol = pypardiso.spsolve(A,b)
                print("Pardiso solver time is {}".format(time() - t_solve))

            else:
                # FOR 'super_lu'
                if A.dtype != np.float64:
                    A = A.astype(np.float64)
                A = A.tocsc()

                if self.solver_context_manager is None:
                    if self.reuse_factorisation is False:
                        sol = spsolve(A,b,permc_spec='MMD_AT_PLUS_A',use_umfpack=True)
                    else:
                        lu = splu(A)
                        sol = lu.solve(b)
                        self.solver_context_manager = lu
                else:
                    sol = self.solver_context_manager.solve(b)



        elif self.solver_type == "iterative":
            # CALL ITERATIVE SOLVER
            # sol = bicgstab(A,b,tol=tol)[0]
            # sol = gmres(A,b,tol=tol)[0]
            sol = cg(A,b,tol=self.iterative_solver_tolerance)[0]

            # PRECONDITIONED ITERATIVE SOLVER - CHECK
            # P = spilu(A, drop_tol=1e-5)
            # M_x = lambda x: P.solve(x)
            # n = A.shape[0]
            # m = A.shape[1]
            # M = LinearOperator((n * m, n * m), M_x)
            # sol = lgmres(A, b, tol=1e-4, M=M)[0]

        elif self.solver_type == "multigrid":
            if self.has_amg_solver is False:
                raise ImportError('A multigrid solver was not found')
            from pyamg import ruge_stuben_solver, rootnode_solver, smoothed_aggregation_solver

            if A.dtype != b.dtype:
                # DOWN-CAST
                b = b.astype(A.dtype)

            if not isspmatrix_csr(A):
                A = A.tocsr()

            t_solve = time()
            # AMG METHOD
            if self.solver_context_manager is None:
                if self.preconditioner_type=="smoothed_aggregation":
                    # EXPLICIT CALL TO KYROLOV SOLVERS WITH AMG PRECONDITIONER
                    # THIS IS TYPICALLY FASTER BUT THE TOLERANCE NEED TO BE SMALLER, TYPICALLY 1e-10
                    # GMRES IS TYPICALLY THE FASTEST
                    ml = smoothed_aggregation_solver(A)
                    M = ml.aspreconditioner()
                    if self.iterative_solver_tolerance > 1e-9:
                        self.iterative_solver_tolerance = 1e-10
                    # sol, info = bicgstab(A, b, M=M, tol=self.iterative_solver_tolerance)
                    # sol, info = cgs(A, b, M=M, tol=self.iterative_solver_tolerance)
                    sol, info = gmres(A, b, M=M, tol=self.iterative_solver_tolerance)
                elif self.preconditioner_type == "ruge_stuben":
                    M = ruge_stuben_solver(A)
                    sol = M.solve(b,tol=self.iterative_solver_tolerance)
                elif self.preconditioner_type == "rootnode":
                    # EXPLICIT CALL TO KYROLOV SOLVERS WITH AMG PRECONDITIONER
                    # ml = rootnode_solver(A, smooth=('energy', {'degree':2}), strength='evolution' )
                    # M = ml.aspreconditioner(cycle='V')
                    ml = rootnode_solver(A)
                    M = ml.aspreconditioner()
                    if self.iterative_solver_tolerance > 1e-9:
                        self.iterative_solver_tolerance = 1e-10
                    sol, info = gmres(A, b, M=M, tol=self.iterative_solver_tolerance)


                if self.reuse_factorisation:
                    self.solver_context_manager = M

            else:
                M = self.solver_context_manager
                if self.preconditioner_type=="smoothed_aggregation":
                    sol, info = gmres(A, b, M=M, tol=self.iterative_solver_tolerance)
                elif self.preconditioner_type == "ruge_stuben":
                    sol = M.solve(b,tol=self.iterative_solver_tolerance)
                elif self.preconditioner_type == "rootnode":
                    sol, info = gmres(A, b, M=M, tol=self.iterative_solver_tolerance)


            print("AMG solver time is {}".format(time() - t_solve))

        return sol


    def GetConditionNumber(self,A):
        self.matrix_condition_number = onenormest(K_b)
        return self.matrix_condition_number


    def CleanUp(self):
        import gc
        del self.solver_context_manager
        gc.collect()
        self.solver_context_manager = None
