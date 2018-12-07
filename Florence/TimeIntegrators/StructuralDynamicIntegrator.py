from __future__ import print_function
import gc, os, sys
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.linalg as sla
from numpy.linalg import norm
from time import time
from copy import deepcopy
from warnings import warn
from time import time

from Florence.FiniteElements.Assembly import Assemble
from Florence import Mesh

__all__ = ["StructuralDynamicIntegrator"]


class StructuralDynamicIntegrator(object):
    """Base class for structural time integerators
    """

    def __init__(self):
        super(StructuralDynamicIntegrator, self).__init__()
        self.gamma   = 0.5
        self.beta    = 0.25
        self.lump_rhs = False

        self.electric_dofs = None
        self.mechanical_dofs = None
        self.columns_in_mech = None
        self.columns_out_mech = None
        self.columns_in_electric = None
        self.columns_out_electric = None

        # NEEDS TO BE SET FOR EVERY STEP/INCREMENT
        self.bc_changed_at_this_step = False


    def GetBoundaryInfo(self, mesh, formulation, boundary_condition, increment=0):

        # Do not compute for steps at which BC does not change
        if increment != 0:
            if formulation.fields == "electro_mechanics":
                if self.columns_in_mech is not None:
                    test_dofs_m = np.intersect1d(boundary_condition.columns_in,self.mechanical_dofs)
                    test_dofs_e = np.intersect1d(boundary_condition.columns_in,self.electric_dofs)
                    if np.array_equal(test_dofs_m, self.columns_in_mech) and\
                        np.array_equal(test_dofs_e, self.columns_in_electric):
                        self.bc_changed_at_this_step = False
                        return
                    else:
                        self.bc_changed_at_this_step = True

            elif formulation.fields == "mechanics":
                if self.columns_out_mech is not None:
                    if np.array_equal(self.columns_out_mech, boundary_condition.columns_out):
                        self.bc_changed_at_this_step = False
                        return
                    else:
                        self.bc_changed_at_this_step = True


        all_dofs = np.arange(mesh.points.shape[0]*formulation.nvar)
        if formulation.fields == "electro_mechanics":
            self.electric_dofs = all_dofs[formulation.nvar-1::formulation.nvar]
            self.mechanical_dofs = np.setdiff1d(all_dofs,self.electric_dofs)

            # GET BOUNDARY CONDITON FOR THE REDUCED MECHANICAL SYSTEM
            self.columns_in_mech = np.intersect1d(boundary_condition.columns_in,self.mechanical_dofs)
            self.columns_in_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_in)

            # GET BOUNDARY CONDITON FOR THE REDUCED ELECTROSTATIC SYSTEM
            self.columns_in_electric = np.intersect1d(boundary_condition.columns_in,self.electric_dofs)
            self.columns_in_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_in)


            # GET FREE MECHANICAL DOFs
            self.columns_out_mech = np.intersect1d(boundary_condition.columns_out,self.mechanical_dofs)
            self.columns_out_mech_idx = np.in1d(self.mechanical_dofs,boundary_condition.columns_out)
            self.columns_out_mech_reverse_idx = np.in1d(boundary_condition.columns_out,self.columns_out_mech)

            # GET FREE ELECTROSTATIC DOFs
            self.columns_out_electric = np.intersect1d(boundary_condition.columns_out,self.electric_dofs)
            self.columns_out_electric_idx = np.in1d(self.electric_dofs,boundary_condition.columns_out)
            self.columns_out_electric_reverse_idx = np.in1d(boundary_condition.columns_out,
                self.columns_out_electric)

            self.applied_dirichlet_mech = boundary_condition.applied_dirichlet[self.columns_out_mech_reverse_idx]
            self.applied_dirichlet_electric = boundary_condition.applied_dirichlet[self.columns_out_electric_reverse_idx]

            # MAPPED QUANTITIES
            out_idx = np.in1d(all_dofs,boundary_condition.columns_out)
            idx_electric = all_dofs[formulation.nvar-1::formulation.nvar]
            idx_mech = np.setdiff1d(all_dofs,idx_electric)

            self.all_electric_dofs = np.arange(mesh.points.shape[0])
            self.electric_out = self.all_electric_dofs[out_idx[idx_electric]]
            self.electric_in = np.setdiff1d(self.all_electric_dofs,self.electric_out)

            self.all_mech_dofs = np.arange(mesh.points.shape[0]*formulation.ndim)
            self.mech_out = self.all_mech_dofs[out_idx[idx_mech]]
            self.mech_in = np.setdiff1d(self.all_mech_dofs,self.mech_out)

        elif formulation.fields == "mechanics":
            self.electric_dofs = []
            self.mechanical_dofs = all_dofs
            self.columns_out_mech = boundary_condition.columns_out
            self.columns_out_mech_reverse_idx = np.ones_like(self.columns_out_mech).astype(bool)

            self.mech_in = boundary_condition.columns_in
            self.mech_out = boundary_condition.columns_out

            self.applied_dirichlet_mech = boundary_condition.applied_dirichlet


    def ComputeMassMatrixInfo(self, M, formulation, fem_solver):
        """Computes the inverse of lumped mass matrix and so on
        """

        invM = None
        if formulation.fields == "electro_mechanics":
            if fem_solver.mass_type == "lumped":
                M = M.ravel()
                invM = np.zeros_like(M)
                invM[self.mechanical_dofs] = np.reciprocal(M[self.mechanical_dofs])
                M_mech = M[self.mechanical_dofs]
            else:
                M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
        else:
            if fem_solver.mass_type == "lumped":
                M = M.ravel()
                M_mech = M
                invM = np.reciprocal(M)
            else:
                M_mech = M

        return M_mech, invM



    def ComputeEnergyDissipation(self, function_space, mesh, material, formulation, fem_solver,
        Eulerx, U, NeumannForces, M, velocities):

        if formulation.fields == "electro_mechanics":
            Eulerp = U[:,-1]

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords = Eulerx[mesh.elements[elem,:],:]

        if formulation.fields == "electro_mechanics":
            ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]
            internal_energy += formulation.GetEnergy(function_space, material,
                LagrangeElemCoords, EulerElemCoords, ElectricPotentialElem, fem_solver, elem)

            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = 0.5*np.dot(velocities.ravel(),M_mech.dot(velocities.ravel()))
        else:
            internal_energy += formulation.GetEnergy(function_space, material,
                LagrangeElemCoords, EulerElemCoords, fem_solver, elem)

            kinetic_energy = 0.5*np.dot(velocities.ravel(),M.dot(velocities.ravel()))

        external_energy = np.dot(U.ravel(),NeumannForces.ravel())

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy



    def ComputePowerDissipation(self, function_space, mesh, material, formulation, fem_solver,
        Eulerx, U, NeumannForces, M, velocities, accelerations):

        if formulation.fields == "electro_mechanics":
            Eulerp = U[:,-1]
        if velocities.ndim == 1:
            velocities = velocities.reshape(U.shape[0],formulation.ndim)

        internal_energy = 0.
        for elem in range(mesh.nelem):
            LagrangeElemCoords = mesh.points[mesh.elements[elem,:],:]
            EulerElemCoords    = Eulerx[mesh.elements[elem,:],:]
            VelocityElem       = velocities[mesh.elements[elem,:],:]

        if formulation.fields == "electro_mechanics":
            ElectricPotentialElem = Eulerp[mesh.elements[elem,:]]
            internal_energy += formulation.GetLinearMomentum(function_space, material,
                LagrangeElemCoords, EulerElemCoords, VelocityElem, ElectricPotentialElem, fem_solver, elem)

            M_mech = M[self.mechanical_dofs,:][:,self.mechanical_dofs]
            kinetic_energy = np.dot(velocities.ravel(),M_mech.dot(accelerations.ravel()))
        else:
            internal_energy += formulation.GetLinearMomentum(function_space, material,
                LagrangeElemCoords, EulerElemCoords, VelocityElem, fem_solver, elem)

            kinetic_energy = np.dot(velocities.ravel(),M.dot(accelerations.ravel()))

        external_energy = np.dot(velocities.ravel(),NeumannForces[self.mechanical_dofs].ravel())

        total_energy = internal_energy + kinetic_energy - external_energy
        return total_energy, internal_energy, kinetic_energy, external_energy






    def UpdateFixMechanicalDoFs(self, AppliedDirichletInc, fsize, nvar):
        """Updates the geometry (DoFs) with incremental Dirichlet boundary conditions
            for fixed/constrained degrees of freedom only. Needs to be applied per time steps"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.mech_out,0] = AppliedDirichletInc

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dU

    def UpdateFreeMechanicalDoFs(self, sol, fsize, nvar):
        """Updates the geometry with iterative solutions of Newton-Raphson
            for free degrees of freedom only. Needs to be applied per time NR iteration"""

        # GET TOTAL SOLUTION
        TotalSol = np.zeros((fsize,1))
        TotalSol[self.mech_in,0] = sol

        # RE-ORDER SOLUTION COMPONENTS
        dU = TotalSol.reshape(int(TotalSol.shape[0]/nvar),nvar)

        return dU





    def LogSave(self, fem_solver, formulation, U, Eulerp, Increment):
        if fem_solver.print_incremental_log:
            dmesh = Mesh()
            dmesh.points = U
            dmesh_bounds = dmesh.Bounds
            if formulation.fields == "electro_mechanics":
                _bounds = np.zeros((2,formulation.nvar))
                _bounds[:,:formulation.ndim] = dmesh_bounds
                _bounds[:,-1] = [Eulerp.min(),Eulerp.max()]
                print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),_bounds)
            else:
                print("\nMinimum and maximum incremental solution values at increment {} are \n".format(Increment),dmesh_bounds)

        # SAVE INCREMENTAL SOLUTION IF ASKED FOR
        if fem_solver.save_incremental_solution:
            # FOR BIG MESHES
            if Increment % fem_solver.incremental_solution_save_frequency !=0:
                return
            from scipy.io import savemat
            filename = fem_solver.incremental_solution_filename
            if filename is not None:
                if ".mat" in filename:
                    filename = filename.split(".")[0]
                savemat(filename+"_"+str(Increment),
                    {'solution':np.hstack((U,Eulerp[:,None]))},do_compression=True)
            else:
                raise ValueError("No file name provided to save incremental solution")
