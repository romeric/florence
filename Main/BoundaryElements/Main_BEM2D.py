######################################################################################################################################################
# Builtin Imports
import numpy as np
import scipy as sp
from scipy import linalg										# For linear algebra
import numpy.linalg as la										# For linear algebra
import cProfile													# For profiling
from time import time
import sys, os, imp
from scipy.sparse.linalg import spsolve	
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix 


#######################################################################################################################################################
# Problem Imports
# from Problems.BoundaryElements.Problem_Rectangle import ProblemData_BEM, BoundaryConditions, ComputeErrorNorms
# from Problems.BoundaryElements.Problem_Arch import ProblemData_BEM, BoundaryConditions, ComputeErrorNorms
# from Problems.BoundaryElements.Problem_Rectangle_LM import ProblemData_BEM, BoundaryConditions, DiscontinuousGlobalCoord, ComputeErrorNorms

# from Problems.BoundaryElements.Problem_Arch_LM import ProblemData_BEM, BoundaryConditions, DiscontinuousGlobalCoord
# import Problems.BoundaryElements.Problem_Arch_LM as Pr 
# Pr = imp.load_source('Problem_Arch','/home/roman/Dropbox/Python/Problems/BoundaryElements/Problem_Arch_LM/ProblemDataBEM_LM_2.py')
Pr = imp.load_source('Problem_Rectangle','/home/roman/Dropbox/Python/Problems/BoundaryElements/Problem_Rectangle/ProblemDataBEM_1.py')



########################################################################################################################################################
# User Imports
from Core.QuadratureRules import GaussQuadrature
from Core.BoundaryElements.PreProcessBEM2D import GenerateCoordinates, CoordsJacobianRadiusatGaussPoints, CoordsJacobianRadiusatGaussPoints_LM
from Core.BoundaryElements.GetBases import GetBases
from Core.BoundaryElements.Assembly import AssemblyBEM2D, AssemblyBEM2D_Sparse
from Core.BoundaryElements.Sort_BEM import Sort_BEM
from Core.BoundaryElements.PostProcessBEM2D import InteriorPostProcess, GetTotalSolution
from Core.BoundaryElements.WritePlotBEM2D import WritePlotBEM2D


# Main routine 
def main(general_data,C,nx,ny,norder=3,postprocess=0,writeplot=0,computeerror=0):
	# Read problem data file
	C, boundary_elements, boundary_points, element_connectivity, internal_points, mesh, geo_args = Pr.ProblemData_BEM(C,nx,ny)
	# Gauss Integration - Linear and Logarithmic
	z,w = GaussQuadrature(C+norder)
	# Get basis at all integration points
	Basis, dN = GetBases(C,z)
	# Get global coordinates 
	global_coord =  GenerateCoordinates(boundary_elements,boundary_points,C,z)
	# Modify global coordinates for discontinuous meshes
	if geo_args.Lagrange_Multipliers == 'activated':
		global_coord = Pr.DiscontinuousGlobalCoord(global_coord,C,geo_args)
	# Get necessary variables at integration points
	Jacobian=0; nx=0; ny=0; XCO=0; YCO=0
	if geo_args.Lagrange_Multipliers == 'activated':
		Jacobian, nx, ny, XCO, YCO = CoordsJacobianRadiusatGaussPoints_LM(boundary_elements,global_coord,C,Basis,dN,w,element_connectivity)
	else:
		Jacobian, nx, ny, XCO, YCO = CoordsJacobianRadiusatGaussPoints(boundary_elements,global_coord,C,Basis,dN,w)
	# Compute kernel matrices
	stiffness_K1, stiffness_K2 =  AssemblyBEM2D(C,global_coord,boundary_elements,element_connectivity,dN,Basis,w,z,Jacobian,nx,ny,XCO,YCO,geo_args)
	# stiffness_K1, stiffness_K2 =  AssemblyBEM2D_Sparse(C,global_coord,boundary_elements,element_connectivity,dN,Basis,w,z,Jacobian,nx,ny,XCO,YCO,geo_args)
	# Apply boundary condition
	boundary_data = Pr.BoundaryConditions(global_coord,C,geo_args)
	# Sort LHS and RHS
	global_K1,mm, total_LHS,total_RHS, LHS2LHS, LHS2RHS, RHS2LHS, RHS2RHS = Sort_BEM(boundary_data,stiffness_K1,stiffness_K2)
	# Solve the system of linear equations
	sol = sp.linalg.solve(global_K1,mm)

	print 'Total number of DoFs for the system is', global_K1.shape[0] 

	#####################################################
	# Post Process

	# Get the total solution
	total_sol = GetTotalSolution(sol,boundary_data,LHS2LHS,RHS2LHS)
	# Compute potential and fluxes at interior
	if postprocess:
		POT, FLUX1, FLUX2 =  InteriorPostProcess(total_sol,internal_points,global_coord,element_connectivity,w,z,boundary_elements,C,dN,Basis,Jacobian,nx,ny,XCO,YCO)
	if writeplot:
		# Write or plot data if necessary
		WritePlotBEM2D(sol,total_sol,POT,FLUX1,FLUX2,LHS2LHS,LHS2RHS,mesh,0,1,0,0)


	rel_err = 0
	if computeerror:
		# rel_err = ComputeErrorNorms(global_coord,total_sol,1,internal_points,POT)
		rel_err = Pr.ComputeErrorNorms(global_coord,total_sol)


	return rel_err, boundary_elements.shape[0],total_sol.shape[0], z.shape[0]



