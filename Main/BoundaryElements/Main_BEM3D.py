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
Pr = imp.load_source('Cube_3D','/home/roman/Dropbox/Python/Problems/BoundaryElements/Cube_3D/ProblemData.py')



########################################################################################################################################################
# User Imports
from Core.NumericalIntegration import GaussQuadrature
from Core.BoundaryElements.PreProcessBEM2D import GenerateCoordinates, CoordsJacobianRadiusatGaussPoints, CoordsJacobianRadiusatGaussPoints_LM
from Core.BoundaryElements.GetBases import GetBases
from Core.BoundaryElements.Assembly import AssemblyBEM2D, AssemblyBEM2D_Sparse
from Core.BoundaryElements.Sort_BEM import Sort_BEM
# from Core.BoundaryElements.PostProcessBEM2D import InteriorPostProcess, GetTotalSolution
# from Core.BoundaryElements.WritePlotBEM2D import WritePlotBEM2D


def main(general_data,postprocess=0,writeplot=0,computeerror=0):

	general_data, mesh_info, BoundaryData = Pr.ProblemData(general_data)
	z,w = GaussQuadrature(general_data.C+general_data.norder)
	print 2