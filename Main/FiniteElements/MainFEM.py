from time import time
import numpy as np
import numpy.linalg as la
import os, sys, imp
# GET THE CURRENT DIRECTORY PARTH
pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

# CORE IMPORTS
from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Core.FiniteElements.PostProcess import *
from Core.FiniteElementslverslver import *
from Core.FiniteElements.PreProcess import PreProcess

#######################################################################################################################################################
# PROBLEM FILE DIRECTORIES
# Pr = imp.load_source('Square_Piezo',pwd+'/Problems/FiniteElements/MultiPhysics_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D_Cube',pwd+'/Problems/FiniteElements/Nonlinear_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Square_Piezo',pwd+'/Problems/FiniteElements/MultiPhysics_Fibre_3D/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Ellipse_Cylinder/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Beam/ProblemData.py')
# 2D
# Pr = imp.load_source('Nonlinear_2D',pwd+'/Problems/FiniteElements/Hollow_Arc_Tri/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle_Electromechanics/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle/ProblemData.py')
Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle_Nurbs/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/MechanicalComponent2D/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Sphere/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Naca_Isotropic/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/RAE2822/ProblemData.py')

########################################################################################################################################################


def main(MainData):
	# READ PROBLEM DATA FILE
	Pr.ProblemData(MainData)
	# print 'The Problem is',MainData.ndim,'Dimensional'
	# PRE-PROCESS
	print 'Pre-Processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...'
	mesh = PreProcess(MainData,Pr,pwd)

	# print mesh.points
	# print mesh.elements 
	# print mesh.points.shape
	# print mesh.edges

	# np.savetxt('/home/roman/Desktop/elements.txt', mesh.elements)
	# np.savetxt('/home/roman/Desktop/points.txt', mesh.points)
	np.savetxt('/home/roman/Desktop/edges_circle.dat', mesh.edges[:,:2],fmt='%d',delimiter=',')

	print 'Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar
	
	# sys.exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh)
	# np.savetxt('/home/roman/Desktop/displacements.txt', TotalDisp[:,:,-1])
	# print 'Total number of DoFs for the system is', sol.shape[0]

	# print 'Post-Processing the information...'
	# POST-PROCESS
	# PostProcess().StressRecovery(MainData,mesh,TotalDisp) 
	# PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp)
	# PostProcess.HighOrderPatch(MainData,mesh,TotalDisp)
	# plt.show()
	# # plt.savefig('/home/roman/Desktop/DumpReport/uniform_aniso_mesh_'+MainData.MaterialArgs.Type+'_p'+str(MainData.C)+'.eps', format='eps', dpi=1000)

	# from Core.Supplementary.SuppPlots.MeshNumbering import PlotMeshNumbering
	# PlotMeshNumbering(mesh)


	# Compute Error Norms
	# L2Norm=0; EnergyNorm=0
	# L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

	if MainData.__NO_DEBUG__ is False:
		_DEBUG(MainData,mesh,TotalDisp)
		# CHECK GAUSS POINTS
		# print np.sum(Domain.Bases,axis=0)
		# print np.sum(Domain.gBasesx,axis=0)
		# print np.sum(Domain.gBasesy,axis=0)
		# print np.sum(Quadrature.weights)

	# sys.exit("STOPPED")


