######################################################################################################################################################
# Builtin Imports
from time import time
import numpy as np
import scipy as sp
import numpy.linalg as la
import scipy.linalg as sla 
import scipy.io as io 
import os, sys, imp

# from vtk import*
# from vtk.util.numpy_support import *
# from pyevtk.hl import gridToVTK


# Get the current folder path
pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))



#######################################################################################################################################################
# Problem Imports
# Pr = imp.load_source('Square_Piezo',pwd+'/Problems/FiniteElements/MultiPhysics_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D_Cube',pwd+'/Problems/FiniteElements/Nonlinear_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Square_Piezo',pwd+'/Problems/FiniteElements/MultiPhysics_Fibre_3D/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Cube/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Ellipse_Cylinder/ProblemData.py')
# Pr = imp.load_source('Nonlinear_3D',pwd+'/Problems/FiniteElements/Nonlinear_Electromechanics_3D_Beam/ProblemData.py')
# 2D
# Pr = imp.load_source('Nonlinear_2D',pwd+'/Problems/FiniteElements/Hollow_Arc_Tri/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle_Electromechanics/ProblemData.py')
Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle/ProblemData.py')

########################################################################################################################################################
# User Imports
from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Solvers.Solver import *
from Core.FiniteElements.PreProcess import PreProcess



def main(MainData):
	# READ PROBLEM DATA FILE
	Pr.ProblemData(MainData)
	# print 'The Problem is',MainData.ndim,'Dimensional'
	# PRE-PROCESS
	print 'Pre-Processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...'
	mesh, nmesh = PreProcess(MainData,Pr,pwd)

	# print nmesh.points
	# print nmesh.elements 
	# print nmesh.points.shape[0]
	# print mesh.faces
	# print mesh.edges
	# print Quadrature.weights
	# print Quadrature.points
	# print Domain.Bases 
	# print Domain.gBasesx
	# print Domain.gBasesy
	# print mesh.elements
	# print mesh.points
	# print mesh.edges
	# print nmesh.edges

	# print nmesh.points.shape

	# print np.sum(Domain.Bases,axis=0)
	# print np.sum(Domain.gBasesx,axis=0)
	# print np.sum(Domain.gBasesy,axis=0)
	# print np.sum(Quadrature.weights)

	# np.savetxt('/home/roman/Desktop/elements.txt', nmesh.elements)
	# np.savetxt('/home/roman/Desktop/points.txt', nmesh.points)
	# np.savetxt('/home/roman/Desktop/edges.txt', nmesh.edges)

	print 'Number of nodes is',nmesh.points.shape[0], 'number of DoFs', nmesh.points.shape[0]*MainData.nvar
	
	# sys.exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh,nmesh)
	# print 'Total number of DoFs for the system is', sol.shape[0]

	# print 'Post-Processing the information...'
	# Post Process
	# PostProcess().StressRecovery(MainData,mesh,nmesh,Quadrature)
	# PostProcess().MeshQualityMeasures(MainData,mesh,nmesh,TotalDisp,Quadrature)

	
	# Compute Error Norms
	# L2Norm=0; EnergyNorm=0
	# L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh,nmesh,AnalyticalSolution,Domain,Quadrature,MaterialArgs)


	# t=np.linspace(0,2*np.pi,300)
	# x=0.5*np.cos(t)
	# y=0.5*np.sin(t)
	# plt.plot(x,y)
	# -------------------------------------------------------------------------------

	# # -------------------------------------------------------------------------------
	# plt.figure()
	# vpoints = np.copy(nmesh.points)
	# vpoints[:,0] += TotalDisp[:,0,-1]
	# vpoints[:,1] += TotalDisp[:,1,-1]
	# # plt.plot(vpoints[:,0],vpoints[:,1],'o',color='#ffffee') 
	# plt.plot(vpoints[:,0],vpoints[:,1],'o',color='#F88379') 

	# dum1=[]; dum2=[]; dum3 = []; ddum=np.array([0,1,2,0])
	# for i in range(0,MainData.C):
	# 	dum1=np.append(dum1,i+3)
	# 	dum2 = np.append(dum2, 2*MainData.C+3 +i*MainData.C -i*(i-1)/2 )
	# 	dum3 = np.append(dum3,MainData.C+3 +i*(MainData.C+1) -i*(i-1)/2 )

	# if MainData.C>0:
	# 	ddum = (np.append(np.append(np.append(np.append(np.append(np.append(0,dum1),1),dum2),2),np.fliplr(dum3.reshape(1,dum3.shape[0]))),0) ).astype(np.int32)

	# for i in range(0,nmesh.elements.shape[0]):
	# 	dum = vpoints[nmesh.elements[i,:],:]
	# 	# plt.plot(dum[ddum,0],dum[ddum,1])
	# 	plt.fill(dum[ddum,0],dum[ddum,1],'#A4DDED')
	# # -------------------------------------------------------------------------------

	# from Core.Supplementary.SuppPlots.MeshNumbering import PlotMeshNumbering
	# PlotMeshNumbering(mesh)

	# plt.axis('equal')
	# plt.axis('off')
	# plt.show()

	# plt.savefig('/home/roman/Desktop/DumpReport/mesh_312_'+MainData.AnalysisType+'_p'+str(MainData.C)+'.eps', format='eps', dpi=1000)

	# sys.exit("STOPPED")
