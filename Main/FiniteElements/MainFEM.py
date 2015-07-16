# from time import time
# import numpy as np
# import numpy.linalg as la
import os, sys, imp
# GET THE CURRENT DIRECTORY PARTH
pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
# from mpi4py import MPI

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
# from time import time
# t_import=time()
from Core.FiniteElements.PreProcess import PreProcess
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Solvers.Solver import *
# print 'TIME',time()-t_import
# sys.exit(0)
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
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Annular_Circle_Nurbs/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/MechanicalComponent2D/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Sphere/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Naca_Isotropic/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/RAE2822/ProblemData.py')
Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Misc/ProblemData.py')

########################################################################################################################################################
# from line_profiler import profile
# @profile
def main(MainData):

	# READ PROBLEM DATA FILE
	Pr.ProblemData(MainData)
	# print 'The Problem is',MainData.ndim,'Dimensional'
	# PRE-PROCESS
	print 'Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...'
	mesh = PreProcess(MainData,Pr,pwd)

	# print mesh.points
	# print mesh.elements
	# print mesh.edges 
	# print mesh.points.shape
	# print mesh.edges.shape
	# print np.where(mesh.elements==19)

	# np.savetxt('/home/roman/Desktop/elements.txt', mesh.elements)
	# np.savetxt('/home/roman/Desktop/points.txt', mesh.points)
	# np.savetxt('/home/roman/Desktop/edges_circle.dat', mesh.edges[:,:2],fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_rae2822_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_rae2822_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_rae2822_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')
	
	# np.savetxt('/home/roman/Desktop/elements_circle_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_circle_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_circle_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_half_circle_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_half_circle_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_half_circle_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_twoarcs_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_twoarcs_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_twoarcs_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/unique_edges_rae2822_p'+str(MainData.C+1)+'.dat', np.unique(mesh.edges),fmt='%d',delimiter=',')

	print 'Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar
	

	# import Core.Supplementary.nurbs.cad as iga 
	# circle = iga.circle(radius=1, center=None, angle=None)
	# print circle.array


	# print edge_elements
	# print np.asarray(x)
	# print mesh.edges

	# boundary_node_arrangement = np.zeros((mesh.edges.shape[0],mesh.edges.shape[1]),dtype=np.int64)
	# for i in range(mesh.edges.shape[0]):
	# 	co = mesh.points[mesh.edges[i,:],:]
	# 	right_length = np.linalg.norm(co[1:-1,:] - co[0,:],axis=1)
	# 	# left_length = np.linalg.norm(co[1:-1,:] - co[-1,:],axis=1)
	# 	left_length = np.linalg.norm(co[2:,:] - co[1,:],axis=1)
	# 	# print left_length, right_length
	# 	length_difference = left_length - right_length
	# 	# print (length_difference < 0).all()
	# 	# print length_difference
	# 	# print left_length
	# # 	# if i==0:
	# # 	# 	print co[1:-1,:]
	# # 	# 	print
	# # 	# 	print co[0,:]
	# # 	# 	print 
	# # 	# 	print co[1:-1,:] - co[0,:]
	# # 	# 	print 
	# # 	# 	print right_length
	# 	if i>17:
	# 		# print left_length
	# 		print np.argsort(left_length)+1
			# print np.argsort(right_length)+1



	# x = mesh.points[mesh.elements[0,:],:]
	# y = x[[0,3,4,1,7,9,2,8,5,0],:2]
	# import matplotlib.pyplot as plt
	# plt.plot(x[:,0],x[:,1],'-ro')
	# plt.plot(y[:,0],y[:,1],'-')
	# plt.show()

	# print mesh.edges
	# x = mesh.points[mesh.edges[19,:],:]
	# print x
	# plt.plot(x[:,0],x[:,1],'-ro')
	# plt.show()
	# print mesh.elements.flags
	# print mesh.points[mesh.edges[22,:],:]
	# print mesh.edges[18:23,]
	print mesh.points[9,:]

	# import matplotlib.pyplot as plt
	# plt.plot(mesh.points[:,0],mesh.points[:,1],'o')
	# plt.show()

	sys.exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh)
	# np.savetxt('/home/roman/Desktop/displacements.txt', TotalDisp[:,:,-1])
	# print 'Total number of DoFs for the system is', sol.shape[0]

	# print 'Post-Processing the information...'
	# POST-PROCESS
	# PostProcess().StressRecovery(MainData,mesh,TotalDisp) 	
	PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp)
	PostProcess.HighOrderPatch(MainData,mesh,TotalDisp)
	import matplotlib.pyplot as plt
	plt.show()
	# # plt.savefig('/home/roman/Desktop/DumpReport/uniform_aniso_mesh_'+MainData.MaterialArgs.Type+'_p'+str(MainData.C)+'.eps', format='eps', dpi=1000)

	# from Core.Supplementary.SuppPlots.MeshNumbering import PlotMeshNumbering


	# Compute Error Norms
	# L2Norm=0; EnergyNorm=0
	# L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

	if MainData.__NO_DEBUG__ is False:
		# NOTE THAT PYTHON'S BUILT-IN DEBUGGER IS ALWAYS TRUE __debug__ WITHOUT -0 FLAG
		_DEBUG(MainData,mesh,TotalDisp)
		mesh_node_order = mesh.CheckNodeNumberingTri()
		if mesh_node_order == 'anti-clockwise':
			print u'\u2713'.encode('utf8')+' : ','Imported mesh has',mesh_node_order,'node ordering'
		else:
			print u'\u2717'.encode('utf8')+' : ','Imported mesh has',mesh_node_order,'node ordering'

		# CHECK GAUSS POINTS
		# print np.sum(Domain.Bases,axis=0)
		# print np.sum(Domain.gBasesx,axis=0)
		# print np.sum(Domain.gBasesy,axis=0)
		# print np.sum(Quadrature.weights)

	# sys.exit("STOPPED")


