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
############################################################################################################################################
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
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Misc/ProblemData.py')
Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Tests/ProblemData.py')

#############################################################################################################################################
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

	# np.savetxt('/home/roman/Desktop/elements_i2rae2822_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_i2rae2822_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_i2rae2822_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_mech2d_seg0_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_mech2d_seg0_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_mech2d_seg0_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_mech2d_seg2_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_mech2d_seg2_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_mech2d_seg2_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_mech2dn_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_mech2dn_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_mech2dn_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_leftpartwithcircle_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_leftpartwithcircle_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_leftpartwithcircle_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_leftcircle_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_leftcircle_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_leftcircle_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/elements_rae2822_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_rae2822_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_rae2822_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Desktop/unique_edges_rae2822_p'+str(MainData.C+1)+'.dat', np.unique(mesh.edges),fmt='%d',delimiter=',')

	# 3D
	# np.savetxt(MainData.Path.Problem+'/elements_sphere_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/points_sphere_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/edges_sphere_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/faces_sphere_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/faces_cube_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')

	# np.savetxt(MainData.Path.Problem+'/elements_sphere2_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/points_sphere2_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/edges_sphere2_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')
	# np.savetxt(MainData.Path.Problem+'/faces_sphere2_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/elements_sphere2_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/points_sphere2_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/faces_sphere2_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')

	np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/elements_nsphere_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/points_nsphere_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/faces_nsphere_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')


	print 'Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar
	print 'Number of mesh edge nodes', np.unique(mesh.edges).shape[0]


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
	# print mesh.points[5,:]

	# import matplotlib.pyplot as plt
	# plt.plot(mesh.points[:,0],mesh.points[:,1],'o')
	# plt.show()
	# mesh.SimplePlot()

	# print mesh.points[mesh.edges[20:24,:],:]
	# print mesh.points[mesh.edges[42:46,:],:]
	# print mesh.elements
	# print mesh.points
	# print mesh.edges
	# print mesh.faces
	# print mesh.edges.shape
	# print mesh.faces.shape
	# print mesh.points.shape
	# print mesh.elements.shape
	# print mesh.points[2,:]*1000
	# print mesh.points[2,:]
	# print mesh.points[mesh.edges[:,:2],:]
	# print mesh.points[:8,:]
	# print mesh.points[92,:]
	

	# PostProcess.HighOrderPatchPlot3D(MainData,mesh)

	# sys.exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh)
	# np.savetxt('/home/roman/Desktop/displacements.txt', TotalDisp[:,:,-1])
	# print 'Total number of DoFs for the system is', sol.shape[0]

	# print TotalDisp.shape
	# print TotalDisp[:,:,0].shape
	# print TotalDisp[1,:,0]
	# from Core.Supplementary.Tensors import makezero
	# print makezero(TotalDisp[:,:,0])
	# print 
	# print np.hstack((makezero(mesh.points),makezero(TotalDisp[:,:,0])))
	# print MainData.xx 
	# print mesh.elements[83,:]
	# print 
	# print mesh.points[mesh.elements[83,:],:]
	# print TotalDisp[2*107,:,:]
	# print TotalDisp[824,:,:]
	# print np.linalg.norm(TotalDisp[824,:,:],axis=0)
	# print np.linalg.norm(TotalDisp[:,:,0],axis=0)
	# print np.linalg.norm(TotalDisp[:,:,:],axis=1)
	# print np.linalg.norm(TotalDisp[2*107,:,:],axis=0)


	# sys.exit("STOPPED")

	# print 'Post-Processing the information...'
	# POST-PROCESS
	# PostProcess().StressRecovery(MainData,mesh,TotalDisp) 
	# print mesh.elements	
	PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp,show_plot=False)
	PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
	# PostProcess.HighOrderPatchPlot3D(MainData,mesh,TotalDisp)
	# PostProcess.HighOrderInterpolatedPatchPlot(MainData,mesh,TotalDisp)
	import matplotlib.pyplot as plt
	# plt.savefig('/home/roman/Dropbox/zdump/DumpReport/mech2d/postmesh_'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'.eps', format='eps', dpi=1000)
	plt.show()

	# from Core.Supplementary.SuppPlots.MeshNumbering import PlotMeshNumbering

	# vpoints = np.copy(mesh.points)
	# vpoints[:,0] += TotalDisp[:,0,-1]
	# vpoints[:,1] += TotalDisp[:,1,-1]
	# np.savetxt('/home/roman/Desktop/elements.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points.dat', vpoints,fmt='%6.4f',delimiter=',')

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


