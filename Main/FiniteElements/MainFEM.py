# from time import time
# import numpy as np
# import numpy.linalg as la
import os, sys, imp
# GET THE CURRENT DIRECTORY PARTH
pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Core.FiniteElements.PreProcess import PreProcess
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Solvers.Solver import *

# from Core.InterpolationFunctions.TwoDimensional.Tri.hpNodalLagrange import hpBasesLagrange
# from Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal import hpBases
# from Core.Supplementary.Tensors import makezero
# from Core.InterpolationFunctions.DegenerateMappings import MapXiEta2RS

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
Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/MechanicalComponent2D/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Wing2D/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Sphere/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Naca_Isotropic/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/RAE2822/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Misc/ProblemData.py')
# Pr = imp.load_source('ProblemData',pwd+'/Problems/FiniteElements/Tests/ProblemData.py')

#############################################################################################################################################
def main(MainData, DictOutput=None):


	# READ PROBLEM DATA FILE
	Pr.ProblemData(MainData)
	# print 'The Problem is',MainData.ndim,'Dimensional'
	# PRE-PROCESS
	print 'Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...'
	mesh = PreProcess(MainData,Pr,pwd)


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

	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/elements_nsphere_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/points_nsphere_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/faces_nsphere_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/elements_wing2d_p'+str(MainData.C+1)+'.dat', 
	# 	mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/points_wing2d_p'+str(MainData.C+1)+'.dat', 
	# 	mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Florence/Problems/FiniteElements/Wing2D/edges_wing2d_p'+str(MainData.C+1)+'.dat', 
	# 	mesh.edges,fmt='%d',delimiter=',')



	print 'Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar
	print 'Number of mesh edge nodes', np.unique(mesh.edges).shape[0]


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

	# sys.exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh)
	# np.savetxt('/home/roman/Desktop/displacements.txt', TotalDisp[:,:,-1])
	# print 'Total number of DoFs for the system is', sol.shape[0]

	# print TotalDisp.shape
	# print TotalDisp[:,:,0].shape
	# from Core.Supplementary.Tensors import makezero
	# print makezero(TotalDisp[:,:,0])

	# print np.linalg.norm(TotalDisp[824,:,:],axis=0)
	# print np.linalg.norm(TotalDisp[:,:,0],axis=0)
	# print np.linalg.norm(TotalDisp[:,:,:],axis=1)
	# print np.linalg.norm(TotalDisp[2*107,:,:],axis=0)


	# sys.exit("STOPPED")

	# print 'Post-Processing the information...'
	# POST-PROCESS
	# PostProcess().StressRecovery(MainData,mesh,TotalDisp) 
	# PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp,show_plot=False)
	# PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
	# PostProcess.HighOrderPatchPlot3D(MainData,mesh,TotalDisp)
	# PostProcess.HighOrderInterpolatedPatchPlot(MainData,mesh,TotalDisp)
	# import matplotlib.pyplot as plt
	# plt.savefig('/home/roman/Dropbox/zdump/DumpReport/mech2d/postmesh_'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'.eps', 
		# format='eps', dpi=1000)
	# plt.savefig('/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/initial_plots/mech2d_planarmesh_'+\
		# MainData.MaterialArgs.Type+'_p'+str(MainData.C+1)+'.eps',format='eps', dpi=1000)
	# plt.savefig('/home/roman/Dropbox/Repository/LaTeX/2015_HighOrderMeshing/initial_plots/mech2d_curvedmesh_'+\
		# MainData.MaterialArgs.Type+'_p'+str(MainData.C+1)+'.eps',	format='eps', dpi=1000)

	# plt.show()

	#------------------------------------------------------------------------

	if MainData.AssemblyParameters.FailedToConverge==False:
		PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp,show_plot=False)
		PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
		import matplotlib.pyplot as plt
		plt.show()
	else:
		MainData.ScaledJacobian = np.NAN

	if DictOutput is not None:
		DictOutput['MeshPoints_P'+str(MainData.C+1)] = mesh.points
		DictOutput['MeshElements_P'+str(MainData.C+1)] = mesh.elements+1
		DictOutput['MeshEdges_P'+str(MainData.C+1)] = mesh.edges+1
		if MainData.ndim==3:
			DictOutput['MeshFaces_P'+str(MainData.C+1)] = mesh.faces+1
		DictOutput['TotalDisplacement_P'+str(MainData.C+1)] = TotalDisp
		DictOutput['nSteps'] = MainData.AssemblyParameters.LoadIncrements
		# print MainData.MaterialArgs.Type


	#-------------------------------------------------------------------------------------------------------------
	# vpoints = np.copy(mesh.points)
	# vpoints[:,0] += TotalDisp[:,0,-1]
	# vpoints[:,1] += TotalDisp[:,1,-1]
	# if MainData.ndim==3:
	# 	vpoints[:,2] += TotalDisp[:,2,-1]

	# print TotalDisp[:,:,-1]
	# mesh.GetInteriorEdgesTri()

	# np.savetxt('/home/roman/Desktop/MeshingElasticity2/'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'_elements.dat',
	# 	mesh.elements,fmt='%d',delimiter=',')	
	# np.savetxt('/home/roman/Desktop/MeshingElasticity2/'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'_planar_points.dat',
	# 	mesh.points,fmt='%10.9f',delimiter=',')	
	# np.savetxt('/home/roman/Desktop/MeshingElasticity2/'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'_displacement.dat',
	# 	TotalDisp[:,:,-1],fmt='%10.9f',delimiter=',')

	# savetxt('/home/roman/Desktop/MeshingElasticity2/'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'Xf.dat')
	# savetxt('/home/roman/Desktop/MeshingElasticity2/'+MainData.AnalysisType+'_p'+str(MainData.C+1)+'Xf.dat')



	#-------------------------------------------------------------------------------------------------------------

	# Compute Error Norms
	# L2Norm=0; EnergyNorm=0
	# L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

	# DEGUGGING 
	if MainData.__NO_DEBUG__ is False:
		# NOTE THAT PYTHON'S BUILT-IN DEBUGGER IS ALWAYS TRUE __debug__ WITHOUT -0 FLAG
		_DEBUG(MainData,mesh,TotalDisp)
		mesh_node_order = mesh.CheckNodeNumberingTri()

		# CHECK GAUSS POINTS
		# print np.sum(Domain.Bases,axis=0)
		# print np.sum(Domain.gBasesx,axis=0)
		# print np.sum(Domain.gBasesy,axis=0)
		# print np.sum(Quadrature.weights)

	# sys.exit("STOPPED")


