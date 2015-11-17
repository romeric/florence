import os

# CORE IMPORTS
# from Core.FiniteElements.ComputeErrorNorms import ComputeErrorNorms
from Core.FiniteElements.PreProcess import PreProcess
from Core.FiniteElements.PostProcess import *
from Core.FiniteElements.Solvers.Solver import *


###########################################################################################################
# PROBLEM FILE DIRECTORIES
# import Examples.FiniteElements.Nonlinear_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.MultiPhysics_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.MultiPhysics_Fibre_3D.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Ellipse_Cylinder.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Cube.ProblemData as Pr
# import Examples.FiniteElements.Nonlinear_Electromechanics_3D_Beam.ProblemData as Pr

# import Examples.FiniteElements.Hollow_Arc_Tri.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle_Electromechanics.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle.ProblemData as Pr
# import Examples.FiniteElements.Annular_Circle_Nurbs.ProblemData as Pr
import Examples.FiniteElements.MechanicalComponent2D.ProblemData as Pr
# import Examples.FiniteElements.Wing2D.ProblemData as Pr
# import Examples.FiniteElements.Sphere.ProblemData as Pr
# import Examples.FiniteElements.Naca_Isotropic.ProblemData as Pr
# import Examples.FiniteElements.RAE2822.ProblemData as Pr
# import Examples.FiniteElements.Misc.ProblemData as Pr
# import Examples.FiniteElements.Tests.ProblemData as Pr

###########################################################################################################


def main(MainData, DictOutput=None, nStep=0):

	# GET THE CURRENT DIRECTORY PARTH
	pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))

	# READ PROBLEM DATA FILE
	Pr.ProblemData(MainData)
	
	# PRE-PROCESS
	print 'Pre-processing the information. Getting paths, solution parameters, mesh info, interpolation bases etc...'
	mesh = PreProcess(MainData,Pr,pwd)


	# np.savetxt('/home/roman/Desktop/elements_rae2822_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Desktop/points_rae2822_p'+str(MainData.C+1)+'.dat', 1000*mesh.points,fmt='%6.4f',delimiter=',')
	# np.savetxt('/home/roman/Desktop/edges_rae2822_p'+str(MainData.C+1)+'.dat', mesh.edges,fmt='%d',delimiter=',')

	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/elements_sphere2_p'+str(MainData.C+1)+'.dat', mesh.elements,fmt='%d',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/points_sphere2_p'+str(MainData.C+1)+'.dat', mesh.points,fmt='%10.9f',delimiter=',')
	# np.savetxt('/home/roman/Dropbox/Matlab_Files/tetplots/faces_sphere2_p'+str(MainData.C+1)+'.dat', mesh.faces,fmt='%d',delimiter=',')


	print 'Number of nodes is',mesh.points.shape[0], 'number of DoFs', mesh.points.shape[0]*MainData.nvar
	print 'Number of elements is', mesh.elements.shape[0], \
			 'and number of mesh edge nodes is', np.unique(mesh.edges).shape[0]

	# exit("STOPPED")
	# CALL THE MAIN ROUTINE
	TotalDisp = MainSolver(MainData,mesh)

	# exit("STOPPED")
	# print 'Post-Processing the information...'
	# POST-PROCESS
	# PostProcess().StressRecovery(MainData,mesh,TotalDisp) 

	# from scipy.io import loadmat
	# pp = loadmat('/home/roman/Desktop/ToFromRogelio/Load_increment_20corr.mat')
	# TotalDisp[:,:,-1] = pp['p'] - mesh.points


	if nStep ==1:
		MainData.mesh = mesh
		MainData.mesh.points = mesh.points + TotalDisp[:,:MainData.ndim,-1]
	
	#------------------------------------------------------------------------

	if MainData.AssemblyParameters.FailedToConverge==False:
		if MainData.MaterialArgs.Type != 'IncrementalLinearElastic':
			PostProcess().MeshQualityMeasures(MainData,mesh,TotalDisp,show_plot=False)
		# PostProcess.HighOrderPatchPlot(MainData,mesh,TotalDisp)
		# import matplotlib.pyplot as plt
		# plt.show()
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


	#---------------------------------------------------------------------------------

	# Compute Error Norms
	# L2Norm=0; EnergyNorm=0
	# L2Norm, EnergyNorm = ComputeErrorNorms(MainData,mesh)

	#----------------------------------------------------------------------------------

	# DEGUGGING 
	if MainData.__NO_DEBUG__ is False:
		# NOTE THAT PYTHON'S BUILT-IN DEBUGGER IS ALWAYS TRUE __debug__ WITHOUT -0 FLAG
		from Core import debug 
		debug(MainData,mesh,TotalDisp)
