import os, sys

def SetPath(Pr,pwd,C,mesh,Analysis,AnalysisType, MaterialArgsType):
	
	# STORE PATHS FOR MAIN, CORE & PROBLEM DIRECTORIES
	############################################################################
	class Path(object):
		"""Getting directory paths"""

		TopLevel = pwd
		Main = pwd+'/Main/FiniteElements'
		Problem = os.path.dirname(Pr.__file__)
		Core = pwd+'/Core/FiniteElements'

	Path = Path


	if os.path.isdir(Path.Problem+'/Results'):
		print 'Writing results in the problem directory:', Path.Problem
	else:
		print 'Writing the results in problem directory:', Path.Problem
		os.mkdir(MainData.Path.Problem+'/Results')


	Path.ProblemResults = Path.Problem+'/Results/'
	Path.ProblemResultsFileNameMATLAB = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(C)+'.mat'
	# FOR NON-LINEAR ANALYSIS - DO NOT ADD THE EXTENSION
	Path.ProblemResultsFileNameVTK = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(C)
	# FOR LINEAR ANALYSIS
	# MainData.Path.ProblemResultsFileNameVTK = 'Results_h'+str(mesh.elements.shape[0])+'_C'+str(C)+'.vtu'

	# CONSIDERATION OF MATERAIL MODEL
	Path.MaterialModel = MaterialArgsType + '_Model/'

	# ANALYSIS SPECIFIC DIRECTORIES
	if Analysis == 'Static':
		if AnalysisType == 'Linear':
			Path.Analysis = 'LinearStatic/'		# ONE STEP/INCREMENT
		# MainData.Path.LinearDynamic = 'LinearDynamic'
		elif AnalysisType == 'Nonlinear':
			Path.Analysis = 'NonlinearStatic/' 	# MANY INCREMENTS
		# Subdirectories
		if os.path.isdir(Path.ProblemResults+Path.Analysis):
			if not os.path.isdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel):
				os.mkdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel)
		else:
			os.mkdir(Path.ProblemResults+Path.Analysis)
			if not os.path.isdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel):
				os.mkdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel)

	elif Analysis == 'Dynamic':
		Path.Analysis = 'NonlinearDynamic/'
		# SUBDIRECTORIES
		if os.path.isdir(MainData.Path.ProblemResults+Path.Analysis):
			if not os.path.isdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel):
				os.mkdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel)
		else:
			os.mkdir(MainData.Path.ProblemResults+MainData.Path.Analysis)
			if not os.path.isdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel):
				os.mkdir(Path.ProblemResults+Path.Analysis+Path.MaterialModel)


	return Path