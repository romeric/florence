import numpy as np 
import numpy.linalg as la
from Core.FiniteElements.ElementalMatrices.KinematicMeasures import *


def ComputeErrorNorms(MainData,mesh,nmesh,AnalyticalSolution,Domain,Quadrature,MaterialArgs):

	# AT THE MOMENT THE ERROR NORMS ARE COMPUTED FOR LINEAR PROBLEMS ONLY
	if MainData.GeometryUpdate:
		raise ValueError('The error norms are computed for linear problems only.')

	# INITIATE/ALLOCATE
	C = MainData.C 
	ndim = MainData.ndim
	nvar = MainData.nvar
	elements = nmesh.elements
	points = nmesh.points
	nelem = elements.shape[0]
	nodeperelem = elements.shape[1]
	w = Quadrature.weights

	# TotalDisp & TotalPot ARE BOTH THIRD ORDER TENSOR (3RD DIMENSION FOR INCREMENTS) - TRANCATE THEM UNLESS REQUIRED
	TotalDisp = MainData.TotalDisp[:,:,-1]
	TotalPot  = MainData.TotalPot[:,:,-1]

	# # print TotalDisp
	# TotalDispa = np.zeros(TotalDisp.shape)
	# uxa = (points[:,1]*np.cos(points[:,0])); uxa=np.zeros(uxa.shape)
	# uya = (points[:,0]*np.sin(points[:,1]))
	# TotalDispa[:,0]=uxa
	# TotalDispa[:,1]=uya
	# # print TotalDispa
	# # print TotalDisp
	# print np.concatenate((TotalDispa,TotalDisp),axis=1)
	# # print points


	# ALLOCATE
	B = np.zeros((Domain.Bases.shape[0]*nvar,MaterialArgs.H_VoigtSize))
	E_nom = 0; E_denom = 0; L2_nom = 0; L2_denom = 0
	# LOOP OVER ELEMENTS
	for elem in range(0,nelem):
		xycoord = points[elements[elem,:],:]	
		# GET THE NUMERICAL SOLUTION WITHIN THE ELEMENT (AT NODES)
		ElementalSol = np.zeros((nodeperelem,nvar))
		ElementalSol[:,:ndim] = TotalDisp[elements[elem,:],:]
		ElementalSol[:,ndim]  = TotalPot[elements[elem,:],:].reshape(nodeperelem)
		# GET THE ANALYTICAL SOLUTION WITHIN THE ELEMENT (AT NODES)
		AnalyticalSolution.Args.node = xycoord
		AnalyticalSol = AnalyticalSolution().Get(AnalyticalSolution.Args)

		# AnalyticalSol[:,0] = ElementalSol[:,0]
		# print np.concatenate((AnalyticalSol[:,:2], ElementalSol[:,:2]),axis=1)
		# print

		# print points
		# print points[elements[elem,:],:]
		# print AnalyticalSol

		# ALLOCATE
		nvarBasis = np.zeros((Domain.Bases.shape[0],nvar))
		ElementalSolGauss  = np.zeros((Domain.AllGauss.shape[0],nvar)); AnalyticalSolGauss  = np.copy(ElementalSolGauss)
		dElementalSolGauss = np.zeros((Domain.AllGauss.shape[0],nvar)); dAnalyticalSolGauss = np.copy(ElementalSolGauss)
		# LOOP OVER GAUSS POINTS
		for counter in range(0,Domain.AllGauss.shape[0]):
			# GET THE NUMERICAL SOLUTION WITHIN THE ELEMENT (AT QUADRATURE POINTS)
			ElementalSolGauss[counter,:] = np.dot(Domain.Bases[:,counter].reshape(1,nodeperelem),ElementalSol)
			# GET THE ANALYTICAL SOLUTION WITHIN THE ELEMENT (AT QUADRATURE POINTS)
			AnalyticalSolution.Args.node = np.dot(Domain.Bases[:,counter],xycoord)
			AnalyticalSolGauss[counter,:] = AnalyticalSolution().Get(AnalyticalSolution.Args)

			# print AnalyticalSolGauss, ElementalSolGauss[:,0]
			# AnalyticalSolGauss[:,0] = ElementalSolGauss[:,0] # REMOVE
			# AnalyticalSolGauss[:,1] = ElementalSolGauss[:,1] # REMOVE
			# print np.concatenate((AnalyticalSolGauss[:,:2], ElementalSolGauss[:,:2]),axis=1)#, AnalyticalSolution.Args.node
			# print 
			# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
			Jm = Domain.Jm[:,:,counter]
			# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
			ParentGradientX=np.dot(Jm,xycoord) #
			# MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
			MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
			# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
			# F = np.dot(EulerELemCoords.T,MaterialGradient.T)
			F = np.eye(ndim,ndim)
			# COMPUTE REMAINING KINEMATIC MEASURES
			StrainTensors = KinematicMeasures(F).Compute()
			# UPDATE/NO-UPDATE GEOMETRY
			if MainData.GeometryUpdate:
				# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
				ParentGradientx=np.dot(Jm,EulerELemCoords)
				# SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
				SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
			else:
				SpatialGradient = MaterialGradient.T

			# SPATIAL ELECTRIC FIELD
			ElectricFieldx = - np.dot(SpatialGradient.T,ElementalSol[:,ndim])
			# COMPUTE SPATIAL ELECTRIC DISPLACEMENT
			ElectricDisplacementx = MainData.ElectricDisplacementx(MaterialArgs,StrainTensors,ElectricFieldx)
			# COMPUTE CAUCHY STRESS TENSOR
			CauchyStressTensor = MainData.CauchyStress(MaterialArgs,StrainTensors,ElectricFieldx)
			# COMPUTE THE HESSIAN AT THIS GAUSS POINT
			H_Voigt = MainData.Hessian(MaterialArgs,ndim,StrainTensors,ElectricFieldx)
			# COMPUTE THE TANGENT STIFFNESS MATRIX
			BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(Domain,B,MaterialGradient,nvar,SpatialGradient,ndim,CauchyStressTensor,ElectricDisplacementx,MaterialArgs,H_Voigt)

			# L2 NORM
			L2_nom   += np.linalg.norm((AnalyticalSolGauss - ElementalSolGauss)**2)*Domain.AllGauss[counter,0]
			L2_denom += np.linalg.norm((AnalyticalSolGauss)**2)*Domain.AllGauss[counter,0]

			# ENERGY NORM
			DiffSol = (AnalyticalSol - ElementalSol).reshape(ElementalSol.shape[0]*ElementalSol.shape[1],1)
			E_nom   += np.linalg.norm(DiffSol**2)*Domain.AllGauss[counter,0]
			E_denom += np.linalg.norm(AnalyticalSol.reshape(AnalyticalSol.shape[0]*AnalyticalSol.shape[1],1)**2)*Domain.AllGauss[counter,0]

	L2Norm = np.sqrt(L2_nom)/np.sqrt(L2_denom)
	EnergyNorm = np.sqrt(E_nom)/np.sqrt(E_denom)
	print L2Norm, EnergyNorm
	return L2Norm, EnergyNorm