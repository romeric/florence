import numpy as np 
import numpy.linalg as la
from KinematicMeasures import *


def Stiffness(MainData,LagrangeElemCoords,EulerELemCoords,ElectricPotentialElem,elem):

	nvar = MainData.nvar
	ndim = MainData.ndim
	# w = MainData.Quadrature.weights

	# ALLOCATE
	stiffness = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.Domain.Bases.shape[0]*nvar))
	tractionforce = np.zeros((MainData.Domain.Bases.shape[0]*nvar,1))
	B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize))
	# Volume = 0

	#--------------------------------------------------------------------------------------------------------#
	# ParentGradientX = np.einsum('ijk,jl',MainData.Domain.Jm,LagrangeElemCoords) #
	# # print MainData.Domain.Jm.shape
	# # print la.inv(np.einsum('jik',ParentGradientX)).shape
	# MaterialGradient1 = np.einsum('ijk,klm',la.inv(np.einsum('jik',ParentGradientX)),MainData.Domain.Jm)
	# MaterialGradient1 = np.einsum('jk,klm',la.inv(np.einsum('jik',ParentGradientX))[2,:,:],MainData.Domain.Jm) #
	# MaterialGradient1 = np.einsum('jk,klm',la.inv(ParentGradientX[:,0,:]),MainData.Domain.Jm) #
	# MaterialGradient1 = np.einsum('ijk,kl',la.inv(np.einsum('jik',ParentGradientX)),MainData.Domain.Jm[:,:,1])
	# print MaterialGradient1.shape
	# # print MainData.Domain.Jm.shape, MainData.Domain.Jm.T.shape
	# # print ParentGradientX.shape, MainData.Domain.AllGauss.shape[0]
	# F = np.dot(EulerELemCoords.T,MaterialGradient.T)
	# # COMPUTE REMAINING KINEMATIC MEASURES
	# StrainTensors = KinematicMeasures(F).Compute(MainData.AnalysisType)
	# SpatialGradient = MaterialGradient.T
	# ElectricFieldx, ElectricDisplacementx =[],[]
	# CauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,0)
	# # COMPUTE THE HESSIAN AT THIS GAUSS POINT
	# H_Voigt = MainData.Hessian(MainData.MaterialArgs,ndim,StrainTensors,ElectricFieldx,elem,0)
	# # COMPUTE THE TANGENT STIFFNESS MATRIX
	# BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt)
	# # detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
	# # detJ = MainData.Domain.AllGauss[:,0]*np.abs(la.det(np.einsum('jik',ParentGradientX)))
	# detJ = np.einsum('i,i',MainData.Domain.AllGauss[:,0],np.abs(la.det(np.einsum('jik',ParentGradientX))) )
	# # print detJ
	# stiffness = BDB_1*detJ
	# # tractionforce = t*detJ
	# # print tractionforce

	# COMPUTE ParentGradientX AND MATERIALGRADIENTX ONCE FOR ALL GAUSS POINTS
	# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
	ParentGradientX = np.dot(MainData.Domain.Jm[:,:,0],LagrangeElemCoords)
	# MaterialGradient1 = np.dot(la.inv(ParentGradientX),MainData.Domain.Jm[:,:,0])
	MaterialGradient = np.einsum('jk,klm',la.inv(ParentGradientX),MainData.Domain.Jm)
	# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
	F = np.einsum('ij,kil',EulerELemCoords,MaterialGradient)

	# UPDATE/NO-UPDATE GEOMETRY
	if MainData.GeometryUpdate:
		ParentGradientx = np.einsum('ijk,jl->kil',MainData.Domain.Jm,EulerELemCoords)
		# print ParentGradientx.shape, MainData.Domain.Jm.shape
		# SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
		SpatialGradient1 = np.einsum('ijk,kli->ijl',la.inv(ParentGradientx),MainData.Domain.Jm)
	else:
		SpatialGradient = np.einsum('jik',MaterialGradient)

	#--------------------------------------------------------------------------------------------------------#

	# LOOP OVER GAUSS POINTS
	# for counter in range(0,MainData.Domain.AllGauss.shape[0]):
	for counter in xrange(MainData.Domain.AllGauss.shape[0]): 
		# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
		Jm = MainData.Domain.Jm[:,:,counter]
		# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
		ParentGradientX = np.dot(Jm,LagrangeElemCoords) #
		# MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
		MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
		# mm = np.linalg.norm(MaterialGradient - np.einsum('jki',MaterialGradient1)[counter,:,:])
		# mm = np.linalg.norm(MaterialGradient - MaterialGradient1[:,:,counter])
		# mm = np.linalg.norm(MaterialGradient - MaterialGradient1)
		# if abs(mm) >1e-12:
			# print 'wow'
		# MaterialGradient = la.solve(ParentGradientX,Jm) # A TINY BIT SLOWER
		# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
		F = np.dot(EulerELemCoords.T,MaterialGradient.T)
		# COMPUTE REMAINING KINEMATIC MEASURES
		# StrainTensors = KinematicMeasures(F[:,:,counter]).Compute(MainData.AnalysisType)
		StrainTensors = KinematicMeasures(F).Compute(MainData.AnalysisType)

		# UPDATE/NO-UPDATE GEOMETRY
		if MainData.GeometryUpdate:
		# if MainData.GeometryUpdate or MainData.Prestress:
			# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
			ParentGradientx=np.dot(Jm,EulerELemCoords)
			# SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
			SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
		else:
			SpatialGradient = MaterialGradient.T

		if MainData.Fields == 'ElectroMechanics':
			# MATERIAL ELECTRIC FIELD  
			# ElectricFieldX = - np.dot(ElectricPotentialElem.T,MaterialGradient.T)
			# SPATIAL ELECTRIC FIELD
			ElectricFieldx = - np.dot(SpatialGradient.T,ElectricPotentialElem)
			# COMPUTE SPATIAL ELECTRIC DISPLACEMENT
			ElectricDisplacementx = MainData.ElectricDisplacementx(MainData.MaterialArgs,StrainTensors,ElectricFieldx)
		else:
			ElectricFieldx, ElectricDisplacementx = [],[]

		# COMPUTE CAUCHY STRESS TENSOR
		CauchyStressTensor = []
		if MainData.AnalysisType == 'Nonlinear':
			CauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)
		elif MainData.Prestress:
			CauchyStressTensor, LastCauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)

		# COMPUTE THE HESSIAN AT THIS GAUSS POINT
		H_Voigt = MainData.Hessian(MainData.MaterialArgs,ndim,StrainTensors,ElectricFieldx,elem,counter)
		# COMPUTE THE TANGENT STIFFNESS MATRIX
		# BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,SpatialGradient[:,:,counter],CauchyStressTensor,ElectricDisplacementx,H_Voigt)
		BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt)


		if MainData.GeometryUpdate:
			# COMPUTE GEOMETRIC STIFFNESS MATRIX
			# BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient[:,:,counter],CauchyStressTensor,nvar,ndim)
			BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,CauchyStressTensor,nvar,ndim)
			# COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
			detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
			# INTEGRATE STIFFNESS
			stiffness += (BDB_1+BDB_2)*detJ
			# INTEGRATE TRACTION FORCE
			tractionforce += t*detJ
		else:
			# COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
			detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
			# detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
			if MainData.Prestress:
				# COMPUTE GEOMETRIC STIFFNESS MATRIX
				# BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
				# BDB_1 += BDB_2
				# BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient[:,:,counter],LastCauchyStressTensor,nvar,ndim)
				BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
				# BDB_1 = BDB_1 + BDB_2
			# INTEGRATE STIFFNESS
			stiffness += (BDB_1)*detJ
			if MainData.AnalysisType == 'Nonlinear' or MainData.Prestress:
				# INTEGRATE TRACTION FORCE
				tractionforce += t*detJ


		if elem==0:
			# print ParentGradientX
			# print ParentGradientx
			# print MaterialGradient
			print SpatialGradient
			# print F  
			print
			# print Jm.T
			pass
	if elem==0:
		print
		print np.einsum('kji',SpatialGradient1)
		# print np.einsum('jki',MaterialGradient1)
		# print np.einsum('jik',ParentGradientX1)
		# print np.einsum('jki',F1)
		pass




	# # CHECK FOR SYMMETRY OF STIFFNESS MATRIX
	# for i in range(0,stiffness.shape[0]):
	# 	for j in range(0,stiffness.shape[0]):
	# 		if ~np.allclose(stiffness[i,j],stiffness[j,i]):
	# 			print i,j


	return stiffness, tractionforce 









# def Stiffness(MainData,LagrangeElemCoords,EulerELemCoords,ElectricPotentialElem,elem):

	# nvar = MainData.nvar
	# ndim = MainData.ndim
	# # w = MainData.Quadrature.weights

	# # ALLOCATE
	# stiffness = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.Domain.Bases.shape[0]*nvar))
	# tractionforce = np.zeros((MainData.Domain.Bases.shape[0]*nvar,1))
	# B = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.MaterialArgs.H_VoigtSize))
	# # Volume = 0

	# # LOOP OVER GAUSS POINTS
	# # for counter in range(0,MainData.Domain.AllGauss.shape[0]):
	# for counter in xrange(MainData.Domain.AllGauss.shape[0]): 
	# 	# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
	# 	Jm = MainData.Domain.Jm[:,:,counter]
	# 	# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
	# 	ParentGradientX=np.dot(Jm,LagrangeElemCoords) #
	# 	# MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
	# 	MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
	# 	# MaterialGradient = la.solve(ParentGradientX,Jm) # A TINY BIT SLOWER
	# 	# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
	# 	F = np.dot(EulerELemCoords.T,MaterialGradient.T)
	# 	# COMPUTE REMAINING KINEMATIC MEASURES
	# 	StrainTensors = KinematicMeasures(F).Compute(MainData.AnalysisType)

	# 	# UPDATE/NO-UPDATE GEOMETRY
	# 	if MainData.GeometryUpdate:
	# 	# if MainData.GeometryUpdate or MainData.Prestress:
	# 		# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
	# 		ParentGradientx=np.dot(Jm,EulerELemCoords)
	# 		# SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
	# 		SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
	# 	else:
	# 		SpatialGradient = MaterialGradient.T

	# 	if MainData.Fields == 'ElectroMechanics':
	# 		# MATERIAL ELECTRIC FIELD  
	# 		# ElectricFieldX = - np.dot(ElectricPotentialElem.T,MaterialGradient.T)
	# 		# SPATIAL ELECTRIC FIELD
	# 		ElectricFieldx = - np.dot(SpatialGradient.T,ElectricPotentialElem)
	# 		# COMPUTE SPATIAL ELECTRIC DISPLACEMENT
	# 		ElectricDisplacementx = MainData.ElectricDisplacementx(MainData.MaterialArgs,StrainTensors,ElectricFieldx)
	# 	else:
	# 		ElectricFieldx, ElectricDisplacementx = [],[]

	# 	# COMPUTE CAUCHY STRESS TENSOR
	# 	CauchyStressTensor = []
	# 	if MainData.AnalysisType == 'Nonlinear':
	# 		CauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)
	# 	elif MainData.Prestress:
	# 		CauchyStressTensor, LastCauchyStressTensor = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx,elem,counter)

	# 	# COMPUTE THE HESSIAN AT THIS GAUSS POINT
	# 	H_Voigt = MainData.Hessian(MainData.MaterialArgs,ndim,StrainTensors,ElectricFieldx,elem,counter)
	# 	# COMPUTE THE TANGENT STIFFNESS MATRIX
	# 	BDB_1, t = MainData().ConstitutiveStiffnessIntegrand(B,nvar,ndim,MainData.AnalysisType,MainData.Prestress,SpatialGradient,CauchyStressTensor,ElectricDisplacementx,H_Voigt)


	# 	if MainData.GeometryUpdate:
	# 		# COMPUTE GEOMETRIC STIFFNESS MATRIX
	# 		BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,CauchyStressTensor,nvar,ndim)
	# 		# COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
	# 		detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
	# 		# INTEGRATE STIFFNESS
	# 		stiffness += (BDB_1+BDB_2)*detJ
	# 		# INTEGRATE TRACTION FORCE
	# 		tractionforce += t*detJ
	# 	else:
	# 		# COMPUTE ONCE detJ (GOOD SPEEDUP COMPARED TO COMPUTING TWICE)
	# 		detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
	# 		# detJ = MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
	# 		if MainData.Prestress:
	# 			# COMPUTE GEOMETRIC STIFFNESS MATRIX
	# 			# BDB_2 = MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
	# 			# BDB_1 += BDB_2
	# 			BDB_1 += MainData().GeometricStiffnessIntegrand(SpatialGradient,LastCauchyStressTensor,nvar,ndim)
	# 			# BDB_1 = BDB_1 + BDB_2
	# 		# INTEGRATE STIFFNESS
	# 		stiffness += (BDB_1)*detJ
	# 		if MainData.AnalysisType == 'Nonlinear' or MainData.Prestress:
	# 			# INTEGRATE TRACTION FORCE
	# 			tractionforce += t*detJ






	# # # CHECK FOR SYMMETRY OF STIFFNESS MATRIX
	# # for i in range(0,stiffness.shape[0]):
	# # 	for j in range(0,stiffness.shape[0]):
	# # 		if ~np.allclose(stiffness[i,j],stiffness[j,i]):
	# # 			print i,j


	# return stiffness, tractionforce 