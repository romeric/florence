# import numpy as np 
import imp, os
import numpy.linalg as la


pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
TwoD = imp.load_source('QuadLagrangeGaussLobatto',pwd+'/Core/InterpolationFunctions/TwoDimensional/Quad/QuadLagrangeGaussLobatto.py')
ThreeD = imp.load_source('HexLagrangeGaussLobatto',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Hexahedral/HexLagrangeGaussLobatto.py')
# Modal
# Tet = imp.load_source('hpModalTet',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Tetrahedral/hpModal.py')
# Nodal
Tri = imp.load_source('hpNodalTri',pwd+'/Core/InterpolationFunctions/TwoDimensional/Tri/hpNodal.py')
Tet = imp.load_source('hpNodalTet',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Tetrahedral/hpNodal.py')


from ElementalMatrices.KinematicMeasures import *
from Core.MeshGeneration import vtk_writer

class PostProcess(object):
	"""docstring for PostProcess"""


	def TotalComponentSol(self,MainData,sol,columns_in,columns_out,AppliedDirichletInc,Iter,fsize):

		nvar = MainData.nvar
		ndim = MainData.ndim
		Analysis = MainData.Analysis

		if MainData.AnalysisType == 'Nonlinear':
			if Analysis =='Static':
				# Get Total solution
				TotalSol = np.zeros((fsize,1))
				for i in range(0,columns_in.shape[0]):
					TotalSol[columns_in[i]] = sol[i]
				if Iter==0:
					for i in range(0,columns_out.shape[0]):
						TotalSol[columns_out[i]] = AppliedDirichletInc[i]
			if Analysis !='Static':
				TotalSol = np.copy(sol)
				for i in range(0,columns_out.shape[0]):
					TotalSol[columns_out[i]] = AppliedDirichletInc[i]
		elif MainData.AnalysisType == 'Linear':
				# Get Total solution
				TotalSol = np.zeros((fsize,1))
				for i in range(0,columns_in.shape[0]):
					TotalSol[columns_in[i]] = sol[i]
				for i in range(0,columns_out.shape[0]):
					TotalSol[columns_out[i]] = AppliedDirichletInc[i]
				

		# RE-ORDER SOLUTION COMPONENTS
		dU = np.zeros((TotalSol.shape[0]/nvar,nvar),dtype=np.float64)
		for invar in range(0,nvar):
			dU[:,invar] = TotalSol[invar:TotalSol.shape[0]:nvar,0]

		return dU




	def StressRecovery(self,MainData,mesh,TotalDisp):

		# KEEP THE IMPORTS LOCAL AS MATPLOTLIB IMPORT IS SLOW
		import matplotlib.pyplot as plt
		from scipy import io 

		C = MainData.C
		nvar = MainData.nvar
		ndim = MainData.ndim
		LoadIncrement = MainData.AssemblyParameters.LoadIncrements
		w = MainData.Quadrature.weights
		z = MainData.Quadrature.points

		ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]; gBasisz=[]
		if mesh.element_type=='hex':
			ns = (C+2)**ndim
			# GET THE BASES AT NODES INSTEAD OF GAUSS POINTS
			Basis = np.zeros((ns,w.shape[0]**ndim))
			gBasisx = np.zeros((ns,w.shape[0]**ndim))
			gBasisy = np.zeros((ns,w.shape[0]**ndim))
			gBasisz = np.zeros((ns,w.shape[0]**ndim))
		elif mesh.element_type=='tet':
			p=C+1
			ns = (p+1)*(p+2)*(p+3)/6
			# GET BASES AT NODES INSTEAD OF GAUSS POINTS
			# BE CAREFUL TAHT 4 STANDS FOR 4 VERTEX NODES (FOR HIGHER C CHECK THIS)
			Basis = np.zeros((ns,4))
			gBasisx = np.zeros((ns,4))
			gBasisy = np.zeros((ns,4))
			gBasisz = np.zeros((ns,4))
		elif mesh.element_type =='tri':
			p=C+1
			ns = (p+1)*(p+2)/2
			# GET BASES AT NODES INSTEAD OF GAUSS POINTS
			# BE CAREFUL TAHT 3 STANDS FOR 3 VERTEX NODES (FOR HIGHER C CHECK THIS)
			Basis = np.zeros((ns,3))
			gBasisx = np.zeros((ns,3))
			gBasisy = np.zeros((ns,3))


		eps=[]
		if mesh.element_type == 'hex':
			counter = 0
			eps = ThreeD.LagrangeGaussLobatto(C,0,0,0)[1]
			for i in range(0,eps.shape[0]):
				ndummy = ThreeD.LagrangeGaussLobatto(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)[0]
				Basis[:,counter] = ndummy[:,0]
				dummy = ThreeD.GradLagrangeGaussLobatto(C,eps[i,0],eps[i,1],eps[i,2],arrange=1)
				gBasisx[:,counter] = dummy[:,0]
				gBasisy[:,counter] = dummy[:,1]
				gBasisz[:,counter] = dummy[:,2]
				counter+=1
		elif mesh.element_type == 'tet':
			counter = 0
			eps = np.array([
				[-1.,-1.,-1.],
				[1.,-1.,-1.],
				[-1.,1.,-1.],
				[-1.,-1.,1.]
				])
			for i in range(0,eps.shape[0]):
				ndummy, dummy = Tet.hpBases(C,eps[i,0],eps[i,1],eps[i,2],1,1)
				ndummy = ndummy.reshape(ndummy.shape[0],1)
				Basis[:,counter] = ndummy[:,0]
				gBasisx[:,counter] = dummy[:,0]
				gBasisy[:,counter] = dummy[:,1]
				gBasisz[:,counter] = dummy[:,2]
				counter+=1
		elif mesh.element_type == 'tri':
			eps = np.array([
				[-1.,-1.],
				[1.,-1.],
				[-1.,1.]
				])
			for i in range(0,eps.shape[0]):
				ndummy, dummy = Tri.hpBases(C,eps[i,0],eps[i,1],1,1)
				ndummy = ndummy.reshape(ndummy.shape[0],1)
				Basis[:,i] = ndummy[:,0]
				gBasisx[:,i] = dummy[:,0]
				gBasisy[:,i] = dummy[:,1]

		# elements = mesh.elements[:,:eps.shape[0]]
		# points = mesh.points[:np.max(elements),:]
		# TotalDisp = TotalDisp[:np.max(elements),:,:]  # BECAREFUL TOTALDISP IS CHANGING HERE
		elements = mesh.elements
		points = mesh.points
		nelem = elements.shape[0]; npoint = points.shape[0]
		nodeperelem = elements.shape[1]


		# FOR AVERAGING SECONDARY VARIABLES AT NODES WE NEED TO KNOW WHICH ELEMENTS SHARE THE SAME NODES
		# GET UNIQUE NODES
		unique_nodes = np.unique(elements)
		# shared_elements = -1*np.ones((unique_nodes.shape[0],nodeperelem))
		shared_elements = -1*np.ones((unique_nodes.shape[0],50)) # This number (50) is totally arbitrary
		position = np.copy(shared_elements)
		for inode in range(0,unique_nodes.shape[0]):
			shared_elems, pos = np.where(elements==unique_nodes[inode])
			for i in range(0,shared_elems.shape[0]):
				shared_elements[inode,i] = shared_elems[i]
				position[inode,i] = pos[i]

		MainData.MainDict['CauchyStress'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
		MainData.MainDict['DeformationGradient'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
		MainData.MainDict['ElectricField'] = np.zeros((ndim,1,npoint,LoadIncrement))
		MainData.MainDict['ElectricDisplacement'] = np.zeros((ndim,1,npoint,LoadIncrement))
		MainData.MainDict['SmallStrain'] = np.zeros((ndim,ndim,npoint,LoadIncrement))
		CauchyStressTensor = np.zeros((ndim,ndim,nodeperelem,nelem))
		ElectricFieldx = np.zeros((ndim,1,nodeperelem,nelem))
		ElectricDisplacementx = np.zeros((ndim,1,nodeperelem,nelem))
		F = np.zeros((ndim,ndim,nodeperelem,nelem))
		strain = np.zeros((ndim,ndim,nodeperelem,nelem))

		
		for Increment in range(0,LoadIncrement):
			# LOOP OVER ELEMENTS
			for elem in range(0,elements.shape[0]):
				# GET THE FIELDS AT THE ELEMENT LEVEL
				Eulerx = points + TotalDisp[:,:,Increment]
				LagrangeElemCoords = points[elements[elem,:],:]
				EulerElemCoords = Eulerx[elements[elem,:],:]
				# if MainData.Fields == 'ElectroMechanics':
					# ElectricPotentialElem =  MainData.TotalPot[elements[elem,:],:,Increment] 

				# LOOP OVER ELEMENTS
				for g in range(0,eps.shape[0]):
					# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
					Jm = np.zeros((ndim,MainData.Domain.Bases.shape[0]))	
					if ndim==3:
						Jm[0,:] = gBasisx[:,g]
						Jm[1,:] = gBasisy[:,g]
						Jm[2,:] = gBasisz[:,g]
					if ndim==2:
						Jm[0,:] = gBasisx[:,g]
						Jm[1,:] = gBasisy[:,g]
					# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
					ParentGradientX=np.dot(Jm,LagrangeElemCoords)
					# MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
					MaterialGradient = np.dot(la.inv(ParentGradientX).T,Jm)

					# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
					F[:,:,g,elem] = np.dot(EulerElemCoords.T,MaterialGradient.T)
					# UPDATE/NO-UPDATE GEOMETRY
					if MainData.GeometryUpdate:
						# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
						ParentGradientx=np.dot(Jm,EulerElemCoords)
						# SPATIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla (N)]
						SpatialGradient = np.dot(la.inv(ParentGradientx),Jm).T
					else:
						SpatialGradient = MaterialGradient.T

					if MainData.Fields == 'ElectroMechanics':
						# SPATIAL ELECTRIC FIELD
						ElectricFieldx[:,:,g,elem] = - np.dot(SpatialGradient.T,ElectricPotentialElem)
						# COMPUTE SPATIAL ELECTRIC DISPLACEMENT
						ElectricDisplacementx[:,:,g,elem] = MainData.ElectricDisplacementx(MaterialArgs,StrainTensors,ElectricFieldx[:,:,g,elem])

					# Compute remaining kinematic/deformation measures
					# StrainTensors = KinematicMeasures(F[:,:,g,elem])
					StrainTensors = KinematicMeasures_NonVectorised(F[:,:,g,elem],MainData.AnalysisType,MainData.Domain.AllGauss.shape[0])
					if MainData.GeometryUpdate is False:
						strain[:,:,g,elem] = StrainTensors['strain'][g]

					# COMPUTE CAUCHY STRESS TENSOR
					if MainData.Prestress:
						CauchyStressTensor[:,:,g,elem]= MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx[:,:,g,elem])[0] 
					else:
						CauchyStressTensor[:,:,g,elem] = MainData.CauchyStress(MainData.MaterialArgs,StrainTensors,ElectricFieldx[:,:,g,elem])
					


			# AVERAGE THE QUANTITIES OVER NODES
			for inode in range(0,unique_nodes.shape[0]):

				x = np.where(shared_elements[inode,:]==-1)[0]
				if x.shape[0]!=0:
					myrange=np.linspace(0,x[0],x[0]+1)
				else:
					myrange = np.linspace(0,elements.shape[1]-1,elements.shape[1])

				for j in myrange:
					MainData.MainDict['DeformationGradient'][:,:,inode,Increment] += F[:,:,position[inode,j],shared_elements[inode,j]]
					MainData.MainDict['CauchyStress'][:,:,inode,Increment] += CauchyStressTensor[:,:,position[inode,j],shared_elements[inode,j]]
					MainData.MainDict['ElectricField'][:,:,inode,Increment] += ElectricFieldx[:,:,position[inode,j],shared_elements[inode,j]]
					MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment] += ElectricDisplacementx[:,:,position[inode,j],shared_elements[inode,j]]
					if ~MainData.GeometryUpdate:
						MainData.MainDict['SmallStrain'][:,:,inode,Increment] += strain[:,:,position[inode,j],shared_elements[inode,j]] 

				MainData.MainDict['DeformationGradient'][:,:,inode,Increment] = MainData.MainDict['DeformationGradient'][:,:,inode,Increment]/(1.0*len(myrange))
				MainData.MainDict['CauchyStress'][:,:,inode,Increment] = MainData.MainDict['CauchyStress'][:,:,inode,Increment]/(1.0*len(myrange))
				MainData.MainDict['ElectricField'][:,:,inode,Increment] = MainData.MainDict['ElectricField'][:,:,inode,Increment]/(1.0*len(myrange))
				MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment] = MainData.MainDict['ElectricDisplacement'][:,:,inode,Increment]/(1.0*len(myrange))
				if ~MainData.GeometryUpdate:
						MainData.MainDict['SmallStrain'][:,:,inode,Increment] = MainData.MainDict['SmallStrain'][:,:,inode,Increment]/(1.0*len(myrange))


		# FOR PLOTTING PURPOSES ONLY COMPUTE THE EXTERIOR SOLUTION
		if C>0:
			ns_0 = np.max(elements[:,:eps.shape[0]]) + 1
			MainData.MainDict['DeformationGradient'] = MainData.MainDict['DeformationGradient'][:,:,:ns_0,:]
			MainData.MainDict['CauchyStress'] = MainData.MainDict['CauchyStress'][:,:,:ns_0,:]
			if ~MainData.GeometryUpdate:
				MainData.MainDict['SmallStrain'] = MainData.MainDict['SmallStrain'][:,:,:ns_0,:]
			if MainData.Fields == 'ElectroMechanics':
				MainData.MainDict['ElectricField'] = MainData.MainDict['ElectricField'][:,:,:ns_0,:]
				MainData.MainDict['ElectricDisplacement'] = MainData.MainDict['ElectricDisplacement'][:,:,:ns_0,:]


		# NEWTON-RAPHSON CONVERGENCE PLOT
		if MainData.nrplot[0]:
			if MainData.nrplot[1] == 'first':
				plt.semilogy(MainData.NRConvergence['Increment_0'],'-ko')		# First increment convergence
			elif MainData.nrplot[1] == 'last':
				plt.plot(np.log10(MainData.NRConvergence['Increment_'+str(len(MainData.NRConvergence)-1)]),'-ko') 	# Last increment convergence
			else:
				plt.plot(np.log10(MainData.NRConvergence['Increment_'+str(MainData.nrplot[1])]),'-ko') 	# Arbitrary increment convergence

			axis_font = {'size':'18'}
			plt.xlabel(r'$No\, of\, Iteration$', **axis_font)
			plt.ylabel(r'$log_{10}|Residual|$', **axis_font)
			# Save plot
			# plt.savefig(MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+'/NR_convergence_'+MaterialArgs.Type+'.eps',
				# format='eps', dpi=1000)
			# Display plot
			plt.show()


		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#
		# Save the dictionary in .mat file
		# MainData.MainDict['Solution'] = MainData.TotalDisp
		# Write to Matlab .mat dictionary
		# io.savemat(MainData.Path.ProblemResults+MainData.Path.ProblemResultsFileNameMATLAB,MainData.MainDict)
		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#


		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#
		#------------------------------------------------------------------------------------------------------------------------------------------#

		# WRITE IN VTK FILE 
		if MainData.write:
			if mesh.element_type =='tri':
				cellflag = 5
			elif mesh.element_type =='quad':
				cellflag = 9
			if mesh.element_type =='tet':
				cellflag = 10
			elif mesh.element_type == 'hex':
				cellflag = 12
			for incr in range(0,MainData.AssemblyParameters.LoadIncrements):
				# PLOTTING ON THE DEFORMED MESH
				elements = mesh.elements[:,:eps.shape[0]]
				points = mesh.points[:np.max(elements)+1,:]
				TotalDisp = TotalDisp[:np.max(elements)+1,:,:] # BECAREFUL TOTALDISP IS CHANGING HERE
				points[:,:nvar] += TotalDisp[:,:nvar,incr]

				# OLDER APPROACH
				# points[:,0] += TotalDisp[:,0,incr] 
				# points[:,1] += TotalDisp[:,1,incr] 
				# if ndim==3:
					# points[:,2] += MainData.TotalDisp[:,2,incr] 

				# WRITE DISPLACEMENTS
				vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=TotalDisp[:,:,incr],
				fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_U_'+str(incr)+'.vtu')
				
				
				if MainData.Fields == 'ElectroMechanics':
					# WRITE ELECTRIC POTENTIAL
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.TotalPot[:,:,incr],
					fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_Phi_'+str(incr)+'.vtu')

			# FOR LINEAR STATIC ANALYSIS
			# vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, pdata=MainData.TotalDisp[:,:,incr], fname=MainData.Path.ProblemResults+'/Results.vtu')


			for incr in range(0,MainData.AssemblyParameters.LoadIncrements):
				# PLOTTING ON THE DEFORMED MESH
				elements = mesh.elements[:,:eps.shape[0]]
				points = mesh.points[:np.max(elements)+1,:]
				TotalDisp = TotalDisp[:np.max(elements)+1,:,:] # BECAREFUL TOTALDISP IS CHANGING HERE
				points[:,:nvar] += TotalDisp[:,:nvar,incr]
				npoint = points.shape[0]

				#---------------------------------------------------------------------------------------------------------------------------------------#
				# CAUCHY STRESS
				vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['CauchyStress'][:,0,:,incr].reshape(npoint,ndim),
					fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_S_i0_'+str(incr)+'.vtu')

				vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['CauchyStress'][:,1,:,incr].reshape(npoint,ndim),
					fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_S_i1_'+str(incr)+'.vtu')

				if ndim==3:
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['CauchyStress'][:,2,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_S_i2_'+str(incr)+'.vtu')
				#---------------------------------------------------------------------------------------------------------------------------------------#


				#---------------------------------------------------------------------------------------------------------------------------------------#
				if MainData.Fields == 'ElectroMechanics':
					# ELECTRIC FIELD
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['ElectricField'][:,0,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_E_'+str(incr)+'.vtu')
					#---------------------------------------------------------------------------------------------------------------------------------------#
					#---------------------------------------------------------------------------------------------------------------------------------------#
					# ELECTRIC DISPLACEMENT
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['ElectricDisplacement'][:,0,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_D_'+str(incr)+'.vtu')
				#---------------------------------------------------------------------------------------------------------------------------------------#


				#---------------------------------------------------------------------------------------------------------------------------------------#
				# STRAIN/KINEMATICS
				if ~MainData.GeometryUpdate:
					# SMALL STRAINS
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['SmallStrain'][:,0,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_Strain_i0_'+str(incr)+'.vtu')

					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['SmallStrain'][:,1,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_Strain_i1_'+str(incr)+'.vtu')

					if ndim==3:
						vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['SmallStrain'][:,2,:,incr].reshape(npoint,ndim),
							fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_Strain_i2_'+str(incr)+'.vtu')
				else:
					# DEFORMATION GRADIENT
					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['DeformationGradient'][:,0,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_F_i0_'+str(incr)+'.vtu')

					vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['DeformationGradient'][:,1,:,incr].reshape(npoint,ndim),
						fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_F_i1_'+str(incr)+'.vtu')

					if ndim==3:
						vtk_writer.write_vtu(Verts=points, Cells={cellflag:elements}, pdata=MainData.MainDict['DeformationGradient'][:,2,:,incr].reshape(npoint,ndim),
							fname=MainData.Path.ProblemResults+MainData.Path.Analysis+MainData.Path.MaterialModel+MainData.Path.ProblemResultsFileNameVTK+'_F_i2_'+str(incr)+'.vtu')
				#---------------------------------------------------------------------------------------------------------------------------------------#



	def MeshQualityMeasures(self,MainData,nmesh,TotalDisp):

		import matplotlib.pyplot as plt

		Domain = MainData.Domain
		points = nmesh.points
		vpoints = np.copy(nmesh.points)
		# vpoints	+= TotalDisp[:,:,-1]
		vpoints	= vpoints+ TotalDisp[:,:,-1]

		elements = nmesh.elements

		MainData.ScaledJacobian = np.zeros(elements.shape[0])
		# MainData.ScaledJacobian = []
		# MainData.ScaledJacobianElem = []

		JMax =[]; JMin=[]
		for elem in range(0,elements.shape[0]):
			LagrangeElemCoords = points[elements[elem,:],:]
			EulerElemCoords = vpoints[elements[elem,:],:]
			for counter in range(0,Domain.AllGauss.shape[0]):
				# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
				Jm = Domain.Jm[:,:,counter]
				# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
				ParentGradientX=np.dot(Jm,LagrangeElemCoords) #
				# MAPPING TENSOR [\partial\vec{x}/ \partial\vec{\varepsilon} (ndim x ndim)]
				ParentGradientx=np.dot(Jm,EulerElemCoords) #

				# MATERIAL GRADIENT TENSOR IN PHYSICAL ELEMENT [\nabla_0 (N)]
				MaterialGradient = np.dot(la.inv(ParentGradientX),Jm)
				# DEFORMATION GRADIENT TENSOR [\vec{x} \otimes \nabla_0 (N)]
				F = np.dot(EulerElemCoords.T,MaterialGradient.T)

				detF = np.abs(np.linalg.det(F))
				H = detF*np.linalg.inv(F).T

				# Jacobian = np.sqrt(np.einsum('ij,ij',H,H))
				# Jacobian = np.einsum('ij,ij',H,H)
				# Jacobian = np.einsum('ij,ij',F,F)
				# Jacobian = np.sqrt(np.einsum('ij,ij',F,F))
				
				# Jacobian = np.sqrt(np.sqrt(np.sqrt(np.einsum('ij,ij',F,F))))

				# from scipy.special import cbrt
				# Jacobian = cbrt(np.einsum('ij,ij',F,F))

				Jacobian = np.abs(np.linalg.det(ParentGradientx))
				# Jacobian = np.abs(np.linalg.det(F))
				# if elem==71:
					# print np.abs(np.linalg.det(ParentGradientx)), np.abs(np.linalg.det(ParentGradientX))
				
				if counter==0:
					JMin=Jacobian; JMax = Jacobian

				if Jacobian > JMax:
					JMax = Jacobian
				if Jacobian < JMin:
					JMin = Jacobian
				# print Jacobian
				# pass
			if np.allclose(1.0,1.0*JMin/JMax,rtol=1e-3):
				pass
			else:
				pass
				# print 1.0*JMin/JMax
				# MainData.ScaledJacobian[elem] = 1.0*JMin/JMax
				# MainData.ScaledJacobian = np.append(MainData.ScaledJacobian,1.0*JMin/JMax)
				# MainData.ScaledJacobianElem = np.append(MainData.ScaledJacobianElem,elem)	

			MainData.ScaledJacobian[elem] = 1.0*JMin/JMax
			# print JMin, JMax
		# print MainData.ScaledJacobian.shape, np.unique(elements).shape


		fig = plt.figure()
		# # plt.bar(np.linspace(0,elements.shape[0]-1,elements.shape[0]),MainData.ScaledJacobian,width=1.,color='#FE6F5E',alpha=0.8)

		plt.bar(np.linspace(0,elements.shape[0]-1,elements.shape[0]),MainData.ScaledJacobian,width=1.,alpha=0.4)
		plt.xlabel(r'$Elements$',fontsize=18)
		plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)



		# plt.bar(np.linspace(0,MainData.ScaledJacobianElem.shape[0]-1,MainData.ScaledJacobianElem.shape[0]),MainData.ScaledJacobian,width=1.,alpha=0.4)
		# plt.xlabel(r'$Elements$',fontsize=18)
		# plt.ylabel(r'$Scaled\, Jacobian$',fontsize=18)




		# plt.savefig('/home/roman/Desktop/DumpReport/scaled_jacobian_312_'+MainData.AnalysisType+'_p'+str(MainData.C)+'.eps', format='eps', dpi=1000)

	@staticmethod	
	def HighOrderPatch(MainData,mesh,TotalDisp):

		import matplotlib.pyplot as plt
		
		plt.figure()
		# print TotalDisp[:,0,-1]
		vpoints = np.copy(mesh.points)
		vpoints[:,0] += TotalDisp[:,0,-1]
		vpoints[:,1] += TotalDisp[:,1,-1]
		# plt.plot(vpoints[:,0],vpoints[:,1],'o',color='#ffffee') 
		plt.plot(vpoints[:,0],vpoints[:,1],'o',color='#F88379') 

		dum1=[]; dum2=[]; dum3 = []; ddum=np.array([0,1,2,0])
		for i in range(0,MainData.C):
			dum1=np.append(dum1,i+3)
			dum2 = np.append(dum2, 2*MainData.C+3 +i*MainData.C -i*(i-1)/2 )
			dum3 = np.append(dum3,MainData.C+3 +i*(MainData.C+1) -i*(i-1)/2 )

		if MainData.C>0:
			ddum = (np.append(np.append(np.append(np.append(np.append(np.append(0,dum1),1),dum2),2),np.fliplr(dum3.reshape(1,dum3.shape[0]))),0) ).astype(np.int32)

		for i in range(0,mesh.elements.shape[0]):
			dum = vpoints[mesh.elements[i,:],:]
			plt.plot(dum[ddum,0],dum[ddum,1],alpha=0.02)
			# plt.fill(dum[ddum,0],dum[ddum,1],'#A4DDED')
			plt.fill(dum[ddum,0],dum[ddum,1],color=(0.75,MainData.ScaledJacobian[i],0.35))	
			# plt.fill(dum[ddum,0],dum[ddum,1],color=(0.75,1.0*i/mesh.elements.shape[0],0.35))	
			# plt.fill(dum[ddum,0],dum[ddum,1],color=(MainData.ScaledJacobian[i],0,1-MainData.ScaledJacobian[i]))	

			plt.plot(dum[ddum,0],dum[ddum,1],'#000000')

		plt.axis('equal')
		# plt.axis('off')	


		# plt.savefig('/home/roman/Desktop/DumpReport/mesh_312_'+MainData.AnalysisType+'_p'+str(MainData.C)+'.eps', format='eps', dpi=1000)

