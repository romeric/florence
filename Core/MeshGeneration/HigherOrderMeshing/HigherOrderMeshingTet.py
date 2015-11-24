import numpy as np 
# from scipy.stats import itemfreq
from time import time
import multiprocessing as MP

import Core.ParallelProcessing.parmap as parmap
import GetInteriorCoordinates as Gett
import Core.InterpolationFunctions.TwoDimensional.Quad.QuadLagrangeGaussLobatto as TwoD 
import Core.InterpolationFunctions.ThreeDimensional.Tetrahedral.hpNodal as Tet 
from Core.QuadratureRules.FeketePointsTet import *
from Core.Supplementary.Where import *
from Core.Supplementary.Tensors import itemfreq_py, makezero, duplicate
from TwoLoopNode_Cython import TwoLoopNode_Cython


#--------------------------------------------------------------------------------------------------------------------------#
def FindDuplicatesXYZ(i,Xs,sorted_repoints,Decimals):#
	
	dups = None
	duplicates_list = None 

	# IF THE MULITPLICITY OF A GIVEN X-VALUE IS 1 THEN IGNORE
	if Xs[i,1]!=1:
		# IF THE MULTIPLICITY IS MORE THAN 1, THEN FIND WHERE ALL IN THE SORTED ARRAY THIS X-VALUE OCCURS
		dups = np.arange(np.sum(np.int64(Xs[:i,1])),np.sum(np.int64(Xs[:i+1,1])))
		# FIND THE Y & Z-COORDINATE VALUES OF THESE MULTIPLICITIES 
		Ys = sorted_repoints[dups,:][:,1:]
		# CALL THE DUPLICATE FUNCTION TO FIND DUPLICATED ROWS IN YZ-COLUMNS
		duplicates_list = duplicate(Ys,decimals=Decimals)


	return duplicates_list, dups

def TwoLoopNode(duplicates_list,duplicates,dups0,counter):
	for j in range(len(duplicates_list)):
		for k in range(1,duplicates_list[j].shape[0]):
			duplicates[counter,:] = dups0+np.array([duplicates_list[j][0],duplicates_list[j][k]])
			counter +=1
	return duplicates, counter


def LoopDuplicates(i,reelements,duplicates):
	return whereEQ(reelements,duplicates[i,1])
	# return np.where(reelements==duplicates[i,1])

def LoopArangeDuplicates(i,duplicates,Ds):
	x=None; y=None; x1=None; y1=None; mini=None
	if Ds[i,1]!=1:
		# x,y = np.where(duplicates==Ds[i,0])
		x,y = whereEQ(duplicates,Ds[i,0])
		mini = np.min(duplicates[x,:])
		# x1,y1 = np.where(duplicates==mini)
		x1,y1 = whereEQ(duplicates,mini)

	return x,y,x1,y1,mini




def HighOrderMeshTet_UNSTABLE(C,mesh,Decimals=10,Zerofy=0,Parallel=False,nCPU=1,ComputeAll=False):

	# SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
	if (mesh.elements.shape[0] < 500) and (C < 6):
		Parallel = False
		nCPU = 1

	eps = FeketePointsTet(C)

	# COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
	Neval = np.zeros((4,eps.shape[0]),dtype=np.float64)
	for i in range(4,eps.shape[0]):
		Neval[:,i] = Tet.hpBases(0,eps[i,0],eps[i,1],eps[i,2],Transform=1,EvalOpt=1)[0]

	nodeperelem = mesh.elements.shape[1]
	renodeperelem = int((C+2.)*(C+3.)*(C+4.)/6.)
	left_over_nodes = renodeperelem - nodeperelem

	reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
	reelements[:,:4] = mesh.elements
	# TOTAL NUMBER OF (INTERIOR+EDGE+FACE) NODES 
	# iesize = int(C*(C-1)*(C-2)/6. + 6.*C)
	iesize = np.int64(C*(C-1)*(C-2)/6. + 6.*C + 2*C*(C-1))
	repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],3),dtype=np.float64)
	repoints[:mesh.points.shape[0],:]=mesh.points

	telements = time()

	xycoord_higher=[]; ParallelTuple1=[]
	if Parallel:
		# GET HIGHER ORDER COORDINATES - PARALLEL
		ParallelTuple1 = parmap.map(ElementLoopTet,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'tet',eps,
			Neval,pool=MP.Pool(processes=nCPU))
	
	maxNode = np.max(reelements)
	for elem in range(0,mesh.elements.shape[0]):
		# maxNode = np.max(reelements) # BIG BOTTLENECK
		if Parallel:
			xycoord_higher = ParallelTuple1[elem]
		else:	
			xycoord =  mesh.points[mesh.elements[elem,:],:]
			# GET HIGHER ORDER COORDINATES
			xycoord_higher = Gett.GetInteriorNodesCoordinates(xycoord,'tet',elem,eps,Neval)

		# EXPAND THE ELEMENT CONNECTIVITY
		# reelements[elem,4:] = np.linspace(maxNode+1,maxNode+left_over_nodes,left_over_nodes).astype(int)
		newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes)
		reelements[elem,4:] = newElements
		# reelements[elem,4:] = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
		# INSTEAD COMPUTE maxNode BY INDEXING
		maxNode = newElements[-1]

		repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[4:,:]

	telements = time()-telements
	# print t0

	#--------------------------------------------------------------------------------------
	# LOOP OVER POINTS
	tnodes = time()

	tol = 1e-14
	# MAKE ARRAYS NEAR ZERO EQUAL TO ZERO - OFF BY DEFAULT
	if Zerofy:
		repoints = makezero(repoints,tol=tol)

	# TO FIND THE DUPLICATES WE NEED THE UNIQUE VALUES OF repoints WHICH IS FLOATING POINT ARRAY, SO WE FIRST
	# NEED TO ROUND OFF TO CERTAIN DECIMAL DIGIT.
	# Decimals = 10
	# SORT ONLY THE X COORDINATE OF repoints
	iSortX = np.argsort(repoints[:,0])
	# BASED ON THE SORTED X-VALUES, PUT THEIR CORRESPONDING Y-VALUES NEXT TO THEM I.E. SORT REPOINTS BASED ON X-VALUES
	sorted_repoints = repoints[iSortX,:]
	# NOW LETS FIND THE UNIQUE VALUES OF THIS SORTED FLOATING POINTS ARRAY
	# NOTE THAT FROM THE INVERSE INDICES OF A UNIQUE ARRAY WE CAN CONSTRUCT THE ACTUAL ARRAY 
	unique_repoints,invX =np.unique(np.round(sorted_repoints[:,0],decimals=Decimals),return_inverse=True)
	# NOW FIND THE MULTIPLICITY OF EACH UNIQUE X-VALUES 

	Xs = itemfreq_py(un_arr=unique_repoints,inv_arr=invX)
	# duplicates = -1*np.ones((reelements.shape[0]*reelements.shape[1],2),dtype=np.int64)
	duplicates = -1*np.ones((repoints.shape[0],2),dtype=np.int64)

	ParallelTupleNode = []
	if Parallel:
		ParallelTupleNode = parmap.map(FindDuplicatesXYZ,np.arange(0,Xs.shape[0]),Xs,sorted_repoints,Decimals,pool=MP.Pool(processes=nCPU))

	counter = 0
	# LOOP OVER POINTS
	for i in range(0,Xs.shape[0]):
		if Parallel:
			duplicates_list = ParallelTupleNode[i][0]
			dups = ParallelTupleNode[i][1]
			if duplicates_list is not None:
				if len(duplicates_list)!=0:
					# duplicates, counter = TwoLoopNode(duplicates_list,duplicates,dups[0],counter)
					duplicates, counter = TwoLoopNode_Cython(duplicates_list,duplicates,dups[0],counter)


		else:
			# IF THE MULITPLICITY OF A GIVEN X-VALUE IS 1 THEN INGONRE
			if Xs[i,1]!=1:
				# IF THE MULTIPLICITY IS MORE THAN 1, THEN FIND WHERE ALL IN THE SORTED ARRAY THIS X-VALUE OCCURS
				# dups =  np.where(i==invX)[0]
				# dups = np.asarray(whereEQ(invX.reshape(invX.shape[0],1),i)[0]) # this can be totally avoided
				dups = np.arange(np.sum(np.int64(Xs[:i,1])),np.sum(np.int64(Xs[:i+1,1])))
				# FIND THE Y & Z-COORDINATE VALUES OF THESE MULTIPLICITIES 
				Ys = sorted_repoints[dups,:][:,1:]
				# FIND YZ-DUPLICATES
				duplicates_list = duplicate(Ys,decimals=Decimals)


				# PUT TUPLE DUPLICATES INTO A NUMPY ARRAY
				if len(duplicates_list)!=0:
					# duplicates, counter = TwoLoopNode(duplicates_list,duplicates,dups[0],counter)
					duplicates, counter = TwoLoopNode_Cython(duplicates_list,duplicates,dups[0],counter)
					# for j in range(len(duplicates_list)):
					# 	for k in range(1,duplicates_list[j].shape[0]):
					# 		duplicates[counter,:] = dups[0]+np.array([duplicates_list[j][0],duplicates_list[j][k]])
					# 		counter +=1


	# RE-ASSIGN DUPLICATE
	duplicates = duplicates[:counter,:]
	# BASED ON THE DUPLICATES OCCURING IN THE SORTED ARRAY sorted_repoints, FIND THE ACTUAL DUPLICATES OCCURING IN repoints
	duplicates = np.asarray([iSortX[duplicates[:,0]],iSortX[duplicates[:,1]] ]).T
	# SORT THE ACTUAL DUPLICATE ROW-WISE SO THAT THE FIRST COLUMN IS ALWAYS SMALLER THAN THE SECOND COLUMN
	# duplicates = np.sort(duplicates,axis=1)

	# t3 = time()
	unique_duplicates,invD = np.unique(duplicates,return_inverse=True)
	Ds = itemfreq_py(un_arr=unique_duplicates,inv_arr=invD)
	ParallelTupleArangeDuplicates = []
	if Parallel:
		ParallelTupleArangeDuplicates = parmap.map(LoopArangeDuplicates,np.arange(Ds.shape[0]),duplicates,Ds,pool=MP.Pool(processes=nCPU))

	for i in range(Ds.shape[0]):
		if Parallel:
			x = ParallelTupleArangeDuplicates[i][0]
			y = ParallelTupleArangeDuplicates[i][1]
			x1 = ParallelTupleArangeDuplicates[i][2]
			y1 = ParallelTupleArangeDuplicates[i][3]
			mini = ParallelTupleArangeDuplicates[i][4]
			if x is not None:
				if Ds[i,0]!=mini:
					duplicates[x,y] = mini
					duplicates[x1,y1] = Ds[i,0]
		else:
			if Ds[i,1]!=1:
				# x,y = np.where(duplicates==Ds[i,0])
				x,y = whereEQ(duplicates,Ds[i,0])
				mini = np.min(duplicates[x,:])
				x1,y1 = whereEQ(duplicates,mini)
				# x1,y1 = np.where(duplicates==mini)

				if Ds[i,0]!=mini:
					duplicates[x,y] = mini
					duplicates[x1,y1] = Ds[i,0]

	# SORT THE ACTUAL DUPLICATE ROW-WISE SO THAT THE FIRST COLUMN IS ALWAYS SMALLER THAN THE SECOND COLUMN
	duplicates = np.sort(duplicates,axis=1)
	# print time()-t3

	
	totalnodes = np.arange(0,repoints.shape[0]) 
	remainingnodes = np.delete(totalnodes,duplicates[:,1])
	mapnodes = np.arange(0,remainingnodes.shape[0]) 	
	

	# t4=time()
	ParallelTupleDuplicates=[]
	if Parallel:
		ParallelTupleDuplicates = parmap.map(LoopDuplicates,np.arange(0,duplicates.shape[0]),reelements,duplicates)

	for i in range(0,duplicates.shape[0]):
		if Parallel:
			x = ParallelTupleDuplicates[i][0]
			y = ParallelTupleDuplicates[i][1]
			reelements[x,y] = duplicates[i,0]
		else:
			# reelements[whereEQ(reelements,duplicates[i,1])] = duplicates[i,0]
			reelements[reelements==duplicates[i,1]] = duplicates[i,0]
	# print time()-t4

	Dx = whereLT(duplicates.astype(np.float64),mesh.points.shape[0])[0]
	# Dx = np.where(duplicate<mesh.points.shape[0])[0]
	if np.asarray(Dx).shape[0] != 0:
		print 'Duplicated points in the original mesh\n', duplicates[Dx,:] 
		# print itemfreq_py(duplicates[:,0]) # MULTIPLICITY OF EACH POINT
		raise ValueError('Original linear mesh has duplicated points')


	tnodes = time()-tnodes




	telements_2 = time()

	ParallelTuple3=[]
	if Parallel:
		ParallelTuple3 = parmap.map(ElementLoopTet_2,np.arange(mesh.points.shape[0],mapnodes.shape[0]),reelements,remainingnodes,pool=MP.Pool(processes=nCPU))

	for i in range(mesh.points.shape[0],mapnodes.shape[0]):
		if Parallel:
			x=ParallelTuple3[i-mesh.points.shape[0]][0]; y=ParallelTuple3[i-mesh.points.shape[0]][1]
			reelements[x,y] = i
		else:
			# reelements[whereEQ(reelements,remainingnodes[i])] = i 
			reelements[reelements==remainingnodes[i]] = i

	telements_2 = time()-telements_2

	# REPOINTS
	repoints = repoints[remainingnodes,:]
	#------------------------------------------------------------------------------------------


	# USE ALTERNATIVE APPROACH TO GET MESHES
	reedges = np.zeros((mesh.edges.shape[0],C+2))
	fsize = int((C+2.)*(C+3.)/2.)
	refaces = np.zeros((mesh.faces.shape[0],fsize))

	# ComputeAll = False
	if ComputeAll == True:

		#------------------------------------------------------------------------------------------
		# BUILD EDGES NOW
		tedges = time()

		reedges = np.zeros((mesh.edges.shape[0],C+2))
		reedges[:,:2]=mesh.edges
		for iedge in range(0,mesh.edges.shape[0]):
			# TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
			node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)
			node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)

			# FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
			L1 = np.linalg.norm(node1-repoints[mesh.points.shape[0]:,:],axis=1)
			L2 = np.linalg.norm(node2-repoints[mesh.points.shape[0]:,:],axis=1)
			L = np.linalg.norm(node1-node2,axis=1)

			# j = np.where(np.abs((L1+L2)-L) < tol)[0]
			j = np.asarray( whereLT1d(np.abs((L1+L2)-L), tol) )
			if j.shape[0]!=0:
				reedges[iedge,2:] = j+mesh.points.shape[0]
		reedges = reedges.astype(int)

		tedges = time()-tedges
		#------------------------------------------------------------------------------------------


		#------------------------------------------------------------------------------------------
		# BUILD FACES NOW 
		# A SIMILAR ANALOGY TO FACES IS FOLLOWED

		# C
		# #
		#   # 	 	if |area(ABD)+area(ADC)+area(BCD) = area(ABC)| then D lies inside the triangular face
		#     #
		#   D   #
		# A       #
		# # # # # # # B    

		# ALSO CHECK COPLANARITIES OF 4 POINTS: 	http://mathworld.wolfram.com/Coplanar.html
		# IF A POINT LIES INSIDE A TRIANGLE IN 2D:  http://mathworld.wolfram.com/TriangleInterior.html
		tfaces = time()

		fsize = int((C+2.)*(C+3.)/2.)
		refaces = np.zeros((mesh.faces.shape[0],fsize))
		refaces[:,:3] = mesh.faces
		# FIND THE UNIT NORMAL VECTOR TO THE FACE
		ParallelTuple6 = []
		if Parallel:
			ParallelTuple6 = parmap.map(LoopFaceTet,np.arange(0,mesh.faces.shape[0]),mesh.points,repoints,mesh.faces,tol,pool=MP.Pool(processes=nCPU))
		for iface in range(0,mesh.faces.shape[0]):
			if Parallel:
				j = ParallelTuple6[iface]
			else:
				# FIND THE POSITION VECTOR OF VERTICES 
				A = mesh.points[mesh.faces[iface,0],:]
				B = mesh.points[mesh.faces[iface,1],:]
				C = mesh.points[mesh.faces[iface,2],:]
				# FIND THE VECTORS OF ALL EDGES
				BA = B - A 		# VECTOR OF EDGE 1
				CA = C - A 		# VECTOR OF EDGE 2
				CB = C - B  	# VECTOR OF EDGE 3

				DA = repoints[mesh.points.shape[0]:,:] - np.repeat(A.reshape(1,3),repoints.shape[0] - mesh.points.shape[0],axis=0 )
				DB = repoints[mesh.points.shape[0]:,:] - np.repeat(B.reshape(1,3),repoints.shape[0] - mesh.points.shape[0],axis=0 )

				# COMPUTE AREAS 
				Area = 0.5*np.linalg.norm(np.cross(BA,CA)); Area = Area*np.ones(DA.shape[0])
				# Area1 = 0.5*np.linalg.norm(np.cross(DA,BA),axis=1); #Area1 = Area1*np.ones(DA.shape[0])
				Area1 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(BA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
				Area2 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(CA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
				Area3 = 0.5*np.linalg.norm(np.cross(DB, np.repeat(CB.reshape(1,3),DB.shape[0],axis=0)),axis=1 )
				# t1 +=( time()-tt)
				# CHECK THE CONDITION
				# j = np.where(np.abs( Area - (Area1+Area2+Area3) )  < tol)[0]
				j = np.asarray(whereLT1d(np.abs( Area - (Area1+Area2+Area3) ),tol)) 
				# j = np.asarray(whereLT(np.abs( Area - (Area1+Area2+Area3) ).reshape(Area.shape[0],1),tol)[0]) 
			if j.shape[0]!=0:
				if j.shape[0]!=3:
					FloatingPointError('More nodes within the tetrahedral face than necessary. \
						This could be due to high/low tolerance')
				refaces[iface,3:] = j+mesh.points.shape[0]
				
		refaces = refaces.astype(int)

		tfaces = time()- tfaces


	class nmesh(object):
		# """Construct pMesh"""
		points = repoints
		elements = reelements
		edges = np.array([[],[]])
		faces = np.array([[],[]])
		nnode = repoints.shape[0]
		nelem = reelements.shape[0]
		info = 'tet'
	if ComputeAll is True:
		nmesh.edges = reedges
		nmesh.faces = refaces


		# print '\npMeshing timing:\n\t\tElement loop 1:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
		#  ' seconds'+'\n\t\tElement loop 2:\t '+str(telements_2)+' seconds\n\t\tEdge loop:\t\t '+str(tedges)+' seconds'+\
		#  '\n\t\tFace loop:\t\t '+str(tfaces)+' seconds\n'

	return nmesh







#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
# SUPPLEMENTARY FUNCTIONS 
# OLD WAY - REPLICATING NODES WITHIN NODES LOOP
def ElementLoopTet(elem,elements,points,MeshType,eps,Neval):
	xycoord_higher = Gett.GetInteriorNodesCoordinates(points[elements[elem,:],:],MeshType,elem,eps,Neval)
	return xycoord_higher


def NodeLoopTet(inode,reelements,repoints,tol):

	difference = np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:]
	# difference = np.sqrt(difference[:,0]*difference[:,0] + difference[:,1]*difference[:,1] + difference[:,2]*difference[:,2]).reshape(inode,1)
	difference = np.sqrt(difference[:,0]*difference[:,0] + difference[:,1]*difference[:,1] + difference[:,2]*difference[:,2])
	
	# return np.asarray(whereLT(difference, tol)[0])
	return np.asarray(whereLT1d(difference, tol))


def ElementLoopTet_2(i,reelements,remainingnodes):
	return whereEQ(reelements,remainingnodes[i])
	# return np.where(reelements==remainingnodes[i])


def LoopFaceTet(iface,points,repoints,faces,tol):
	# FIND THE POSITION VECTOR OF VERTICES 
	A = points[faces[iface,0],:]
	B = points[faces[iface,1],:]
	C = points[faces[iface,2],:]
	# FIND THE VECTORS OF ALL EDGES
	BA = B - A 		# VECTOR OF EDGE 1
	CA = C - A 		# VECTOR OF EDGE 2
	CB = C - B  	# VECTOR OF EDGE 3

	DA = repoints[points.shape[0]:,:] - np.repeat(A.reshape(1,3),repoints.shape[0] - points.shape[0],axis=0 )
	DB = repoints[points.shape[0]:,:] - np.repeat(B.reshape(1,3),repoints.shape[0] - points.shape[0],axis=0 )

	# COMPUTE AREAS 
	Area = 0.5*np.linalg.norm(np.cross(BA,CA)); Area = Area*np.ones(DA.shape[0])
	Area1 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(BA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
	Area2 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(CA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
	Area3 = 0.5*np.linalg.norm(np.cross(DB, np.repeat(CB.reshape(1,3),DB.shape[0],axis=0)),axis=1 )
	# CHECK THE CONDITION
	j = np.where(np.abs( Area - (Area1+Area2+Area3) )  < tol)[0]
	# j = np.asarray(whereLT(np.abs( Area - (Area1+Area2+Area3) ).reshape(Area.shape[0],1),tol)[0]) 
	# j = np.asarray(whereLT1d(np.abs( Area - (Area1+Area2+Area3) ),tol)) 

	return j 






def HighOrderMeshTet(C,mesh,Parallel=False,nCPU=1):

	# SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
	if (mesh.elements.shape[0] < 500) and (C < 6):
		Parallel = False
		nCPU = 1

	eps = FeketePointsTet(C)

	# COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
	Neval = np.zeros((4,eps.shape[0]),dtype=np.float64)
	for i in range(4,eps.shape[0]):
		Neval[:,i] = Tet.hpBases(0,eps[i,0],eps[i,1],eps[i,2],Transform=1)[0]

	nodeperelem = mesh.elements.shape[1]
	renodeperelem = int((C+2.)*(C+3.)*(C+4.)/6.)
	left_over_nodes = renodeperelem - nodeperelem

	reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
	reelements[:,:4] = mesh.elements
	# TOTAL NUMBER OF (INTERIOR+EDGE+FACE) NODES 
	# iesize = int(C*(C-1)*(C-2)/6. + 6.*C)
	iesize = int(C*(C-1)*(C-2)/6. + 6.*C + 2*C*(C-1))
	repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],3),dtype=np.float64)
	repoints[:mesh.points.shape[0],:]=mesh.points

	telements = time()

	xycoord_higher=[]; ParallelTuple1=[]
	if Parallel:
		# GET HIGHER ORDER COORDINATES - PARALLEL
		ParallelTuple1 = parmap.map(ElementLoopTet,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,'tet',eps,
			Neval,pool=MP.Pool(processes=nCPU))

	maxNode = np.max(reelements)
	for elem in range(0,mesh.elements.shape[0]):
		# maxNode = np.max(reelements)
		if Parallel:
			xycoord_higher = ParallelTuple1[elem]
		else:	
			xycoord =  mesh.points[mesh.elements[elem,:],:]
			# GET HIGHER ORDER COORDINATES
			xycoord_higher = Gett.GetInteriorNodesCoordinates(xycoord,'tet',elem,eps,Neval)
		# EXPAND THE ELEMENT CONNECTIVITY
		# reelements[elem,4:] = np.linspace(maxNode+1,maxNode+left_over_nodes,left_over_nodes).astype(int)
		# reelements[elem,4:] = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
		newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
		reelements[elem,4:] = newElements
		maxNode = newElements[-1]
		repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[4:,:]

	telements = time()-telements

	#--------------------------------------------------------------------------------------
	# LOOP OVER POINTS

	tnodes = time()

	tol = 1e-14
	# duplicates = np.zeros((1,2))
	duplicates = -1*np.ones((reelements.shape[0]*reelements.shape[1],2),dtype=np.int64)

	ParallelTuple2=[]
	if Parallel:
		ParallelTuple2 = parmap.map(NodeLoopTet,np.arange(mesh.points.shape[0],repoints.shape[0]),reelements,repoints,tol,pool=MP.Pool(processes=nCPU))
	# marked=[]
	# for inode in range(0,repoints.shape[0]):
	counter =0
	for inode in range(mesh.points.shape[0],repoints.shape[0]): 
		if Parallel:
			j=ParallelTuple2[inode-mesh.points.shape[0]]
		else:	
			# difference =  np.linalg.norm( np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:],axis=1 )
			# j = np.where(difference <tol)[0]
			
			difference = np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:]
			difference = np.sqrt(difference[:,0]*difference[:,0] + difference[:,1]*difference[:,1] + difference[:,2]*difference[:,2]).reshape(inode,1)
			j = np.asarray(whereLT(difference, tol)[0])
		
		if j.shape[0]!=0:
			# x,y=np.where(reelements==inode)
			# reelements[x[0],y[0]] = j[0]
			# reelements[reelements==inode] = j[0]
			# jmin = np.min(j) # j[0] IS BY DEFAULT jmin
			reelements[whereEQ(reelements,inode)] = j[0]
			# duplicates = np.concatenate((duplicates,np.array([[j[0],inode]])),axis=0)

			# if inode!=j[0]:
			duplicates[counter,:] = np.array([j[0],inode])
			counter +=1


	tnodes = time()-tnodes


	# duplicates = (duplicates[1:,:]).astype(int)
	duplicates = duplicates[:counter,:]
	totalnodes = (np.linspace(0,repoints.shape[0]-1,repoints.shape[0])).astype(int)
	remainingnodes = np.delete(totalnodes,duplicates[:,1])
	mapnodes = (np.linspace(0,remainingnodes.shape[0]-1,remainingnodes.shape[0])).astype(int)


	telements_2 = time()

	ParallelTuple3=[]
	if Parallel:
		ParallelTuple3 = parmap.map(ElementLoopTet_2,np.arange(mesh.points.shape[0],mapnodes.shape[0]),reelements,remainingnodes,pool=MP.Pool(processes=nCPU))

	for i in range(mesh.points.shape[0],mapnodes.shape[0]):
		if Parallel:
			x=ParallelTuple3[i-mesh.points.shape[0]][0]; y=ParallelTuple3[i-mesh.points.shape[0]][1]
			reelements[x,y] = i
		else:
			reelements[whereEQ(reelements,remainingnodes[i])] = i 
			# reelements[reelements==remainingnodes[i]] = i
			# x,y=np.where(reelements==remainingnodes[i])
			# reelements[x,y] = i

	telements_2 = time()-telements_2

	# REPOINTS
	repoints = repoints[remainingnodes,:]
	#------------------------------------------------------------------------------------------




	#------------------------------------------------------------------------------------------
	# BUILD EDGES NOW
	tedges = time()

	reedges = np.zeros((mesh.edges.shape[0],C+2))
	reedges[:,:2]=mesh.edges
	for iedge in range(0,mesh.edges.shape[0]):
		# TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
		node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)
		node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)

		# FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
		L1 = np.linalg.norm(node1-repoints[mesh.points.shape[0]:,:],axis=1)
		L2 = np.linalg.norm(node2-repoints[mesh.points.shape[0]:,:],axis=1)
		L = np.linalg.norm(node1-node2,axis=1)

		j = np.where(np.abs((L1+L2)-L) < tol)[0]
		if j.shape[0]!=0:
			reedges[iedge,2:] = j+mesh.points.shape[0]
	reedges = reedges.astype(int)

	tedges = time()-tedges
	#------------------------------------------------------------------------------------------


	#------------------------------------------------------------------------------------------
	# BUILD FACES NOW 
	# A SIMILAR ANALOGY TO FACES IS FOLLOWED

	# C
	# #
	#   # 	 	if |area(ABD)+area(ADC)+area(BCD) = area(ABC)| then D lies inside the triangular face
	#     #
	#   D   #
	# A       #
	# # # # # # # B    

	# ALSO CHECK COPLANARITIES OF 4 POINTS: 	http://mathworld.wolfram.com/Coplanar.html
	# IF A POINT LIES INSIDE A TRIANGLE IN 2D:  http://mathworld.wolfram.com/TriangleInterior.html
	tfaces = time()

	fsize = int((C+2.)*(C+3.)/2.)
	refaces = np.zeros((mesh.faces.shape[0],fsize))
	refaces[:,:3] = mesh.faces
	# FIND THE UNIT NORMAL VECTOR TO THE FACE
	ParallelTuple6 = []
	if Parallel:
		ParallelTuple6 = parmap.map(LoopFaceTet,np.arange(0,mesh.faces.shape[0]),mesh.points,repoints,mesh.faces,tol,pool=MP.Pool(processes=nCPU))
	for iface in range(0,mesh.faces.shape[0]):
		if Parallel:
			j = ParallelTuple6[iface]
		else:
			# FIND THE POSITION VECTOR OF VERTICES 
			A = mesh.points[mesh.faces[iface,0],:]
			B = mesh.points[mesh.faces[iface,1],:]
			C = mesh.points[mesh.faces[iface,2],:]
			# FIND THE VECTORS OF ALL EDGES
			BA = B - A 		# VECTOR OF EDGE 1
			CA = C - A 		# VECTOR OF EDGE 2
			CB = C - B  	# VECTOR OF EDGE 3

			DA = repoints[mesh.points.shape[0]:,:] - np.repeat(A.reshape(1,3),repoints.shape[0] - mesh.points.shape[0],axis=0 )
			DB = repoints[mesh.points.shape[0]:,:] - np.repeat(B.reshape(1,3),repoints.shape[0] - mesh.points.shape[0],axis=0 )

			# COMPUTE AREAS 
			Area = 0.5*np.linalg.norm(np.cross(BA,CA)); Area = Area*np.ones(DA.shape[0])
			# Area1 = 0.5*np.linalg.norm(np.cross(DA,BA),axis=1); #Area1 = Area1*np.ones(DA.shape[0])
			Area1 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(BA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
			Area2 = 0.5*np.linalg.norm(np.cross(DA, np.repeat(CA.reshape(1,3),DA.shape[0],axis=0)),axis=1 )
			Area3 = 0.5*np.linalg.norm(np.cross(DB, np.repeat(CB.reshape(1,3),DB.shape[0],axis=0)),axis=1 )
			# t1 +=( time()-tt)
			# CHECK THE CONDITION
			j = np.where(np.abs( Area - (Area1+Area2+Area3) )  < tol)[0]
			# j = np.asarray(whereLT(np.abs( Area - (Area1+Area2+Area3) ).reshape(Area.shape[0],1),tol)[0]) 
		if j.shape[0]!=0:
			refaces[iface,3:] = j+mesh.points.shape[0]

	refaces = refaces.astype(int)

	tfaces = time()- tfaces


	class nmesh(object):
		# """Construct pMesh"""
		points = repoints
		elements = reelements
		edges = reedges
		faces = refaces
		nnode = repoints.shape[0]
		nelem = reelements.shape[0]
		info = 'tet'


	# print '\npMeshing timing:\n\t\tElement loop 1:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
	#  ' seconds'+'\n\t\tElement loop 2:\t '+str(telements_2)+' seconds\n\t\tEdge loop:\t\t '+str(tedges)+' seconds'+\
	#  '\n\t\tFace loop:\t\t '+str(tfaces)+' seconds\n'

	return nmesh
#---------------------------------------------------------------------------------------------------------------------------------------#
