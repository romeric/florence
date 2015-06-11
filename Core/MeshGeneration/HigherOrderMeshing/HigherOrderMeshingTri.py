from time import time
import numpy as np 
# from scipy.stats import itemfreq
import multiprocessing as MP

import Core.ParallelProcessing.parmap as parmap
import GetInteriorCoordinates as Gett
import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri 
from Core.NumericalIntegration.FeketePointsTri import *
from Core.Supplementary.Where import *
from Core.Supplementary.Tensors import itemfreq_py
from NodeLoopTriNPSP_Cython import NodeLoopTriNPSP_Cython 

# import pyximport; pyximport.install()
# from Core.Supplementary.Where.whereEQ import *
# from Core.Supplementary.Where.whereLT import *


#---------------------------------------------------------------------------------------------------------------------------------------#
# NEW METHOD - BASED ON (SCIPY/NUMPY) ITEMFREQ, UNIQUE & ARGSORT --> UNSTABLE METHOD DUE TO ROUND-OFF

def NodeLoopTriNPSP_PARMAP_1(i,sorted_repoints,Xs,invX,tol):
	# IF THE MULITPLICITY OF A GIVEN X-VALUE IS 1 THEN INGONRE
	dups= None; Ys = None
	flags = None
	if Xs[i,1]!=1:
		# IF THE MULTIPLICITY IS MORE THAN 1, THEN FIND WHERE ALL IN THE SORTED ARRAY THIS X-VALUE THEY OCCURS
		# dups =  np.where(i==invX)[0]
		dups = np.asarray(whereEQ(invX.reshape(invX.shape[0],1),i)[0])
		# FIND THE Y-COORDINATE VALUES OF THESE MULTIPLICITIES 
		Ys = sorted_repoints[dups,:][:,1]
		if Ys.shape[0]==2:
			if np.abs(Ys[1]-Ys[0]) < tol:
				flags=1

	return dups, Ys, flags


def DuplicatesLoopTri(i,reelements,duplicates):
	return whereEQ(reelements,duplicates[i,1])
	# return np.where(reelements==duplicates[i,1])


def NodeLoopTriNPSP_PARMAP(sorted_repoints,Xs,invX,iSortX,duplicates,Decimals,tol,nCPU):

	ParallelTuple4 = parmap.map(NodeLoopTriNPSP_PARMAP_1,np.arange(0,Xs.shape[0]),sorted_repoints,Xs,invX,tol,pool=MP.Pool(processes=nCPU))

	counter=0
	# LOOP OVER POINTS
	for i in range(0,Xs.shape[0]):
		dups = ParallelTuple4[i][0]
		Ys = ParallelTuple4[i][1]
		flags = ParallelTuple4[i][2]
		if dups is not None:
			# if Ys.shape[0]==2:
			# 	if np.abs(Ys[1]-Ys[0]) < tol:
			# 		# IF EQUAL MARK THIS POINT AS DUPLICATE
			# 		duplicates[counter,:] = dups
			# 		# INCREASE THE COUNTER
			# 		counter += 1
			if flags is not None:
				# IF EQUAL MARK THIS POINT AS DUPLICATE
				duplicates[counter,:] = dups
				# INCREASE THE COUNTER
				counter += 1
			# MULTIPLICITY CAN BE GREATER THAN 2, IN WHICH CASE FIND MULTIPLICITY OF Ys
			else:
				# Ysy=itemfreq(np.round(Ys,decimals=Decimals))
				Ysy = itemfreq_py(Ys,decimals=Decimals)
				# IF itemfreq GIVES THE SAME LENGTH ARRAY, MEANS ALL VALUES ARE UNIQUE/DISTINCT AND WE DON'T HAVE TO CHECK
				if Ysy.shape[0]!=Ys.shape[0]:
					# OTHERWISE LOOP OVER THE ARRAY AND
					for j in range(0,Ysy.shape[0]):
						# FIND WHERE THE VALUES OCCUR
						YsYs = np.where(Ysy[j,0]==np.round(Ys,decimals=Decimals))[0]
						# THIS LEADS TO A SITUATION WHERE SAY 3 NODES HAVE THE SAME X-VALUE, BUT TWO OF THEIR Y-VALUES ARE THE
						# SAME AND ONE IS UNIQUE. CHECK IF THIS IS JUST A NODE WITH NO Y-MULTIPLICITY
						if dups[YsYs].shape[0]!=1:
							# IF NOT THEN MARK AS DUPLICATE
							duplicates[counter,:] = dups[YsYs]
							# INCREASE COUNTER
							counter += 1


	# RE-ASSIGN DUPLICATE
	duplicates = duplicates[:counter,:]
	# BASED ON THE DUPLICATES OCCURING IN THE SORTED ARRAY sorted_repoints, FIND THE ACTUAL DUPLICATES OCCURING IN repoints
	duplicates = np.asarray([iSortX[duplicates[:,0]],iSortX[duplicates[:,1]] ]).T
	# SORT THE ACTUAL DUPLICATE ROW-WISE SO THAT THE FIRST COLUMN IS ALWAYS SMALLER THAN THE SECOND COLUMN
	duplicates = np.sort(duplicates,axis=1)

	return duplicates


def NodeLoopTriNPSP(sorted_repoints,Xs,invX,iSortX,duplicates,Decimals,tol):
	# print sorted_repoints.shape, Xs.shape, invX.shape,iSortX.shape, duplicates.shape
	# print type(sorted_repoints[0,0]), type(Xs[0,0]), type(invX[0]), type(iSortX[0]), type(duplicates[0,0])
	counter = 0
	# LOOP OVER POINTS
	for i in range(0,Xs.shape[0]):
		# IF THE MULITPLICITY OF A GIVEN X-VALUE IS 1 THEN INGONRE
		if Xs[i,1]!=1:
			# IF THE MULTIPLICITY IS MORE THAN 1, THEN FIND WHERE ALL IN THE SORTED ARRAY THIS X-VALUE OCCURS
			# dups =  np.where(i==invX)[0]
			dups = np.asarray(whereEQ(invX.reshape(invX.shape[0],1),i)[0])

			# FIND THE Y-COORDINATE VALUES OF THESE MULTIPLICITIES 
			Ys = sorted_repoints[dups,:][:,1]
			# IF MULTIPLICITY IS 2 THEN FIND IF THEY ARE Y-VALUES ARE EQUAL  
			if Ys.shape[0]==2:
				if np.abs(Ys[1]-Ys[0]) < tol:
					# IF EQUAL MARK THIS POINT AS DUPLICATE
					duplicates[counter,:] = dups
					# INCREASE THE COUNTER
					counter += 1
			# MULTIPLICITY CAN BE GREATER THAN 2, IN WHICH CASE FIND MULTIPLICITY OF Ys
			else:
				# Ysy=itemfreq(np.round(Ys,decimals=Decimals))
				Ysy = itemfreq_py(Ys,decimals=Decimals)
				# IF itemfreq GIVES THE SAME LENGTH ARRAY, MEANS ALL VALUES ARE UNIQUE/DISTINCT AND WE DON'T HAVE TO CHECK
				if Ysy.shape[0]!=Ys.shape[0]:
					# OTHERWISE LOOP OVER THE ARRAY AND
					for j in range(0,Ysy.shape[0]):
						# FIND WHERE THE VALUES OCCUR
						YsYs = np.where(Ysy[j,0]==np.round(Ys,decimals=Decimals))[0]
						# THIS LEADS TO A SITUATION WHERE SAY 3 NODES HAVE THE SAME X-VALUE, BUT TWO OF THEIR Y-VALUES ARE THE
						# SAME AND ONE IS UNIQUE. CHECK IF THIS IS JUST A NODE WITH NO Y-MULTIPLICITY
						if dups[YsYs].shape[0]!=1:
							# IF NOT THEN MARK AS DUPLICATE
							duplicates[counter,:] = dups[YsYs]
							# INCREASE COUNTER
							counter += 1

	# RE-ASSIGN DUPLICATE
	duplicates = duplicates[:counter,:]
	# BASED ON THE DUPLICATES OCCURING IN THE SORTED ARRAY sorted_repoints, FIND THE ACTUAL DUPLICATES OCCURING IN repoints
	duplicates = np.asarray([iSortX[duplicates[:,0]],iSortX[duplicates[:,1]] ]).T
	# SORT THE ACTUAL DUPLICATE ROW-WISE SO THAT THE FIRST COLUMN IS ALWAYS SMALLER THAN THE SECOND COLUMN
	duplicates = np.sort(duplicates,axis=1)

	return duplicates


def HighOrderMeshTri_UNSTABLE(C,mesh,info=0,Decimals=10,Parallel=False,nCPU=1):


	# SWITCH OFF MULTI-PROCESSING FOR SMALLER PROBLEMS WITHOUT GIVING A MESSAGE
	if (mesh.elements.shape[0] < 1000) and (C < 8):
		Parallel = False
		nCPU = 1

	if mesh.points.shape[1]!=2:
		raise ValueError('Incompatible mesh coordinates size. mesh.point.shape[1] must be 2')

	eps =  FeketePointsTri(C)
	# COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
	Neval = np.zeros((3,eps.shape[0]),dtype=np.float64)
	for i in range(3,eps.shape[0]):
			Neval[:,i]  = Tri.hpBases(0,eps[i,0],eps[i,1],1)[0]

	nodeperelem = mesh.elements.shape[1]
	renodeperelem = int((C+2.)*(C+3.)/2.)
	left_over_nodes = renodeperelem - nodeperelem

	reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
	reelements[:,:3] = mesh.elements
	iesize = int( C*(C+5.)/2. )
	repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],2),dtype=np.float64)
	repoints[:mesh.points.shape[0],:]=mesh.points[:,:2]

	pshape0, pshape1 = mesh.points.shape[0], mesh.points.shape[1]
	repshape0, repshape1 = repoints.shape[0], repoints.shape[1]

	telements = time()

	xycoord_higher=[]; ParallelTuple1=[]
	if Parallel:
		# GET HIGHER ORDER COORDINATES - PARALLEL
		ParallelTuple1 = parmap.map(ElementLoopTri,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,info.MeshType,eps,
			Neval,pool=MP.Pool(processes=nCPU))

	# LOOP OVER ELEMENTS
	maxNode = np.max(reelements)
	for elem in range(0,mesh.elements.shape[0]):
		# maxNode = np.max(np.unique(reelements)) # BIGGEST BOTTLENECK
		# maxNode = np.max(reelements)

		# GET HIGHER ORDER COORDINATES
		if Parallel:
			xycoord_higher = ParallelTuple1[elem]
		else:
			# xycoord =  mesh.points[mesh.elements[elem,:],:]
			xycoord_higher = Gett.GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],info.MeshType,elem,eps,Neval)
	
		# EXPAND THE ELEMENT CONNECTIVITY
		newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
		# reelements[elem,3:] = np.arange(maxNode+1,maxNode+1+left_over_nodes) 	
		reelements[elem,3:] = newElements
		maxNode = newElements[-1]
		
		repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[3:,:]

	telements = time()-telements
	#--------------------------------------------------------------------------------------
	# LOOP OVER POINTS - FIND DUPLICATES
	tnodes=time()

	# TO FIND THE DUPLICATES WE NEED THE UNIQUE VALUES OF repoints WHICH IS FLOATING POINT ARRAY, SO WE FIRST
	# NEED TO ROUND OFF TO CERTAIN DECIMAL DIGIT.
	# Decimals = 10
	# SORT ONLY THE X COORDINATE OF repoints
	iSortX = np.argsort(repoints[:,0])
	# BASED ON THE SORTED X-VALUES, PUT THEIR CORRESPONDING Y-VALUES NEXT TO THEM I.E. SORT REPOINTS BASED ON X-VALUES
	sorted_repoints = repoints[iSortX,:]
	# NOW LETS FIND THE UNIQUE VALUES OF THIS SORTED FLOATING POINTS ARRAY
	# NOTE THAT FROM THE INVERSE INDICES OF A UNIQUE ARRAY WE CAN CONSTRUCT THE ACTUAL ARRAY 
	# invX =np.unique(np.round(sorted_repoints[:,0],decimals=Decimals),return_inverse=True)[1]
	unique_repoints,invX =np.unique(np.round(sorted_repoints[:,0],decimals=Decimals),return_inverse=True)
	
	# NOW FIND THE MULTIPLICITY OF EACH UNIQUE X-VALUES 
	# t2=time()
	# Xs =  itemfreq(np.round(sorted_repoints[:,0],decimals=Decimals)) # BIG PERFORMANCE BOTTLENECK
	
	# Xs = np.zeros((unique_repoints.shape[0],2),dtype=np.float64)
	# Xs[:,0] = unique_repoints; Xs[:,1]=np.bincount(invX)
	Xs = itemfreq_py(un_arr=unique_repoints,inv_arr=invX)
	# print np.linalg.norm(Xss -Xs)
	# print time()-t2
	

	tol = 1.0e-14
	duplicates = -1*np.ones((reelements.shape[0]*reelements.shape[1],2),dtype=np.int64)
	if Parallel:
		duplicates = NodeLoopTriNPSP_PARMAP(sorted_repoints,Xs,invX,iSortX,duplicates,Decimals,tol,nCPU)
	else:
		duplicates = NodeLoopTriNPSP(sorted_repoints,Xs,invX,iSortX,duplicates,Decimals,tol)
		# duplicates = NodeLoopTriNPSP_Cython(sorted_repoints,Xs,invX,iSortX,duplicates,Decimals,tol) # CYTHON VERSION
	# print duplicates

	ParallelTuple5 = []
	if Parallel:
		ParallelTuple5 = parmap.map(DuplicatesLoopTri,np.arange(0,duplicates.shape[0]),reelements,duplicates,pool=MP.Pool(processes=nCPU))

	for i in range(0,duplicates.shape[0]):
		if Parallel:
			x = ParallelTuple5[i][0]; y = ParallelTuple5[i][1];
			reelements[x,y] = duplicates[i,0] 
		else:
			reelements[whereEQ(reelements,duplicates[i,1])] = duplicates[i,0] 

	tnodes = time()-tnodes

	totalnodes = np.arange(0,repshape0) 
	remainingnodes = np.delete(totalnodes,duplicates[:,1])
	mapnodes = np.arange(0,remainingnodes.shape[0]) 

	telements_2 = time()

	ParallelTuple3=[]
	if Parallel:
		ParallelTuple3 = parmap.map(ElementLoopTri_2,np.arange(mesh.points.shape[0],mapnodes.shape[0]),reelements,remainingnodes,pool=MP.Pool(processes=nCPU))

	for i in range(pshape0,mapnodes.shape[0]):
		if Parallel:
			x=ParallelTuple3[i-mesh.points.shape[0]][0]; y=ParallelTuple3[i-mesh.points.shape[0]][1]
			reelements[x,y] = i
		else:
			reelements[reelements==remainingnodes[i]] = i 
			# reelements[np.where(reelements==remainingnodes[i])] = i 
			# x,y=np.where(reelements==remainingnodes[i])
		# reelements[x,y] = i

	telements_2 = time()-telements_2

	# REPOINTS
	repoints = repoints[remainingnodes,:]
	# UPDATE repshape0 & repshape1
	repshape0, repshape1 = repoints.shape[0], repoints.shape[1]
	#------------------------------------------------------------------------------------------


	#------------------------------------------------------------------------------------------
	# BUILD EDGES NOW
	reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
	reedges[:,:2]=mesh.edges

	tedges = time()
	for iedge in range(0,mesh.edges.shape[0]):
		# TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
		node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repshape1),repshape0-pshape0,axis=0)
		node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repshape1),repshape0-pshape0,axis=0)

		# FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
		L1 = np.linalg.norm(node1-repoints[pshape0:],axis=1)
		L2 = np.linalg.norm(node2-repoints[pshape0:],axis=1)
		L = np.linalg.norm(node1-node2,axis=1)

		j = np.where(np.abs((L1+L2)-L) < tol)[0]
		if j.shape[0]!=0:
			reedges[iedge,2:] = j+mesh.points.shape[0]
	reedges = reedges.astype(int)

	tedges = time()-tedges
	#------------------------------------------------------------------------------------------
	
	class nmesh(object):
		# """Construct pMesh"""
		points = repoints
		elements = reelements
		edges = reedges
		faces = []
		nnode = repoints.shape[0]
		nelem = reelements.shape[0]
		info = 'tri'

	# print '\npMeshing timing:\n\t\tElement loop 1:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
	#  ' seconds'+'\n\t\tElement loop 2:\t '+str(telements_2)+' seconds\n\t\tEdge loop:\t\t '+str(tedges)+' seconds\n'

	return nmesh
#---------------------------------------------------------------------------------------------------------------------------------------#




#---------------------------------------------------------------------------------------------------------------------------------------#
# OLD WAY - REPLICATING NODES WITHIN NODES LOOP
def ElementLoopTri(elem,elements,points,MeshType,eps,Neval):
	xycoord_higher = Gett.GetInteriorNodesCoordinates(points[elements[elem,:],:],MeshType,elem,eps,Neval)
	return xycoord_higher

def ElementLoopTri_2(i,reelements,remainingnodes):
	# return np.where(reelements==remainingnodes[i])
	return whereEQ(reelements,remainingnodes[i])

def NodeLoopTri(inode,reelements,repoints):
	tol=1.0e-14
	#--------------------------------------------------------------------------------------------------------------------------------------#
	# difference =  np.linalg.norm( np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:],axis=1 )
	# j = np.where(difference < tol)[0]
	#--------------------------------------------------------------------------------------------------------------------------------------#

	# difference =  np.linalg.norm( np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:],axis=1 ).reshape(repoints[:inode,:].shape[0],1)
	# j = np.asarray(whereLT(difference.reshape(difference.shape[0],1), tol)[0])
	# j = np.asarray(whereLT(difference, tol)[0])

	# DO IT THIS WAY
	#--------------------------------------------------------------------------------------------------------------------------------------#
	difference = np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:]
	difference = np.sqrt(difference[:,0]*difference[:,0] + difference[:,1]*difference[:,1]).reshape(inode,1)
	# difference = np.abs(difference[:,0] + difference[:,1]).reshape(inode,1) # INCORRECT
	# difference = np.abs(np.sum(repoints[:inode,:],axis=1) - np.sum(repoints[inode,:])).reshape(inode,1) # FASTEST BUT INCORRECT
	#--------------------------------------------------------------------------------------------------------------------------------------#


	j = np.asarray(whereLT(difference, tol)[0])

	return j


def HighOrderMeshTri(C,mesh,info=0,Parallel=False,nCPU=1):

	if mesh.points.shape[1]!=2:
		raise ValueError('Incompatible mesh coordinates size. mesh.point.shape[1] must be 2')

	eps =  FeketePointsTri(C)
	# COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
	Neval = np.zeros((3,eps.shape[0]),dtype=np.float64)
	for i in range(3,eps.shape[0]):
			Neval[:,i]  = Tri.hpBases(0,eps[i,0],eps[i,1],1)[0]

	nodeperelem = mesh.elements.shape[1]
	renodeperelem = int((C+2.)*(C+3.)/2.)
	left_over_nodes = renodeperelem - nodeperelem

	reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=np.int64)
	reelements[:,:3] = mesh.elements
	iesize = int( C*(C+5.)/2. )
	repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],2),dtype=np.float64)
	repoints[:mesh.points.shape[0],:]=mesh.points[:,:2]

	pshape0, pshape1 = mesh.points.shape[0], mesh.points.shape[1]
	repshape0, repshape1 = repoints.shape[0], repoints.shape[1]

	telements = time()

	xycoord_higher=[]; ParallelTuple1=[]
	if Parallel:
		# GET HIGHER ORDER COORDINATES - PARALLEL
		ParallelTuple1 = parmap.map(ElementLoopTri,np.arange(0,mesh.elements.shape[0]),mesh.elements,mesh.points,info.MeshType,eps,
			Neval,pool=MP.Pool(processes=nCPU))

	maxNode = np.max(reelements)
	for elem in range(0,mesh.elements.shape[0]):
		# maxNode = np.max(np.unique(reelements)) # BIGGEST BOTTLENECK
		# maxNode = np.max(reelements) # ALSO BIG BOTTLENECK


		# GET HIGHER ORDER COORDINATES
		if Parallel:
			xycoord_higher = ParallelTuple1[elem]
		else:
			# xycoord =  mesh.points[mesh.elements[elem,:],:]
			xycoord_higher = Gett.GetInteriorNodesCoordinates(mesh.points[mesh.elements[elem,:],:],info.MeshType,elem,eps,Neval)

		# EXPAND THE ELEMENT CONNECTIVITY
		newElements = np.arange(maxNode+1,maxNode+1+left_over_nodes) 
		# reelements[elem,3:] = np.arange(maxNode+1,maxNode+1+left_over_nodes) 	
		reelements[elem,3:] = newElements
		maxNode = newElements[-1]

		repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[3:,:]

	telements = time()-telements
	#--------------------------------------------------------------------------------------
	# LOOP OVER POINTS

	tnodes = time()

	tol = 1e-14
	# duplicates = np.zeros((1,2))
	duplicates = -1*np.ones((reelements.shape[0]*reelements.shape[1],2),dtype=np.int64)
	
	ParallelTuple2=[]
	if Parallel:
		ParallelTuple2 = parmap.map(NodeLoopTri,np.arange(mesh.points.shape[0],repshape0),reelements,repoints,pool=MP.Pool(processes=nCPU))
	
	counter = 0
	# for inode in range(mesh.points.shape[0],repoints.shape[0]): 	# FOR SOME REASON THIS TAKES MORE TIME FOR FINE MESHES
	for inode in range(pshape0,repshape0): 	# FOR SOME REASON THIS TAKES MORE TIME FOR FINE MESHES
	
		if Parallel:
			j = ParallelTuple2[inode-mesh.points.shape[0]]

		else:
			# difference =  np.linalg.norm( np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:],axis=1 )
			# j = np.where(difference <tol)[0]
			# j = np.asarray(whereLT(difference.reshape(difference.shape[0],1), tol)[0])

			#--------------------------------------------------------------------------------------------------------------------------------------#
			difference = np.repeat(repoints[inode,:].reshape(1,repshape1),inode,axis=0) - repoints[:inode,:]
			difference = np.sqrt(difference[:,0]*difference[:,0] + difference[:,1]*difference[:,1]).reshape(inode,1)
			# difference = np.abs(difference[:,0] + difference[:,1]).reshape(inode,1) # INCORRECT
			# difference = np.abs(np.sum(repoints[:inode,:],axis=1) - np.sum(repoints[inode,:])).reshape(inode,1) # FAST BUT INCORRECT
			#--------------------------------------------------------------------------------------------------------------------------------------#
			
			j = np.asarray(whereLT(difference, tol)[0])
		
		if j.shape[0]!=0:
			if j!=inode:
				reelements[whereEQ(reelements,inode)] = j 
			# reelements[whereEQ(reelements,inode)] = j 
			# reelements[reelements==inode] = j 

			# x,y=np.where(reelements==inode)
			# reelements[x[0],y[0]] = j

			# duplicates = np.concatenate((duplicates,np.array([[j,inode]])),axis=0)
			duplicates[counter,:] = np.array([j,inode])
			counter +=1

	tnodes = time()-tnodes

	# duplicates = (duplicates[1:,:]).astype(int)
	duplicates = duplicates[:counter,:]
	# totalnodes = np.arange(0,repoints.shape[0]) 
	totalnodes = np.arange(0,repshape0) 
	remainingnodes = np.delete(totalnodes,duplicates[:,1])
	mapnodes = np.arange(0,remainingnodes.shape[0]) 



	telements_2 = time()

	ParallelTuple3=[]
	if Parallel:
		ParallelTuple3 = parmap.map(ElementLoopTri_2,np.arange(mesh.points.shape[0],mapnodes.shape[0]),reelements,remainingnodes,pool=MP.Pool(processes=nCPU))

	for i in range(pshape0,mapnodes.shape[0]):
		if Parallel:
			x=ParallelTuple3[i-mesh.points.shape[0]][0]; y=ParallelTuple3[i-mesh.points.shape[0]][1]
			reelements[x,y] = i
		else:
			reelements[reelements==remainingnodes[i]] = i 
			# reelements[np.where(reelements==remainingnodes[i])] = i 
			# x,y=np.where(reelements==remainingnodes[i])
		# reelements[x,y] = i

	telements_2 = time()-telements_2

	# REPOINTS
	repoints = repoints[remainingnodes,:]
	# UPDATE repshape0 & repshape1
	repshape0, repshape1 = repoints.shape[0], repoints.shape[1]
	#------------------------------------------------------------------------------------------


	#------------------------------------------------------------------------------------------
	# BUILD EDGES NOW
	reedges = np.zeros((mesh.edges.shape[0],C+2),dtype=np.int64)
	reedges[:,:2]=mesh.edges

	tedges = time()
	for iedge in range(0,mesh.edges.shape[0]):
		# TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
		# node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)
		# node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)
		node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repshape1),repshape0-pshape0,axis=0)
		node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repshape1),repshape0-pshape0,axis=0)

		# FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
		# L1 = np.linalg.norm(node1-repoints[mesh.points.shape[0]:],axis=1)
		# L2 = np.linalg.norm(node2-repoints[mesh.points.shape[0]:],axis=1)
		L1 = np.linalg.norm(node1-repoints[pshape0:],axis=1)
		L2 = np.linalg.norm(node2-repoints[pshape0:],axis=1)
		L = np.linalg.norm(node1-node2,axis=1)

		j = np.where(np.abs((L1+L2)-L) < tol)[0]
		if j.shape[0]!=0:
			reedges[iedge,2:] = j+mesh.points.shape[0]
	reedges = reedges.astype(int)

	tedges = time()-tedges
	#------------------------------------------------------------------------------------------
	
	class nmesh(object):
		# """Construct pMesh"""
		points = repoints
		elements = reelements
		edges = reedges
		faces = []
		nnode = repoints.shape[0]
		nelem = reelements.shape[0]
		info = 'tri'

	# print '\npMeshing timing:\n\t\tElement loop 1:\t '+str(telements)+' seconds\n\t\tNode loop:\t\t '+str(tnodes)+\
	#  ' seconds'+'\n\t\tElement loop 2:\t '+str(telements_2)+' seconds\n\t\tEdge loop:\t\t '+str(tedges)+' seconds\n'

	return nmesh
#---------------------------------------------------------------------------------------------------------------------------------------#
