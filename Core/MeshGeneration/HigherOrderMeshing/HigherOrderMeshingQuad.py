import numpy as np 
# from scipy.stats import itemfreq
from time import time
import multiprocessing as MP

import Core.ParallelProcessing.parmap as parmap
import GetInteriorCoordinates as Gett
import Core.InterpolationFunctions.TwoDimensional.Quad.QuadLagrangeGaussLobatto as TwoD 
from Core.Supplementary.Where import *
from Core.Supplementary.Tensors import itemfreq_py

# import imp, os
# pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
# TwoD = imp.load_source('QuadLagrangeGaussLobatto',pwd+'/Core/InterpolationFunctions/TwoDimensional/Quad/QuadLagrangeGaussLobatto.py')



#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------------------------------------------------#

def HighOrderMeshQuad(C,mesh,Parallel=False,nCPU=1):

	if mesh.points.shape[1]!=2:
		raise ValueError('Incompatible mesh coordinates size. mesh.point.shape[1] must be 2')

	eps = TwoD.LagrangeGaussLobatto(C,0,0)[1]

	# COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
	Neval = np.zeros((4,eps.shape[0]),dtype=np.float64)
	for i in range(0,eps.shape[0]):
		Neval[:,i] = TwoD.LagrangeGaussLobatto(0,eps[i,0],eps[i,1])[0]

	nodeperelem = mesh.elements.shape[1]
	renodeperelem = (C+2)**2
	left_over_nodes = renodeperelem - nodeperelem

	reelements = -1*np.ones((mesh.elements.shape[0],renodeperelem),dtype=int)
	# reelements[:,:4] = mesh.elements
	# repoints = np.copy(mesh.points)
	iesize = 4*C + C**2
	repoints = np.zeros((mesh.points.shape[0]+iesize*mesh.elements.shape[0],2))
	repoints[:mesh.points.shape[0],:]=mesh.points[:,:2]


	# QUAD NODES ARE ARANGED CONSISTENTLY COUNTER-CLOCKWISE, HENCE WE NEED NODES BELONG TO THE VERTICES
	vNnodes = (C+1)*np.array([0,1,2,3])
	ieNodes = np.delete(np.arange(0,renodeperelem),vNnodes)

	for elem in range(0,mesh.elements.shape[0]):
		maxNode=[]
		if elem==0:
			# maxNode = np.max(np.unique(mesh.elements))
			maxNode = np.max(mesh.elements)
		else:
			maxNode = np.max(reelements)

		xycoord =  mesh.points[mesh.elements[elem,:],:]
		# GET HIGHER ORDER COORDINATES
		xycoord_higher = Gett.GetInteriorNodesCoordinates(xycoord,'quad',elem,eps)
		# EXPAND THE ELEMENT CONNECTIVITY
		# reelements[elem,4:] = np.linspace(maxNode+1,maxNode+left_over_nodes,left_over_nodes).astype(int)
		reelements[elem,0:reelements.shape[1]-C**2:C+1] = mesh.elements[elem,:]
		reelements[elem,ieNodes] = np.linspace(maxNode+1,maxNode+left_over_nodes,left_over_nodes).astype(int)
		# reelements[elem,0:reelements.shape[1]-C**2:C+1] = mesh.elements[elem,:]
		# repoints = np.concatenate((repoints,xycoord_higher[3:,:]),axis=0)
		repoints[mesh.points.shape[0]+elem*iesize:mesh.points.shape[0]+(elem+1)*iesize] = xycoord_higher[ieNodes,:]

	#--------------------------------------------------------------------------------------
	# LOOP OVER POINTS
	tol = 1e-14
	duplicates = np.zeros((1,2))
	# for inode in range(0,repoints.shape[0]):
	for inode in range(mesh.points.shape[0],repoints.shape[0]): 	# FOR SOME REASON THIS TAKES MORE TIME FOR FINES MESHES

		difference =  np.linalg.norm( np.repeat(repoints[inode,:].reshape(1,repoints.shape[1]),inode,axis=0) - repoints[:inode,:],axis=1 )
		j = np.where(difference <tol)[0]
		if j.shape[0]!=0:
			x,y=np.where(reelements==inode)
			reelements[x[0],y[0]] = j[0]
			duplicates = np.concatenate((duplicates,np.array([[j,inode]])),axis=0)

	duplicates = (duplicates[1:,:]).astype(int)
	totalnodes = (np.linspace(0,repoints.shape[0]-1,repoints.shape[0])).astype(int)
	remainingnodes = np.delete(totalnodes,duplicates[:,1])
	mapnodes = (np.linspace(0,remainingnodes.shape[0]-1,remainingnodes.shape[0])).astype(int)

	for i in range(mesh.points.shape[0],mapnodes.shape[0]):
		x,y=np.where(reelements==remainingnodes[i])
		reelements[x,y] = i

	# REPOINTS
	repoints = repoints[remainingnodes,:]
	#------------------------------------------------------------------------------------------

	#------------------------------------------------------------------------------------------
	# BUILD EDGES NOW
	reedges = np.zeros((mesh.edges.shape[0],C+2))
	reedges[:,:2]= mesh.edges
	for iedge in range(0,mesh.edges.shape[0]):
		# TWO NODES OF THE LINEAR MESH REPLICATED REPOINTS.SHAPE[0] NUMBER OF TIMES 
		node1 = np.repeat(mesh.points[mesh.edges[iedge,0],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)
		node2 = np.repeat(mesh.points[mesh.edges[iedge,1],:].reshape(1,repoints.shape[1]),repoints.shape[0]-mesh.points.shape[0],axis=0)

		# FIND WHICH NODES LIE ON THIS EDGE BY COMPUTING THE LENGTHS  -   A-----C------B  /IF C LIES ON AB THAN AC+CB=AB 
		L1 = np.linalg.norm(node1-repoints[mesh.points.shape[0]:],axis=1)
		L2 = np.linalg.norm(node2-repoints[mesh.points.shape[0]:],axis=1)
		L = np.linalg.norm(node1-node2,axis=1)

		j = np.where(np.abs((L1+L2)-L) < tol)[0]
		if j.shape[0]!=0:
			reedges[iedge,2:] = j+mesh.points.shape[0]
	reedges = reedges.astype(int)
	#------------------------------------------------------------------------------------------


	class nmesh(object):
		# """Construct pMesh"""
		points = repoints
		elements = reelements
		edges = reedges
		faces = []
		nnode = repoints.shape[0]
		nelem = reelements.shape[0]
		info = 'quad'

	return nmesh