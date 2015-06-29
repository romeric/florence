import numpy as np
# import igakit.cad as iga
import cad as iga
# from igakit.igalib import bsp
# from createc1fromlines import curve_length
from igalib import bsp as bsp_local
from warnings import warn 
from time import time
import itertools
import multiprocessing as mp 

def Nurbs(mesh,nurbs,BoundaryData,BasesOrder):

	# nOfBCstrings = len(bcs)
	nOfBCstrings = 1
	dirichletFaces = np.zeros((mesh.edges.shape[0], 3),dtype=np.int64)
	indexFace = 0
	listFaces  = []


	ndim=0
	if mesh.element_type == 'tri':
		ndim = 2 
	elif mesh.element_type == 'tet':
		ndim = 3
	else:
		raise NotImplementedError('Boundary indentification with NURBS is only implemented for tris and tets')


	# t1=time()
	t1=0
	ProjU  = np.zeros((mesh.edges.shape[0],ndim),dtype=np.float64)
	ProjID = np.zeros((mesh.edges.shape[0],ndim),dtype=np.int64)
	
	for kFace in xrange(mesh.edges.shape[0]):
		edge_coords = mesh.points[mesh.edges[kFace,:2],:] # THE LINEAR MESH EDGES

		for iBC in xrange(1):
			# print np.all(BoundaryData().bcs(edge_coords))
			# if np.all(BoundaryData().bcs(edge_coords)) == False:
				# print np.linalg.norm(mesh.points[mesh.edges[kFace],:],axis=1)
			if np.all(BoundaryData().NURBSCondition(edge_coords)) == True:
				# print np.linalg.norm(mesh.points[mesh.edges[kFace],:],axis=1)
				listFaces.append(kFace)
				dirichletFaces[indexFace,:2] = mesh.edges[kFace,:2]
				# print mesh.edges[kFace,:2]
				indexFace += 1
				# for i in range(mesh.edges[kFace,:2].shape[0]):
				for iVertex in range(ndim):
					# _, uProjI, nurbsProjI = projectNodeNurbsBoundary(edge_coords[iVertex,:],nurbs,ndim) 
					# t9=time()
					uProjI, nurbsProjI = projectNodeNurbsBoundary(edge_coords[iVertex,:],nurbs,ndim) 
					# t1 += (time()-t9)
					# print nurbsProj
					# print xProj
					ProjU[kFace,iVertex] = uProjI
					ProjID[kFace,iVertex] = nurbsProjI

					# from OCC.gp import gp_Pnt
					# from OCC.GeomAPI import GeomAPI_ProjectPointOnCurve
					# pp=edge_coords[iVertex,:]

					#---------------------------------------------------------------------#
					#					CORRECT PARAMETER FOR PERIODIC NURBS
					#---------------------------------------------------------------------#

					# idNurbs =  identifyIdNurbs(nurbs,edge_coords) # DON'T NEED THIS FOR 2D 
				idNurbs = ProjID[kFace,1]
				dirichletFaces[indexFace,2] = idNurbs


	dirichletFaces = dirichletFaces[:indexFace,:]
	# print dirichletFaces.shape
	# print dirichletFaces
	# print mesh.edges[listFaces,:]
	# print np.linalg.norm(mesh.points[np.unique(mesh.edges[listFaces,:]),:],axis=1)
	# print ProjID
	print ProjU
	# print listFaces

	# print time()-t1
	# print t1

	# FIX PERIODIC NURBS 2D
	nOfNurbs = len(nurbs)
	uMin = 1e10 + np.zeros(nOfNurbs)
	uMax = -1e-10 + np.zeros(nOfNurbs)
	Lmin = np.zeros(nOfNurbs)
	Lmax = np.zeros(nOfNurbs)

	for kFace in listFaces:
		u1 = ProjU[kFace,0]
		u2 = ProjU[kFace,1]
		idNurbs = ProjID[kFace,1]
		uMin[idNurbs] = min([uMin[idNurbs],u1,u2])
		uMax[idNurbs] = max([uMax[idNurbs],u1,u2])

	# print uMin,uMax
	lengthTOL = 1e-10
	for idNurbs in range(nOfNurbs):
		aNurbs = nurbs[idNurbs]
		u1 = aNurbs['U'][0][0]
		u2 = min([uMin[idNurbs],aNurbs['start']])
		if abs(u1-u2) < lengthTOL:
			Lmin[idNurbs] = 0
		else:
			Lmin[idNurbs] = CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL,BasesOrder) # CHECK THIS WITH RUBEN'S CODE

		u1 = max([uMax[idNurbs],aNurbs['end']])
		u2 = aNurbs['U'][0][-1]
		if abs(u1-u2) < lengthTOL:
			Lmax[idNurbs] = 0
		else:
			Lmax[idNurbs] = CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL,BasesOrder) # CHECK THIS WITH RUBEN'S CODE

	correctMaxMin = -np.ones(nOfNurbs)
	# 0 CORRECT MIN & 1 CORRECT MAX
	for idNurbs in range(nOfNurbs):
		if Lmin[idNurbs] < Lmax[idNurbs]:
			correctMaxMin[idNurbs] = 0
		else:
			correctMaxMin[idNurbs] = 1

	indexFace = 0

	
	for kFace in listFaces:
		u1 = ProjU[kFace,0]
		u2 = ProjU[kFace,1]
		idNurbs = dirichletFaces[indexFace,2]
		indexFace +=1 # THIS IS A DEPARTURE FROM RUBEN'S CODE
		if correctMaxMin[idNurbs]==0:
			# CORRECT MIN
			if u1>u2 and np.abs(u2 - uMin[idNurbs]) < 1e-10:
				# PROBLEM WITH PERIODIC 
				ProjU[kFace,1] = nurbs[idNurbs]['U'][0][-1]
			elif u1<u2 and np.abs(u1 - uMin[idNurbs]) < 1e-10:
				ProjU[kFace,0] = nurbs[idNurbs]['U'][0][0]
		else:
			# CORRECT MAX
			if u1>u2 and np.abs(u1 - uMax[idNurbs]) < 1e-10:
				ProjU[kFace,0] = nurbs[idNurbs]['U'][0][0]
			elif u1<u2 and np.abs(u2 - uMax[idNurbs]) < 1e-10:
				ProjU[kFace,1] = nurbs[idNurbs]['U'][0][-1]
	print 
	print ProjU


	# IDENTIFY HIGHER ORDRE DIRICHLET NODES
	nOfDirichletFaces = indexFace
	FaceNodes = np.unique(mesh.edges)
	nOfFaceNodes = mesh.edges.shape[1]
	# nodesDBC = np.zeros(nOfDirichletFaces*nOfFaceNodes,dtype=np.int64)
	nodesDBC = mesh.edges[listFaces,:].ravel()
	displacementDBC = np.zeros((nOfDirichletFaces*nOfFaceNodes, 2),dtype=np.float64)
	indexNode = np.arange(nOfFaceNodes)

	for iDirichletFace in range(nOfDirichletFaces):

		idNurbs = dirichletFaces[iDirichletFace,2]
		u1 = ProjU[listFaces[iDirichletFace],0]
		u2 = ProjU[listFaces[iDirichletFace],1]
		uEq = CurveEquallySpacedPoints(nurbs[idNurbs], u1, u2, nOfFaceNodes,1e-06,BasesOrder)
		xEq = np.zeros((nOfFaceNodes,2))
		for i in xrange(nOfFaceNodes):
			pt = CurvePoint(nurbs[idNurbs],uEq[i])[0]
			xEq[i,:] = pt[:2]

		xOld = mesh.points[nodesDBC[indexNode],:]
		displacementDBC[indexNode,:] = xEq[np.append(np.array([0,-1]),np.arange(1,nOfFaceNodes-1)),:] - xOld
		indexNode += nOfFaceNodes

	posUnique = np.unique(nodesDBC,return_index=True)[1]

	return nodesDBC[posUnique], displacementDBC[posUnique,:]



def CurveEquallySpacedPoints(aNurbs, u1, u2, nPoints, lengthTOL=1e-06,BasesOrder=1):
	nOfMaxIterations = 1000
	uEq = np.zeros(nPoints,dtype=np.float64)
	uEq[0] = u1
	uEq[nPoints-1] = u2
	lengthU = np.abs(u2-u1)

	length = CurveLengthAdaptive(aNurbs, u1, u2, lengthTOL,BasesOrder)
	lengthSub = float(length)/(float(nPoints)-1.0)


	uGuess = np.linspace(u1,u2,nPoints)
	nOfIntPoints = nPoints-2

	for iIntPoints in range(nOfIntPoints):
		currentLength = (iIntPoints+1)*lengthSub
		a = u1
		b = u2
		# INITIAL GUESS 
		uOld = 0.5*(uGuess[iIntPoints] + uGuess[iIntPoints+2])
		# print uGuess[iIntPoints], uGuess[iIntPoints+2]
		length = CurveLengthAdaptive(aNurbs, u1 , uOld, lengthTOL,BasesOrder)

		for niter in range(nOfMaxIterations):
			f = length - currentLength
			if np.abs(f) < lengthTOL:
				uEq[iIntPoints+1] = uOld
				break
			elif f>0.0:
				b=uOld
			else:
				a=uOld

			uEq[iIntPoints+1] = 0.5*(a+b)

			length = CurveLengthAdaptive(aNurbs, u1 , uEq[iIntPoints+1], lengthTOL,BasesOrder)
			errU = np.abs(uOld-uEq[iIntPoints+1])/lengthU
			errF = np.abs(currentLength - length)/currentLength
			if errU<lengthTOL and errF < lengthTOL:
				break
			uOld = uEq[iIntPoints+1]

	return uEq



def TotalCurveLength(c,res=1000):
	"""Length of the total NURBS curve"""
	u=np.linspace(c['U'][0][0],c['U'][0][-1],res)
	C=np.zeros((u.shape[0],3))
	L=0.
	for i in range(u.shape[0]):
		# C[i,:] =  bsp_local.Evaluate1(c['degree'],c['U'],c['Pw'],u[i])
		C[i,:] =  CurvePoint(c,u[i])[0][:3]
		# L=0.
	for i in range(C.shape[0]-1):
		L += np.linalg.norm(C[i+1,:]-C[i,:])

	return L


def CurveLength_Gauss(c,u1,u2,res=500):
	"""Length of NURBS curve between parametric points u1 and u2 using Gaussian quadrature
	but does not take into account kinks"""
	pass




def CurveLength(c,u1,u2,res=500):
	"""Length of NURBS curve between parametric points u1 and u2"""
	
	u=np.linspace(u1,u2,res)
	C=np.zeros((u.shape[0],3))
	L=0.
	# t1=time()
	C = map(CurvePoint, itertools.repeat(c, u.shape[0]),u)
	# print time()-t1
	for i in range(len(C)-1):
		L += np.linalg.norm(C[i+1][0]-C[i][0])

	return L


def CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL,BasesOrder=1):
	
	L=None
	kMax = 20
	if BasesOrder < 3:
		n = 10
	else:
		n=10 # CHANGE THIS FOR SPEED
	# DECREASE LENGTHTOL FOR MORE COMPLEX CURVES
	L0 = CurveLength(aNurbs,u1,u2,n)
	# print u1,u2
	for k in range(2,kMax):
		L1 = CurveLength(aNurbs,u1,u2,k*n)
		errInt = np.abs(1.0*(L1 - L0)/L1)
		if errInt < lengthTOL:
			L = L1
			break
		L0 = L1 

	if k==kMax-1:
		raise StopIteration('Tolerance has not been achieved')
		L = L1

	return L



def projectNodeNurbsBoundary(x,nurbs,ndim):
	# NO OF NURBS CURVES/SURFACES
	nOfNurbs = len(nurbs)
	# print nOfNurbs

	d = 1e10
	for iNurbs in range(nOfNurbs):
		if ndim==2:
			u,pu = nurbsCurvePointProjection(nurbs[iNurbs], x)
		elif ndim==3:
			raise NotImplementedError('3D verion not implemented')

		dist = np.linalg.norm(pu[:ndim]-x)
		# print u,pu
		# print dist, pu
		# print x,pu[:ndim]
		if dist < d:
			nurbsProj = iNurbs
			# xProj = pu[:ndim]
			uProj = u 
			d = dist

	# return xProj, uProj, nurbsProj
	return uProj, nurbsProj




def nurbsCurvePointProjection(nurbs,p):

	# IF NURBS IS A SEGMENT
	if p.shape[0] == 2:
		p = np.append(p,0)

	if nurbs['Pw'].shape[0] == 2:
		p0 = nurbs['Pw'][0,:2]
		p1 = nurbs['Pw'][1,:2]
		A = np.array([
			[p1[0]-p0[0], p1[1]-p0[1]],
			[p0[1]-p1[1], p1[0]-p0[0]]
			])
		b = np.array([-p[0]*(p1[0]-p0[0]) - p[1]*(p1[1]-p0[1]), -p0[1]*(p1[0]-p0[0]) + p0[0]*(p1[1]-p0[1])])
		q = np.linalg.solve(A,b)

		# Check that the point is inside the segment
		dp1p0 = np.linalg.norm(p0-p1)
		dQp0 = np.linalg.norm(p0-q)
		dQp1 = np.linalg.norm(p1-q)

		if dQp0>dp1p0 or dQp1>dp1p0:
			# Outside. Take the projection as the closest end point
			if dQp0<dQp1:
				Cu = p0
				u = nurbs['U'][0]
			else:
				Cu = p1
				u = nurbs['U'][-1]
		else:
			Cu = q
			u = nurbs['U'][0] + 1.0*(dQp0/dp1p0)*(nurbs['U'][-1] - nurbs['U'][0])

	else:

		nMaxIter = 1000
		tol1 = 1e-10
		tol2 = 1e-10
		n = 1000

		# if p.shape[0] == 2:
			# p[3]=0

		pIni = CurvePoint(nurbs,nurbs['start'])[0]
		pEnd = CurvePoint(nurbs,nurbs['end'])[0]

		peroidic = 0
		if np.linalg.norm(pIni - pEnd) < tol1:
			peroidic = 1

		d = float('Inf') 
		us = np.linspace(nurbs['start'], nurbs['end'], n)

		for i in range(n):
			q = CurvePoint(nurbs,us[i])[0]
			dNew = np.linalg.norm(p-q)
			if dNew < d:
				u = us[i]
				d = dNew

		dMin = d
		Cu = CurvePoint(nurbs, u)[0]
		dCu = CurveDersPoints(nurbs,u)
		d2Cu = CurveSecondDersPoints(nurbs,u)

		for niter in xrange(nMaxIter):
			cond1 = Cu - p 
			# print cond1
			if np.linalg.norm(cond1) < tol1:
				break
			cond2 = np.linalg.norm(np.dot(dCu,cond1)/(np.linalg.norm(dCu)*np.linalg.norm(cond1)))
			if cond2 < tol2:
				break 

			uNew = u - np.dot(dCu,(Cu-p))/(np.dot(d2Cu,(Cu-p))+np.sum(dCu**2))
			# uNew = u - dCu*(Cu-p)'/(d2Cu*(Cu-p)' + sum(dCu.^2));
			if peroidic == 0:
				if uNew < nurbs['start']:
					uNew = nurbs['start']
				elif uNew > nurbs['end']:
					uNew = nurbs['end']
			else:
				if uNew < nurbs['start']:
					uNew = nurbs['end'] - (nurbs['start'] - uNew)
				elif uNew > nurbs['end']:
					uNew = nurbs['start'] - (nurbs['end'] - uNew)

				if uNew > nurbs['end'] or uNew < nurbs['start']:
					break

			cond3 = np.linalg.norm(np.dot((uNew-u),dCu))
			if cond3 < tol1:
				break
			u = uNew
			Cu = CurvePoint(nurbs, u)[0]
			dCu = CurveDersPoints(nurbs,u)
			d2Cu = CurveSecondDersPoints(nurbs,u)


			if np.linalg.norm(cond1) < dMin:
				dMin = np.linalg.norm(cond1)

		if niter == nMaxIter-1:
			raise StopIteration('Convergence not achieved for point projection')

		U = np.unique(nurbs['U'][0])  # tip: Completely avoid this, it is okay to loop over all knots than finding unique and calling CurvePoint on a loop
		# U = nurbs['U'][0] # use something like this instead 
		# print U, dMin
		# CHECK IF A KNOT IS CLOSER AND WE HAVE NOT CONVERED TO IT (END POINTS OR TOLERANCE)
		for uu in U:
			Cu = CurvePoint(nurbs, uu)[0]
			d = np.linalg.norm(Cu-p)
			if dMin - d > -tol1:
				u = uu 
				dMin = d

		Cu = CurvePoint(nurbs, u)[0]

		return u,Cu

	







def CurvePoint(nurbs,u,homogeneous = True):

	if nurbs['U'][0].shape[0] > 0:
		# pt = bsp.Evaluate1(nurbs['degree'],nurbs['U'],nurbs['Pw'],u)
		# print nurbs['U'][0].shape,nurbs['Pw'].shape
		pt = bsp_local.Evaluate1(nurbs['degree'],nurbs['U'],nurbs['Pw'],u)
		# if nurbs['degree']==0:
			# print pt
			# print nurbs['U'][0].shape,nurbs['Pw'].shape
		wt = pt[0,-1]
		if homogeneous:
			pt = pt[0,:3]/pt[0,-1]
		else:
			pt = pt[0,:3]
	else:
		pt = np.zeros(3)
	return pt, wt 
	# return p[0,:3]/p[0,-1]


def CurveDersPoints(nurbs,u):
	nurbsDer = CurveDersControlPoints(nurbs)
	p1,w1 = CurvePoint(nurbsDer,u,homogeneous=False)
	# print p1,w1
	p2,w2 = CurvePoint(nurbs,u)
	return 1.0*(p1 - p2*w1)/w2


def CurveSecondDersPoints(nurbs,u):
	nurbsDer = CurveDersControlPoints(nurbs)
	nurbsSecondDer = CurveDersControlPoints(nurbsDer)
	p1,w1 = CurvePoint(nurbsSecondDer,u,homogeneous=False)
	p2,w2 = CurvePoint(nurbsDer,u,homogeneous=False)
	p3,w3 = CurvePoint(nurbs,u)
	du = 1.0*(p2 - p3*w2)/w3
	return (p1 -2.0*w2*du-w1*p3)/w3



def CurveDersControlPoints(nurbs):
	m = nurbs['U'][0].shape[0] - 1
	p = nurbs['degree']
	n = m - p -1

	PwDers = np.zeros((n,4),dtype=np.float64)
	for i in range(n):
		if nurbs['U'][0][i+p+1] - nurbs['U'][0][i+1] < 1e-5:
			PwDers[i,:] = 0
		else:
			PwDers[i,:] = p*(nurbs['Pw'][i+1,:]-nurbs['Pw'][i,:])/( nurbs['U'][0][i+p+1] - nurbs['U'][0][i+1] ) 

	# nurbsDer = {'U':(nurbs['U'][0][1:-1],),'Pw':(PwDers,),'start':nurbs['U'][0][1],'end':nurbs['U'][0][-1],'degree':p-1}
	# return nurbsDer
	# return  {'U':(nurbs['U'][0][1:-1],),'Pw':(PwDers,),'start':nurbs['U'][0][1],'end':nurbs['U'][0][-1],'degree':p-1}
	return  {'U':(nurbs['U'][0][1:-1],),'Pw':PwDers,'start':nurbs['U'][0][1],'end':nurbs['U'][0][-1],'degree':p-1}
	# return  {'U':(nurbs['U'][0][1:-1],),'Pw':PwDers,'start':nurbs['U'][0][1],'end':nurbs['U'][0][-1],'degree':p}