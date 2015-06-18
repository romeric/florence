import numpy as np
# import igakit.cad as iga
import cad as iga
# from igakit.igalib import bsp
# from createc1fromlines import curve_length
from igalib import bsp as bsp_local
from warnings import warn 
from time import time
# from igalib import bsp
# import igalib
# print dir(igalib.bsp)

def Nurbs(mesh,nurbs,BoundaryData):

	# nOfBCstrings = len(bcs)
	nOfBCstrings = 1
	dirichletFaces = np.zeros((mesh.edges.shape[0], 3),dtype=np.int64)
	indexFace = 0
	listFaces  = []


	# print mesh.edges
	nsd=0
	if mesh.element_type == 'tri':
		nsd = 2 
	elif mesh.element_type == 'tet':
		nsd = 3
	else:
		raise NotImplementedError('Boundary indentification with NURBS is only implemented for tris and tets')

	# GET COORDINATES OF NODES
	# unedges = np.unique(mesh[:,:2])
	# edge_coords = mesh.points[np.unique(mesh.edges),:]
	# print edge_coords

	ProjU  = np.zeros((mesh.edges.shape[0],nsd),dtype=np.float64)
	ProjID = np.zeros((mesh.edges.shape[0],nsd),dtype=np.int64)

	for kFace in range(mesh.edges.shape[0]):
		edge_coords = mesh.points[mesh.edges[kFace,:2],:] # THE LINEAR MESH EDGES

		for iBC in range(1):
			# print np.all(BoundaryData().bcs(edge_coords))
			if np.all(BoundaryData().bcs(edge_coords)) == True:
				listFaces.append(kFace)
				dirichletFaces[indexFace,:2] = mesh.edges[kFace,:2]
				indexFace += 1
				# for i in range(mesh.edges[kFace,:2].shape[0]):
				for iVertex in range(nsd):
					xProj, uProjI, nurbsProjI = projectNodeNurbsBoundary(edge_coords[iVertex,:],nurbs,nsd) 
					# print nurbsProj
					# print xProj
					ProjU[kFace,iVertex] = uProjI
					ProjID[kFace,iVertex] = nurbsProjI


					#---------------------------------------------------------------------#
					#					CORRECT PARAMETER FOR PERIODIC NURBS
					#---------------------------------------------------------------------#

					# idNurbs =  identifyIdNurbs(nurbs,edge_coords) # DON'T NEED THIS FOR 2D 
					idNurbs = ProjID[kFace,1]
					dirichletFaces[indexFace,2] = idNurbs


	dirichletFaces = dirichletFaces[:indexFace,:]
	# print dirichletFaces

	# FIX PERIODIC NURBS
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
			Lmin[idNurbs] = CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL) # CHECK THIS WITH RUBEN'S CODE

		u1 = max([uMax[idNurbs],aNurbs['end']])
		u2 = aNurbs['U'][0][-1]
		if abs(u1-u2) < lengthTOL:
			Lmax[idNurbs] = 0
		else:
			Lmax[idNurbs] = CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL) # CHECK THIS WITH RUBEN'S CODE

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


	# IDENTIFY HIGHER ORDRE DIRICHLET NODES
	nOfDirichletFaces = indexFace
	FaceNodes = np.unique(mesh.edges)
	nOfFaceNodes = mesh.edges.shape[1]
	# print nOfFaceNodes
	nodesDBC = np.zeros(nOfDirichletFaces*nOfFaceNodes,dtype=np.int64)
	displacementDBC = np.zeros((nOfDirichletFaces*nOfFaceNodes, 2))
	indexNode = np.arange(nOfFaceNodes)

	tt = 0
	for iDirichletFace in range(nOfDirichletFaces):
		# print iDirichletFace
		# print mesh.edges[iDirichletFace,:]
		nodesDBC[indexNode] = mesh.edges[iDirichletFace,:]
		idNurbs = dirichletFaces[iDirichletFace,2]
		u1 = ProjU[listFaces[iDirichletFace],0]
		u2 = ProjU[listFaces[iDirichletFace],1]
		t1=time()
		uEq = CurveEquallySpacedPoints(nurbs[idNurbs], u1, u2, nOfFaceNodes, lengthTOL=1e-06)
		# print uEq
		tt += (time()-t1)

		xEq = np.zeros((nOfFaceNodes,2))
		for i in range(nOfFaceNodes):
			pt = CurvePoint(nurbs[idNurbs],uEq[i])[0]
			xEq[i,:] = pt[:2]
		# xOld = mesh.points[nodesDBC[indexNode],:]
		# displacementDBC(indexNode,:) = xEq - xOld
		displacementDBC[indexNode,:] = xEq - mesh.points[nodesDBC[indexNode],:]
		# print xOld

		indexNode += nOfFaceNodes
		# print nodesDBC
	print tt
	# print np.sort(nodesDBC)
	# posUnique = np.unique(nodesDBC,return_inverse=True,return_index=True)[1]
	posUnique = np.unique(nodesDBC,return_index=True)[1]
	# print posUnique.shape
	nodesDBC = nodesDBC[posUnique]
	# print nodesDBC
	uDBC = displacementDBC[posUnique,:]
	nOfDBCnodes = nodesDBC.shape[0]
	DBCmatrix = np.zeros((nsd*nOfDBCnodes, 2))

	for kNode in range(nOfDBCnodes):
		iNode = nodesDBC[kNode]
		displacement = uDBC[kNode,:]
		DBCmatrix[kNode,:] = [iNode, displacement[0]]
		DBCmatrix[kNode+nOfDBCnodes,:] = [iNode + mesh.points.shape[0],displacement[1]]

	# print DBCmatrix
	return DBCmatrix







def CurveEquallySpacedPoints(aNurbs, u1, u2, nPoints, lengthTOL=1e-06):
	nOfMaxIterations = 1000
	uEq = np.zeros(nPoints,dtype=np.float64)
	uEq[0] = u1
	uEq[nPoints-1] = u2
	lengthU = np.abs(u2-u1)
	# from time import time 
	# t1=time()
	length = CurveLengthAdaptive(aNurbs, u1, u2, lengthTOL)
	# print length, u1,u2
	# print time()-t1
	lengthSub = float(length)/(float(nPoints)-1.0)
	# print lengthSub


	uGuess = np.linspace(u1,u2,nPoints)
	nOfIntPoints = nPoints-2

	for iIntPoints in range(nOfIntPoints):
		currentLength = (iIntPoints+1)*lengthSub
		a = u1
		b = u2
		# INITIAL GUESS 
		uOld = 0.5*(uGuess[iIntPoints] + uGuess[iIntPoints+2])
		length = CurveLengthAdaptive(aNurbs, u1 , uOld, lengthTOL)
		# print uOld,length

		for niter in range(nOfMaxIterations):
			# print length, currentLength
			f = length - currentLength
			# print f
			# import sys; sys.exit(0)
			if np.abs(f) < lengthTOL:
				uEq[iIntPoints+1] = uOld
				break
			elif f>0.0:
				b=uOld
			else:
				a=uOld

			uEq[iIntPoints+1] = 0.5*(a+b)

			length = CurveLengthAdaptive(aNurbs, u1 , uEq[iIntPoints+1], lengthTOL)
			errU = np.abs(uOld-uEq[iIntPoints+1])/lengthU
			errF = np.abs(currentLength - length)/currentLength
			if errU<lengthTOL and errF < lengthTOL:
				break
			uOld = uEq[iIntPoints+1]
			# print niter

	return uEq






def TotalCurveLength(c,res=2000):
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


def CurveLength(c,u1,u2,res=500):
	"""Length of NURBS curve between parametric points u1 and u2"""
	
	u=np.linspace(u1,u2,res)
	C=np.zeros((u.shape[0],3))
	L=0.
	for i in range(u.shape[0]):
		# C[i,:] =  bsp_local.Evaluate1(c['degree'],c['U'],c['Pw'],u[i])
		C[i,:] =  CurvePoint(c,u[i])[0][:3]
	for i in range(C.shape[0]-1):
		L += np.linalg.norm(C[i+1,:]-C[i,:])

	return L


def CurveLength_Gauss(c,u1,u2,res=500):
	"""Length of NURBS curve between parametric points u1 and u2 using Gaussian quadrature
	but does not take into account kinks"""
	
	u=np.linspace(u1,u2,res)
	C=np.zeros((u.shape[0],3))
	L=0.
	for i in range(u.shape[0]):
		# C[i,:] =  bsp_local.Evaluate1(c['degree'],c['U'],c['Pw'],u[i])
		C[i,:] =  CurvePoint(c,u[i])[0][:3]
	for i in range(C.shape[0]-1):
		L += np.linalg.norm(C[i+1,:]-C[i,:])

	return L


def CurveLengthAdaptive(aNurbs,u1,u2,lengthTOL):
	
	L=None
	kMax = 6
	# DECREASE LENGTHTOL FOR MORE COMPLEX CURVES
	L0 = CurveLength(aNurbs,u1,u2,200)

	for k in range(2,kMax):
		L1 = CurveLength(aNurbs,u1,u2,k*200)
		errInt = np.abs(1.0*(L1 - L0)/L1)
		if errInt < lengthTOL:
			L = L1
			break
		L0 = L1 

	if k==kMax+1:
		warn('Tolerance has not been achieved')
		L = L1
	# print L0, CurveLength(aNurbs,0,0.5)

	return L






def projectNodeNurbsBoundary(x,nurbs,nsd):
	# NO OF NURBS CURVES/SURFACES
	nOfNurbs = len(nurbs)
	# print nOfNurbs

	d = 1e10
	for iNurbs in range(nOfNurbs):
		if nsd==2:
			u,pu = nurbsCurvePointProjection(nurbs[iNurbs], x)
		elif nsd==3:
			raise NotImplementedError('3D verion not implemented')
		dist = np.linalg.norm(pu[:nsd]-x)
		# print dist
		if dist < d:
			nurbsProj = iNurbs
			xProj = pu[:nsd]
			uProj = u 
			d = dist

	return xProj, uProj, nurbsProj




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
		n = 1000#00

		# print p
		# if p.shape[0] == 2:
			# p[3]=0

		# pIni = nurbsCurvePoint(nurbs, nurbs.iniParam)
		# CHECK THIS
		# pIni = nurbs['points'][0]  
		# pEnd = nurbs['points'][-1]
		# print dir(bspline)
		# degree=nurbs['degree'] # SHIFT THIS TO THE MAIN CALL
		# pIni = bsp.Evaluate1(p,nurbs['U'],nurbs['Pw'],nurbs['start'])[0,:3]
		# pEnd = bsp.Evaluate1(p,nurbs['U'],nurbs['Pw'],nurbs['end'])[0,:3]
		pIni = CurvePoint(nurbs,nurbs['start'])[0]
		pEnd = CurvePoint(nurbs,nurbs['end'])[0]
		# pIni = bsp.evaluate1(p,nurbs['U'],nurbs['Pw'],nurbs['start'])[0,:3]
		# pEnd = bsp.evaluate1(p,nurbs['U'],nurbs['Pw'],nurbs['end'])[0,:3]
		# print pIni, pEnd

		# CurvePoint(nurbs,nurbs['start'])
		# print len(nurbs)
		# print pIni, pEnd
		peroidic = 0
		if np.linalg.norm(pIni - pEnd) < tol1:
			peroidic = 1

		d = float('Inf') 
		us = np.linspace(nurbs['start'], nurbs['end'], n)
		# print us
		# print us[-2]
		for i in range(n):

			# q = bsp.Evaluate1(p,nurbs['U'],nurbs['Pw'],us[i]); 	q = q[0,:3]/q[0,-1] 
			q = CurvePoint(nurbs,us[i])[0]
			# print p,q
			dNew = np.linalg.norm(p-q)
			if dNew < d:
				u = us[i]
				d = dNew

		# print u 
		dMin = d
		Cu = CurvePoint(nurbs, u)[0]
		dCu = CurveDersPoints(nurbs,u)
		d2Cu = CurveSecondDersPoints(nurbs,u)
		# print Cu,dCu,d2Cu

		for niter in range(nMaxIter):
			cond1 = Cu - p 
			# print cond1
			if np.linalg.norm(cond1) < tol1:
				break
			cond2 = np.linalg.norm(np.dot(dCu,cond1)/(np.linalg.norm(dCu)*np.linalg.norm(cond1)))
			if cond2 < tol2:
				break 

			uNew = u - np.dot(dCu,(Cu-p))/(np.dot(d2Cu,(Cu-p))+np.sum(dCu**2))
			# print uNew, np.sum(dCu**2)
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
			# print cond3
			if cond3 < tol1:
				break
			u = uNew
			Cu = CurvePoint(nurbs, u)[0]
			dCu = CurveDersPoints(nurbs,u)
			d2Cu = CurveSecondDersPoints(nurbs,u)
			# print Cu,dCu,d2Cu
			# print cond1
			# if niter==1:
				# print Cu,p

			if np.linalg.norm(cond1) < dMin:
				dMin = np.linalg.norm(cond1)

		if niter == nMaxIter:
			warn('Convergence not achieved for point projection')

		U = np.unique(nurbs['U'][0])  # tip: Completely avoid this, it is okay to loop over all knots than finding unique and calling CurvePoint on a loop
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

		# import sys
		# sys.exit("STOPPED")

	







def CurvePoint(nurbs,u,homogeneous = True):

	if nurbs['U'][0].shape[0] > 0:
		# pt = bsp.Evaluate1(nurbs['degree'],nurbs['U'],nurbs['Pw'],u)
		pt = bsp_local.Evaluate1(nurbs['degree'],nurbs['U'],nurbs['Pw'],u)
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





# def CurvePoint(nurbs,u):
# 	# print nurbs['U']
# 	if len(nurbs['U'][0]) > 0:
# 		# p = np.where(nurbs['U'][0]==nurbs['U'][0][0])[0].shape[0] - 1 # if p is not known this is how you find it
# 		p=2
# 		span = bsp.FindSpan(p,nurbs['U'][0],u) # bsp span is one shorter than Ruben's. maybe a Python 0 indexing thing? becareful though
# 		N = bsp.EvalBasisFuns(p,nurbs['U'][0],u,span)
# 		pt = np.zeros(4)
# 		# print N,u,span
# 		for i in range(p+1):
# 			# print nurbs['Pw'][span-p+i,:], N[i]
# 			pt = pt + N[i]*nurbs['Pw'][span-p+i,:]
# 	else:
# 		pt=np.zeros(3)

# 	return pt 

# 	# print 