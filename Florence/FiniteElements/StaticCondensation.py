import numpy as np
import numpy.linalg as la
import imp, os

pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
TwoD = imp.load_source('QuadLagrangeGaussLobatto',pwd+'/Florence/InterpolationFunctions/TwoDimensional/Quad/QuadLagrangeGaussLobatto.py')



def StaticCondensation(K,f,C,nvar):
	# First determine which rows and columns should be condensed out i.e interior and exteriors
	eps = TwoD.LagrangeGaussLobatto(C,0,0)[1]
	# eps = TwoD.Lagrange(C,0,0)[1]
	interior_nodes = []
	for i in range(0,eps.shape[0]):
		if (eps[i,0]!=-1 and eps[i,0]!=1) and (eps[i,1]!=-1 and eps[i,1]!=1):
			interior_nodes = np.append(interior_nodes,i)

	allnodes = np.linspace(0,eps.shape[0]-1,eps.shape[0])
	exterior_nodes = np.delete(allnodes,interior_nodes)

	exteriors = []
	for i in range(0,exterior_nodes.shape[0]):
		for j in range(0,nvar):
			exteriors = np.append(exteriors,nvar*exterior_nodes[i]+j)

	allvar = np.linspace(0,K.shape[0]-1,K.shape[0])
	interiors = np.delete(allvar,exteriors)

	interiors = np.array(interiors,dtype=int)
	exteriors = np.array(exteriors,dtype=int)


	# Now perform static condensation
	Kee = K[exteriors,:][:,exteriors]
	Kei = K[exteriors,:][:,interiors]
	Kie = K[interiors,:][:,exteriors]
	Kii = K[interiors,:][:,interiors]

	Kcondensed = Kee-Kei.dot(np.dot(la.inv(Kii),Kie))

	# RHS 
	fe = f[exteriors]
	fi = f[interiors]
	fcondensed = fe-Kei.dot(np.dot(la.inv(Kii),fi))

	return Kcondensed, fcondensed







# Edge and Interior
def StaticCondensation_EdgeInterior(K,f,C,nvar):
	# First determine which rows and columns should be condensed out i.e interior and exteriors
	eps = TwoD.LagrangeGaussLobatto(C,0,0)[1]
	# eps = TwoD.Lagrange(C,0,0)[1]
	exterior_nodes = []
	for i in range(0,eps.shape[0]):
		# if np.allclose(eps[i,0],-1) and np.allclose(eps[i,1],-1):
		if eps[i,0]==-1 and eps[i,1]==-1:
			exterior_nodes = np.append(exterior_nodes,i)
		elif eps[i,0]==1 and eps[i,1]==-1:
			exterior_nodes = np.append(exterior_nodes,i)
		elif eps[i,0]==1 and eps[i,1]==1:
			exterior_nodes = np.append(exterior_nodes,i)
		elif eps[i,0]==-1 and eps[i,1]==1:
			exterior_nodes = np.append(exterior_nodes,i)

	allnodes = np.linspace(0,eps.shape[0]-1,eps.shape[0])
	interior_nodes = np.delete(allnodes,exterior_nodes)

	exteriors = []
	for i in range(0,exterior_nodes.shape[0]):
		for j in range(0,nvar):
			exteriors = np.append(exteriors,nvar*exterior_nodes[i]+j)

	allvar = np.linspace(0,K.shape[0]-1,K.shape[0])
	interiors = np.delete(allvar,exteriors)

	interiors = np.array(interiors,dtype=int)
	exteriors = np.array(exteriors,dtype=int)


	# Now perform static condensation
	Kee = K[exteriors,:][:,exteriors]
	Kei = K[exteriors,:][:,interiors]
	Kie = K[interiors,:][:,exteriors]
	Kii = K[interiors,:][:,interiors]

	Kcondensed = Kee-Kei.dot(np.dot(la.inv(Kii),Kie))

	# RHS 
	fe = f[exteriors]
	fi = f[interiors]
	fcondensed = fe-Kei.dot(np.dot(la.inv(Kii),fi))

	return Kcondensed, fcondensed
