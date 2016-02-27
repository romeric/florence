# First Get the file path for OneDimensional basis since its not initiated properly
import imp, os
import numpy as np 

PathOneD = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
OneD = imp.load_source('OneDimensional',PathOneD+'/OneDimensional/BasisFunctions.py')
# import Florence.InterpolationFunctions.OneDimensional as OneD
from Florence.FiniteElements.GetCounterClockwiseIndices import GetCounterClockwiseIndices


def LagrangeGaussLobatto(C,zeta,eta,arrange=1):
	# This routine computes stable higher order Lagrangian bases with Gauss-Lobatto-Legendre points
	# Refer to: Spencer's Spectral hp elements for details

	# Allocate
	Bases = np.zeros(((C+2)**2,1))
	# Bases = np.zeros((C+2,C+2))
	Neta = np.zeros((C+2,1));	Nzeta = np.zeros((C+2,1))
	# epszeta = np.zeros((C+2,1));	epseta = np.zeros((C+2,1))
	# Compute each from one-dimensional bases
	Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
	Neta[:,0] =  OneD.LagrangeGaussLobatto(C,eta)[0]
	# Ternsorial product
	if arrange==0:
		Bases[:,0] = np.dot(Nzeta,Neta.T).reshape((C+2)**2)
	elif arrange==1:
		# Arrange in counterclockwise - THIS FUNCTION NEEDS TO BE OPTIMISED
		zeta_index, eta_index = GetCounterClockwiseIndices(C)
		TBases = np.dot(Nzeta,Neta.T)
		for i in range(0,(C+2)**2):
			Bases[i] = TBases[zeta_index[i],eta_index[i]]



	# Coordinates of nodes at parent element
	epszeta = OneD.LagrangeGaussLobatto(C,zeta)[2]
	epseta = OneD.LagrangeGaussLobatto(C,eta)[2]
	eps  = np.zeros((1,2))
	for i in range(0,epszeta.shape[0]):
		for j in range(0,epseta.shape[0]):
			eps = np.concatenate((eps, np.array([epszeta[i],epseta[j]]).reshape(1,2)),axis=0)
	eps = np.delete(eps,0,0)

	# BE VERY CAREFULL ABOUT THIS
	eps[:,0] = eps[zeta_index,1]
	eps[:,1] = eps[eta_index,1]



	# check = np.array([
		# 0.25*(1-zeta)*(1-eta),
		# 0.25*(1+zeta)*(1-eta),
		# 0.25*(1+zeta)*(1+eta),
		# 0.25*(1-zeta)*(1+eta)
		# ])
	# print check


	return Bases, eps

def GradLagrangeGaussLobatto(C,zeta,eta,arrange=1):
	# This routine computes stable higher order Lagrangian bases with Gauss-Lobatto-Legendre points
	# Refer to: Spencer's Spectral hp elements for details

	# Allocate
	gBases = np.zeros(((C+2)**2,2))
	Nzeta = np.zeros((C+2,1));	Neta = np.zeros((C+2,1))
	gNzeta = np.zeros((C+2,1));	gNeta = np.zeros((C+2,1))
	# Compute each from one-dimensional bases
	Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
	Neta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[0]
	gNzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[1]
	gNeta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[1]
	# Ternsorial product
	if arrange==0:
		gBases[:,0] = np.dot(gNzeta,Neta.T).reshape((C+2)**2)
		gBases[:,1] = np.dot(Nzeta,gNeta.T).reshape((C+2)**2)
	elif arrange==1:
		# Arrange counterclockwise
		zeta_index, eta_index = GetCounterClockwiseIndices(C)
		gTBases0 = np.dot(gNzeta,Neta.T)
		gTBases1 = np.dot(Nzeta,gNeta.T)

		for i in range(0,(C+2)**2):
			gBases[i,0] = gTBases0[zeta_index[i],eta_index[i]]
			gBases[i,1] = gTBases1[zeta_index[i],eta_index[i]]


	# check =  0.25*np.array([[eta-1.,1-eta,1+eta,-1.-eta],[zeta-1.,-zeta-1.,1+zeta,1-zeta]])


	return gBases





# Bases = QuadLagrangeGaussLobattoBases(0,-1./np.sqrt(3),-1./np.sqrt(3))[0]
# Bases = QuadLagrangeGaussLobattoBases(0,0.26,0.25)[0]
# print Bases[np.array([0,3,1,2])].T
# print Bases[np.array([0,2,3,1])].T
# print Bases.T
# gBases = GradQuadLagrangeGaussLobattoBases(0,-0.5,0.5)
# print gBases

# print BasisFunctions().LagrangeGaussLobattoShapeFunctions(1,-1)[0]
# print Lagrange(2,0)