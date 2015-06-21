import imp, os
PathOneD = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
OneD = imp.load_source('OneDimensional',PathOneD+'/OneDimensional/BasisFunctions.py')

import numpy as np 
# import scipy as sp 
from Core.FiniteElements.GetCounterClockwiseIndices import GetCounterClockwiseIndices


def LagrangeGaussLobatto(C,zeta,eta,beta,arrange=1):

	# Bases = np.zeros(((C+2)**3,1))
	Neta = np.zeros((C+2,1));	Nzeta = np.zeros((C+2,1)); Nbeta = np.zeros((C+2,1))

	Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
	Neta[:,0] =  OneD.LagrangeGaussLobatto(C,eta)[0]
	Nbeta[:,0] =  OneD.LagrangeGaussLobatto(C,beta)[0]

	if arrange==0:
		Bases = np.zeros((C+2,C+2,C+2))
		for i in range(0,C+2):
			Bases[:,:,i] = Nbeta[i]*np.dot(Nzeta,Neta.T)
		Bases = Bases.reshape((C+2)**3,1)
	elif arrange==1:
		Bases = np.zeros(((C+2)**3,1))
		# Arrange in counterclockwise
		zeta_index, eta_index = GetCounterClockwiseIndices(C)
		TBases = np.dot(Nzeta,Neta.T)
		counter=0
		for j in range(0,C+2):
			for i in range(0,(C+2)**2):
				Bases[counter] = Nbeta[j]*TBases[zeta_index[i],eta_index[i]]
				counter+=1


	# Coordinates of nodes at parent element
	epszeta = OneD.LagrangeGaussLobatto(C,zeta)[2]
	epseta = OneD.LagrangeGaussLobatto(C,eta)[2]
	# epsbeta = OneD.LagrangeGaussLobatto(C,beta)[2]
	eps  = np.zeros((1,2))
	for i in range(0,epszeta.shape[0]):
		for j in range(0,epseta.shape[0]):
			eps = np.concatenate((eps, np.array([epszeta[i],epseta[j]]).reshape(1,2)),axis=0)
	eps = np.delete(eps,0,0)

	# BE VERY CAREFULL ABOUT THIS
	eps[:,0] = eps[zeta_index,1]
	eps[:,1] = eps[eta_index,1]

	# Make it 3D
	eps_3d = np.zeros(((C+2)*eps.shape[0],3))
	for i in range(0,C+2):
		size1 = i*eps.shape[0]; 	size2 = (i+1)*eps.shape[0]
		eps_3d[size1:size2,0:2] = eps[:,0:2]; eps_3d[size1:size2,2] = np.sort(epszeta)[i] 

	return Bases, eps_3d

def GradLagrangeGaussLobatto(C,zeta,eta,beta,arrange=1):
	# This routine computes stable higher order Lagrangian bases with Gauss-Lobatto-Legendre points
	# Refer to: Spencer's Spectral hp elements for details

	# Allocate
	gBases = np.zeros(((C+2)**3,3))
	Nzeta = np.zeros((C+2,1));	Neta = np.zeros((C+2,1));	Nbeta = np.zeros((C+2,1))
	gNzeta = np.zeros((C+2,1));	gNeta = np.zeros((C+2,1));  gNbeta = np.zeros((C+2,1))
	# Compute each from one-dimensional bases
	Nzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[0]
	Neta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[0]
	Nbeta[:,0] = OneD.LagrangeGaussLobatto(C,beta)[0]
	gNzeta[:,0] = OneD.LagrangeGaussLobatto(C,zeta)[1]
	gNeta[:,0] = OneD.LagrangeGaussLobatto(C,eta)[1]
	gNbeta[:,0] = OneD.LagrangeGaussLobatto(C,beta)[1]

	# Ternsorial product
	if arrange==0:
		gBases1 = np.zeros((C+2,C+2,C+2)); gBases2 = np.zeros((C+2,C+2,C+2)); gBases3 = np.zeros((C+2,C+2,C+2))
		for i in range(0,C+2):
			gBases1[:,:,i] = Nbeta[i]*np.dot(gNzeta,Neta.T)
			gBases2[:,:,i] = Nbeta[i]*np.dot(Nzeta,gNeta.T)
			gBases3[:,:,i] = gNbeta[i]*np.dot(Nzeta,Neta.T)
		gBases1 = gBases1.reshape((C+2)**3,1)
		gBases2 = gBases2.reshape((C+2)**3,1)
		gBases3 = gBases3.reshape((C+2)**3,1)
		gBases[:,0]=gBases1
		gBases[:,1]=gBases2
		gBases[:,2]=gBases3

	elif arrange==1:
		# Arrange in counterclockwise
		zeta_index, eta_index = GetCounterClockwiseIndices(C)
		gBases1 = np.dot(gNzeta,Neta.T)
		gBases2 = np.dot(Nzeta,gNeta.T)
		gBases3 = np.dot(Nzeta,Neta.T)
		counter=0
		for j in range(0,C+2):
			for i in range(0,(C+2)**2):
				gBases[counter,0] = Nbeta[j]*gBases1[zeta_index[i],eta_index[i]]
				gBases[counter,1] = Nbeta[j]*gBases2[zeta_index[i],eta_index[i]]
				gBases[counter,2] = gNbeta[j]*gBases3[zeta_index[i],eta_index[i]]
				counter+=1


	return gBases