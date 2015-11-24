import numpy as np
from Core.QuadratureRules import GaussQuadrature, QuadraturePointsWeightsTet, QuadraturePointsWeightsTri
from Core.FiniteElements.GetBases import *

def GetBasesAtInegrationPoints(C,norder,QuadratureOpt,MeshType):
	"""Compute interpolation functions at all integration points"""

	ndim = 2
	if MeshType == 'tet' or MeshType == 'hex':
		ndim = 3


	z=[]; w=[]; 

	if MeshType == "quad" or MeshType == "hex":
		z, w = GaussQuadrature(norder,-1.,1.)
		# z, w = GaussQuadrature(MainData.C+MainData.norder,-1.,1.)
	elif MeshType == "tet":
		# zw = QuadraturePointsWeightsTet.QuadraturePointsWeightsTet(norder,QuadratureOpt)
		zw = QuadraturePointsWeightsTet.QuadraturePointsWeightsTet(C+1,QuadratureOpt)
		z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]
	elif MeshType == "tri":
		zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(norder,QuadratureOpt) # PUT C+1 OR HIGHER
		# zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(MainData.C+1,QuadratureOpt) # PUT C+1 OR HIGHER
		# zw = QuadraturePointsWeightsTri.QuadraturePointsWeightsTri(MainData.C+1,QuadratureOpt) # PUT C+4 OR HIGHER
		z = zw[:,:-1]; z=z.reshape(z.shape[0],z.shape[1]); w=zw[:,-1]


	class Quadrature(object):
		"""Quadrature rules"""
		points = z
		weights = w
		Opt = QuadratureOpt




	if MeshType == 'tet' or MeshType == 'hex':
		# GET BASES AT ALL INTEGRATION POINTS (VOLUME)
		Domain = GetBases3D(C,Quadrature,MeshType)
		# GET BOUNDARY BASES AT ALL INTEGRATION POINTS (SURFACE)
		# Boundary = GetBasesBoundary(MainData.C,z,MainData.ndim)
	elif MeshType == 'tri' or MeshType == 'quad':
		# Get basis at all integration points (surface)
		Domain = GetBases(C,Quadrature,MeshType)
		# GET BOUNDARY BASES AT ALL INTEGRATION POINTS (LINE)
		# Boundary = GetBasesBoundary(MainData.C,z,MainData.ndim)
	Boundary = []

	############################################################################
	# from scipy.io import savemat
	# Dict = {'GaussPoints':z,'GaussWeights':w,'Bases':Domain.Bases,'gBasesx':Domain.gBasesx, 'gBasesy':Domain.gBasesy}
	# savemat('/home/roman/Desktop/Bases_P'+str(C+1)+'_Quad_P'+str(norder),Dict)
	# exit(0)


	# COMPUTING GRADIENTS AND JACOBIAN A PRIORI FOR ALL INTEGRATION POINTS
	############################################################################
	Domain.Jm = []; Domain.AllGauss=[]
	if MeshType == 'hex':
		Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim))	
		Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))	
		counter = 0
		for g1 in range(0,w.shape[0]):
			for g2 in range(0,w.shape[0]): 
				for g3 in range(0,w.shape[0]):
					# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
					Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
					Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
					Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

					Domain.AllGauss[counter,0] = w[g1]*w[g2]*w[g3]

					counter +=1

	elif MeshType == 'quad':
		Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]**ndim))	
		Domain.AllGauss = np.zeros((w.shape[0]**ndim,1))	
		counter = 0
		for g1 in range(0,w.shape[0]):
			for g2 in range(0,w.shape[0]): 
				# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
				Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
				Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

				Domain.AllGauss[counter,0] = w[g1]*w[g2]
				counter +=1

	elif MeshType == 'tet':
		Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))	
		Domain.AllGauss = np.zeros((w.shape[0],1))	
		for counter in range(0,w.shape[0]):
			# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
			Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
			Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]
			Domain.Jm[2,:,counter] = Domain.gBasesz[:,counter]

			Domain.AllGauss[counter,0] = w[counter]

	elif MeshType == 'tri':
		Domain.Jm = [];  Domain.AllGauss = []

		Domain.Jm = np.zeros((ndim,Domain.Bases.shape[0],w.shape[0]))	
		Domain.AllGauss = np.zeros((w.shape[0],1))	
		for counter in range(0,w.shape[0]):
			# Gradient Tensor in Parent Element [\nabla_\varepsilon (N)]
			Domain.Jm[0,:,counter] = Domain.gBasesx[:,counter]
			Domain.Jm[1,:,counter] = Domain.gBasesy[:,counter]

			Domain.AllGauss[counter,0] = w[counter]

	return Domain, Boundary, Quadrature