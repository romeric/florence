import numpy as np 
import numpy.linalg as la

# CHECK THIS ROUTINE
def Mass(MainData,w,LagrangeElemCoords,EulerELemCoords,elem):

	ndim = MainData.ndim
	nvar = MainData.Minimal.nvar

	N = np.zeros((MainData.Domain.Bases.shape[0]*nvar,nvar))
	mass = np.zeros((MainData.Domain.Bases.shape[0]*nvar,MainData.Domain.Bases.shape[0]*nvar))

	# LOOP OVER GAUSS POINTS
	for counter in range(0,Domain.AllGauss.shape[0]):
		# GRADIENT TENSOR IN PARENT ELEMENT [\nabla_\varepsilon (N)]
		Jm = MainData.Domain.Jm[:,:,counter]
		Bases = MainData.Domain.Bases[:,counter]
		# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
		ParentGradientX=np.dot(Jm,LagrangeElemCoords)

		# UPDATE/NO-UPDATE GEOMETRY
		if MainData.GeometryUpdate:
			# MAPPING TENSOR [\partial\vec{X}/ \partial\vec{\varepsilon} (ndim x ndim)]
			ParentGradientx = np.dot(Jm,EulerELemCoords)
		else:
			ParentGradientx = ParentGradientX

		# COMPUTE THE MASS INTEGRAND
		rhoNN = MainData().MassIntegrand(Bases,N,MainData.Minimal,MainData.MaterialArgs)

		if MainData.GeometryUpdate:
			# INTEGRATE MASS
			mass += rhoNN*MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))
			# mass += rhoNN*w[g1]*w[g2]*w[g3]*np.abs(la.det(ParentGradientX))*np.abs(StrainTensors.J)
		else:
			# INTEGRATE MASS
			mass += rhoNN*MainData.Domain.AllGauss[counter,0]*np.abs(la.det(ParentGradientX))

	return mass 