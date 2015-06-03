import numpy as np 
import imp, os

pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
TwoD = imp.load_source('QuadLagrangeGaussLobatto',pwd+'/Core/InterpolationFunctions/TwoDimensional/Quad/QuadLagrangeGaussLobatto.py')
ThreeD = imp.load_source('HexLagrangeGaussLobatto',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Hexahedral/HexLagrangeGaussLobatto.py')
# OneD = imp.load_source('OneDimensional',pwd+'/Core/InterpolationFunctions/OneDimensional/BasisFunctions.py')
# from Core.InterpolationFunctions.TwoDimensional.Tri.hpModal import hpBases, GradhpBases
# Modal Bases
# Tri = imp.load_source('hpModalTri',pwd+'/Core/InterpolationFunctions/TwoDimensional/Tri/hpModal.py')
# Tet = imp.load_source('hpModalTet',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Tetrahedral/hpModal.py')
# Nodal Bases
Tri = imp.load_source('hpNodalTri',pwd+'/Core/InterpolationFunctions/TwoDimensional/Tri/hpNodal.py')
Tet = imp.load_source('hpNodalTet',pwd+'/Core/InterpolationFunctions/ThreeDimensional/Tetrahedral/hpNodal.py')

def GetBases(C,Quadrature,info):

	w = Quadrature.weights
	z = Quadrature.points

	ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]
	if info=='tri':
		p=C+1
		ns = (p+1)*(p+2)/2
		Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
		gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
		gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)
	elif info=='quad':
		ns = (C+2)**2
		Basis = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)
		gBasisx = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)
		gBasisy = np.zeros((ns,z.shape[0]*z.shape[0]),dtype=np.float64)


	if info == 'quad':
		counter = 0
		for i in range(0,z.shape[0]):
			for j in range(0,z.shape[0]):
				ndummy = TwoD.LagrangeGaussLobatto(C,z[i],z[j])[0]
				Basis[:,counter] = ndummy[:,0]
				dummy = TwoD.GradLagrangeGaussLobatto(C,z[i],z[j])
				gBasisx[:,counter] = dummy[:,0]
				gBasisy[:,counter] = dummy[:,1]
				counter+=1
	elif info == 'tri':
		for i in range(0,w.shape[0]):
			ndummy, dummy = Tri.hpBases(C,z[i,0],z[i,1],Quadrature.Opt)
			Basis[:,i] = ndummy
			gBasisx[:,i] = dummy[:,0]
			gBasisy[:,i] = dummy[:,1]



	class Domain(object):
		"""docstring for Domain"""
		Bases = Basis
		gBasesx = gBasisx
		gBasesy = gBasisy
		gBasesz = np.zeros(gBasisx.shape)


	return Domain



def GetBases3D(C,Quadrature,info):

	# ndim = general_data.ndim
	ndim = 3

	w = Quadrature.weights
	z = Quadrature.points

	ns=[]; Basis=[]; gBasisx=[]; gBasisy=[]; gBasisz=[]
	if info=='hex':
		ns = (C+2)**ndim
		Basis = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
		gBasisx = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
		gBasisy = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
		gBasisz = np.zeros((ns,(z.shape[0])**ndim),dtype=np.float64)
	elif info=='tet':
		p=C+1
		ns = (p+1)*(p+2)*(p+3)/6
		Basis = np.zeros((ns,w.shape[0]),dtype=np.float64)
		gBasisx = np.zeros((ns,w.shape[0]),dtype=np.float64)
		gBasisy = np.zeros((ns,w.shape[0]),dtype=np.float64)
		gBasisz = np.zeros((ns,w.shape[0]),dtype=np.float64)	
	

	if info=='hex':
		counter = 0
		for i in range(0,w.shape[0]):
			for j in range(0,w.shape[0]):
				for k in range(0,w.shape[0]):
					ndummy = ThreeD.LagrangeGaussLobatto(C,z[i],z[j],z[k])[0]
					dummy = ThreeD.GradLagrangeGaussLobatto(C,z[i],z[j],z[k])

					Basis[:,counter] = ndummy[:,0]
					gBasisx[:,counter] = dummy[:,0]
					gBasisy[:,counter] = dummy[:,1]
					gBasisz[:,counter] = dummy[:,2]
					counter+=1
	elif info=='tet':
		for i in range(0,w.shape[0]):
			ndummy, dummy = Tet.hpBases(C,z[i,0],z[i,1],z[i,2],Quadrature.Opt)
			Basis[:,i] = ndummy
			gBasisx[:,i] = dummy[:,0]
			gBasisy[:,i] = dummy[:,1]
			gBasisz[:,i] = dummy[:,2]


	class Domain(object):
		"""docstring for Domain"""
		Bases = Basis
		gBasesx = gBasisx
		gBasesy = gBasisy
		gBasesz = gBasisz
			

	return Domain



def GetBasesBoundary(C,z,ndim):

	BasisBoundary = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
	gBasisBoundaryx = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
	gBasisBoundaryy = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))
	gBasisBoundaryz = np.zeros(((C+2)**(ndim),(z.shape[0])**(ndim-1),2*ndim))

	# eps = OneD.LagrangeGaussLobatto(C,0)
	eps = np.array([-1.,1.,-1.,1.,-1.,1.])
	

	for k in range(0,eps.shape[0]):
		counter = 0
		for i in range(0,z.shape[0]):
			for j in range(0,z.shape[0]):
				if k==0 or k==1:
					ndummy = ThreeD.LagrangeGaussLobatto(C,eps[k],z[i],z[j])[0]
					BasisBoundary[:,counter,k] = ndummy[:,0]

					dummy = ThreeD.GradLagrangeGaussLobatto(C,eps[k],z[i],z[j])
					gBasisBoundaryx[:,counter,k] = dummy[:,0]
					gBasisBoundaryy[:,counter,k] = dummy[:,1]
					gBasisBoundaryz[:,counter,k] = dummy[:,2]

				elif k==2 or k==3:
					ndummy = ThreeD.LagrangeGaussLobatto(C,z[i],eps[k],z[j])[0]
					BasisBoundary[:,counter,k] = ndummy[:,0]

					dummy = ThreeD.GradLagrangeGaussLobatto(C,z[i],eps[k],z[j])
					gBasisBoundaryx[:,counter,k] = dummy[:,0]
					gBasisBoundaryy[:,counter,k] = dummy[:,1]
					gBasisBoundaryz[:,counter,k] = dummy[:,2]

				elif k==4 or k==5:
					ndummy = ThreeD.LagrangeGaussLobatto(C,z[i],z[j],eps[k])[0]
					BasisBoundary[:,counter,k] = ndummy[:,0]

					dummy = ThreeD.GradLagrangeGaussLobatto(C,z[i],z[j],eps[k])
					gBasisBoundaryx[:,counter,k] = dummy[:,0]
					gBasisBoundaryy[:,counter,k] = dummy[:,1]
					gBasisBoundaryz[:,counter,k] = dummy[:,2]

				
				counter+=1

	class Boundary(object):
		"""docstring for BasisBoundary"""
		def __init__(self, arg):
			super(BasisBoundary, self).__init__()
			self.arg = arg
		Basis  = BasisBoundary
		gBasisx = gBasisBoundaryx
		gBasisy = gBasisBoundaryy
		gBasisz = gBasisBoundaryz


	return Boundary

			