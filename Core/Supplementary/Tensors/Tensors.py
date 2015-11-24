import numpy as np 
from Core.Supplementary.CythonBuilds.voigt_sym.voigt_sym import voigt_sym



def Cross(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

	AB = np.zeros((3,3),dtype=np.float64)
	if A.shape==(3,3) and B.shape==(3,3):
		AB = np.array([
			[ A11*B22 - A12*B21 - A21*B12 + A22*B11, A12*B20 - A10*B22 + A20*B12 - A22*B10, A10*B21 - A11*B20 - A20*B11 + A21*B10],
			[ A02*B21 - A01*B22 + A21*B02 - A22*B01, A00*B22 - A02*B20 - A20*B02 + A22*B00, A01*B20 - A00*B21 + A20*B01 - A21*B00],
			[ A01*B12 - A02*B11 - A11*B02 + A12*B01, A02*B10 - A00*B12 + A10*B02 - A12*B00, A00*B11 - A01*B10 - A10*B01 + A11*B00]
			])
 
	elif A.shape==(2,2) and B.shape==(2,2):
		AB[2,2] = A00*B11 - A01*B10 - A10*B01 + A11*B00

	else:
		raise ValueError('Incompatible dimensions. Input matrices must be either 3x3 or 2x2')

	return AB


def SecondTensor2Vector(A):

	# Check size of the matrix
	if A.shape[0]>3 or A.shape[0]==1 or A.shape[0]!=A.shape[1]:
		raise ValueError('Only square 2x2 and 3x3 matrices can be transformed to Voigt vector form')
	if A.shape[0]==3:
		# Matrix is symmetric
		vecA = np.array([
			A[0,0],A[1,1],A[2,2],A[0,1],A[0,2],A[1,2]
			])
		# Check for symmetry
		if ~np.allclose(A.T, A, rtol=1e-12,atol=1e-15):
			# Matrix is non-symmetric
			vecA = np.array([
				A[0,0],A[1,1],A[2,2],A[0,1],A[0,2],A[1,2],A[1,0],A[2,0],A[2,1]
				])
	elif A.shape[0]==2:
		# Matrix is symmetric
		vecA = np.array([
			A[0,0],A[1,1],A[0,1]
			])
		# Check for symmetry
		if ~np.allclose(A.T, A, rtol=1e-12,atol=1e-15):
			# Matrix is non-symmetric
			vecA = np.array([
				A[0,0],A[1,1],A[0,1],A[1,0]
				])

	return vecA


def Voigt(A,sym=0):
	# Given a 4th order tensor A, puts it in 6x6 format
	# Given a 3rd order tensor A, puts it in 3x6 format (symmetric wrt the first two indices)
	# Given a 2nd order tensor A, puts it in 1x6 format

	# sym returns the symmetrised tensor (only for 3rd and 4th order). Switched off by default.

	# GET THE DIMESNIONS
	if len(A.shape)==4:
		# GIVEN TENSOR IS FOURTH ORDER
		# C=A
		if sym:
			VoigtA = voigt_sym(A)

		# 	if C.shape[0]==3:
		# 		VoigtA = 0.5*np.array([
		# 			[2*C[0,0,0,0],2*C[0,0,1,1],2*C[0,0,2,2],C[0,0,0,1]+C[0,0,1,0],C[0,0,0,2]+C[0,0,2,0],C[0,0,1,2]+C[0,0,2,1]],
		# 			[0			 ,2*C[1,1,1,1],2*C[1,1,2,2],C[1,1,0,1]+C[1,1,1,0],C[1,1,0,2]+C[1,1,2,0],C[1,1,1,2]+C[1,1,2,1]],
		# 			[0			 ,0			  ,2*C[2,2,2,2],C[2,2,0,1]+C[2,2,1,0],C[2,2,0,2]+C[2,2,2,0],C[2,2,1,2]+C[2,2,2,1]],
		# 			[0			 ,0			  ,0		   ,C[0,1,0,1]+C[0,1,1,0],C[0,1,0,2]+C[0,1,2,0],C[0,1,1,2]+C[0,1,2,1]],
		# 			[0			 ,0			  ,0		   ,0					 ,C[0,2,0,2]+C[0,2,2,0],C[0,2,1,2]+C[0,2,2,1]],
		# 			[0			 ,0			  ,0		   ,0					 ,0					   ,C[1,2,1,2]+C[1,2,2,1]]
		# 			])

		# 	else:
		# 		VoigtA = 0.5*np.array([
		# 			[2*C[0,0,0,0],2*C[0,0,1,1],C[0,0,0,1]+C[0,0,1,0]],
		# 			[0			 ,2*C[1,1,1,1],C[1,1,0,1]+C[1,1,1,0]],
		# 			[0			 ,0			  ,C[0,1,0,1]+C[0,1,1,0]]
		# 			])

		# 	VoigtA = VoigtA+VoigtA.T 
		# 	for i in range(0,VoigtA.shape[0]):
		# 		VoigtA[i,i] = VoigtA[i,i]/2.0

			

		else:
			C=A
			if C.shape[0]==3:
				VoigtA = np.array([
					[C[0,0,0,0],C[0,0,1,1],C[0,0,2,2],C[0,0,0,1],C[0,0,0,2],C[0,0,1,2]],
					[C[1,1,0,0],C[1,1,1,1],C[1,1,2,2],C[1,1,0,1],C[1,1,0,2],C[1,1,1,2]],
					[C[2,2,0,0],C[2,2,1,1],C[2,2,2,2],C[2,2,0,1],C[2,2,0,2],C[2,2,1,2]],
					[C[0,1,0,0],C[0,1,1,1],C[0,1,2,2],C[0,1,0,1],C[0,1,0,2],C[0,1,1,2]],
					[C[0,2,0,0],C[0,2,1,1],C[0,2,2,2],C[0,2,0,1],C[0,2,0,2],C[0,2,1,2]],
					[C[1,2,0,0],C[1,2,1,1],C[1,2,2,2],C[1,2,0,1],C[1,2,0,2],C[1,2,1,2]]
					])
			elif C.shape[0]==2:
				VoigtA = np.array([
					[C[0,0,0,0],C[0,0,1,1],C[0,0,0,1]],
					[C[1,1,0,0],C[1,1,1,1],C[1,1,0,1]],
					[C[0,1,0,0],C[0,1,1,1],C[0,1,0,1]]
					])

	elif len(A.shape)==3:
		e=A
		if e.shape[0]==3:
			if ~sym:
				VoigtA = 0.5*np.array([
					[2.*e[0,0,0],2.*e[1,0,0],2.*e[2,0,0]],
					[2.*e[0,1,1],2.*e[1,1,1],2.*e[2,1,1]],
					[2.*e[0,2,2],2.*e[1,2,2],2.*e[2,2,2]],
					[e[0,0,1]+e[0,1,0],e[1,0,1]+e[1,1,0],e[2,0,1]+e[2,1,0]],
					[e[0,0,2]+e[0,2,0],e[1,0,2]+e[1,2,0],e[2,0,2]+e[2,2,0]],
					[e[0,1,2]+e[0,2,1],e[1,1,2]+e[1,2,1],e[2,1,2]+e[2,2,1]]
					])
			else:
				VoigtA = np.array([
					[e[0,0,0],e[1,0,0],e[2,0,0]],
					[e[0,1,1],e[1,1,1],e[2,1,1]],
					[e[0,2,2],e[1,2,2],e[2,2,2]],
					[e[0,0,1],e[1,0,1],e[2,0,1]],
					[e[0,0,2],e[1,0,2],e[2,0,2]],
					[e[0,1,2],e[1,1,2],e[2,1,2]]
					])
		elif e.shape[0]==2:
			if ~sym:
				VoigtA = 0.5*np.array([
					[2.*e[0,0,0],2.*e[1,0,0]],
					[2.*e[0,1,1],2.*e[1,1,1]],
					[e[0,0,1]+e[0,1,0],e[1,0,1]+e[1,1,0]]
					])
			else:
				VoigtA = np.array([
					[e[0,0,0],e[1,0,0]],
					[e[0,1,1],e[1,1,1]],
					[e[0,0,1],e[1,0,1]]
					])

	elif len(A.shape)==2:
		VoigtA = SecondTensor2Vector(A)

	else:
		VoigtA = np.array([])

	return VoigtA




def UnVoigt(v):
	A = []
	if len(v.shape)==2:
		if v.shape[1]>1:
			pass
			# DO IT LATER FOR shape>1
		elif v.shape[1]==1:
			# VECTORS TO SYMMETRIC 2ND ORDER TENSORS
			if v.shape[0]==6:
				A = np.array([
					[v[0,0],v[3,0],v[4,0]],
					[v[3,0],v[1,0],v[5,0]],
					[v[4,0],v[5,0],v[2,0]]
					])
			elif v.shape[0]==3:
				A = np.array([
					[v[0,0],v[2,0]],
					[v[2,0],v[1,0]]
					])

	elif len(v.shape)==1:
		# VECTORS TO SYMMETRIC 2ND ORDER TENSORS
		if v.shape[0]==6:
			A = np.array([
				[v[0],v[3],v[4]],
				[v[3],v[1],v[5]],
				[v[4],v[5],v[2]]
				])
		elif v.shape[0]==3:
			A = np.array([
				[v[0],v[2]],
				[v[2],v[1]]
				])

	return A 






def IncrementallyLinearisedStress(Stress_k,H_Voigt_k,I,strain,Gradu):
	# IN PRINCIPLE WE NEED GRADU AND NOT STRAIN FOR V_strain
	V_strain =  Voigt(strain)[:,None]
					# STRESS 						HESSSIAN 'I_W:GRADU'
	return np.dot(Stress_k,(I+strain)) + UnVoigt( np.dot(H_Voigt_k,V_strain) )
	









########################################################################################################
# Note that these matrices are all symmetrised
def AijBkl(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]
	
	Tens = 1.0*np.array([ 
		[ A00*B00, A00*B11, A00*B22, A00*B01, A00*B02, A00*B12],
		[ A11*B00, A11*B11, A11*B22, A11*B01, A11*B02, A11*B12],
		[ A22*B00, A22*B11, A22*B22, A22*B01, A22*B02, A22*B12],
		[ A01*B00, A01*B11, A01*B22, A01*B01, A01*B02, A01*B12],
		[ A02*B00, A02*B11, A02*B22, A02*B01, A02*B02, A02*B12],
		[ A12*B00, A12*B11, A12*B22, A12*B01, A12*B02, A12*B12]
		])


	return Tens 


def AikBjl(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

	Tens = 1.0*np.array([
		[ A00*B00, A01*B01, A02*B02, A00*B01, A00*B02, A01*B02],
		[ A10*B10, A11*B11, A12*B12, A10*B11, A10*B12, A11*B12],
		[ A20*B20, A21*B21, A22*B22, A20*B21, A20*B22, A21*B22],
		[ A00*B10, A01*B11, A02*B12, A00*B11, A00*B12, A01*B12],
		[ A00*B20, A01*B21, A02*B22, A00*B21, A00*B22, A01*B22],
		[ A10*B20, A11*B21, A12*B22, A10*B21, A10*B22, A11*B22]
		])


	return Tens


def AilBjk(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

	Tens = 1.0*np.array([
		[ A00*B00, A01*B01, A02*B02, A01*B00, A02*B00, A02*B01],
		[ A10*B10, A11*B11, A12*B12, A11*B10, A12*B10, A12*B11],
		[ A20*B20, A21*B21, A22*B22, A21*B20, A22*B20, A22*B21],
		[ A00*B10, A01*B11, A02*B12, A01*B10, A02*B10, A02*B11],
		[ A00*B20, A01*B21, A02*B22, A01*B20, A02*B20, A02*B21],
		[ A10*B20, A11*B21, A12*B22, A11*B20, A12*B20, A12*B21]
		])


	return Tens


# A TENSOR AND A VECTOR
def AijUk(A,U):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[ A00*U0, A01*U1, A02*U2, A00*U1, A00*U2, A01*U2],
		[ A10*U0, A11*U1, A12*U2, A10*U1, A10*U2, A11*U2],
		[ A20*U0, A21*U1, A22*U2, A20*U1, A20*U2, A21*U2]
		])

	return Tens


def AikUj(A,U):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[ A00*U0, A01*U1, A02*U2, A01*U0, A02*U0, A02*U1],
		[ A10*U0, A11*U1, A12*U2, A11*U0, A12*U0, A12*U1],
		[ A20*U0, A21*U1, A22*U2, A21*U0, A22*U0, A22*U1]
		])


	return Tens


def UiAjk(U,A):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[ A00*U0, A11*U0, A22*U0, A01*U0, A02*U0, A12*U0],
		[ A00*U1, A11*U1, A22*U1, A01*U1, A02*U1, A12*U1],
		[ A00*U2, A11*U2, A22*U2, A01*U2, A02*U2, A12*U2]
		])


	return Tens






# class VoigtTensors(object):
# 	"""docstring for VoigtTensors"""
# 	def __init__(self, arg):
# 		super(VoigtTensors, self).__init__()
# 		self.arg = arg
# 	AijBkl = AijBkl
# 	AikBjl = AikBjl
# 	AilBjk = AilBjk

# 	AijUkVl = AijUkVl
# 	AikUjVl = AikUjVl
# 	AilUjVk = AilUjVk
# 	UiVjAkl = UiVjAkl
# 	UiVkAjl = UiVkAjl
# 	UiVlAjk = UiVlAjk

# 	AijUk = AijUk
# 	AikUj = AikUj
# 	UiAjk = UiAjk

# 	UiVjXkYl = UiVjXkYl
# 	UiVjXk = UiVjXk


