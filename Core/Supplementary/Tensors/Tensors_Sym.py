
#################################################################################################################
# 						THESE TENSORS ARE SYMMETRISED WHICH MAKE THEM NOT APPLICABLE TO MOST SCENARIOS 
#################################################################################################################
import numpy as np 

def SecondTensor2Vector(A):

	# Check size of the matrix
	if A.shape[0]>3 or A.shape[0]==1 or A.shape[0]!=A.shape[1]:
		raise ValueError('Only square 2x2 and 3x3 matrices can be transformed to Voigt vector form')
	if A.shape[0]==3:
		# Matrix is symmetric
		vecA = np.array([
			A[0,0],A[1,1],A[2,2],A[0,1],A[0,2],A[1,2]
			])
		# print np.allclose(A.T,A,rtol=1e-10,atol=1e-15)
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



# Note that these matrices are all symmetrised
def AijBkl(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]
	
	Tens = 1.0*np.array([ 
		[                   A00*B00,                   A00*B11,                   A00*B22, (A00*B01)/2 + (A00*B10)/2, (A00*B02)/2 + (A00*B20)/2, (A00*B12)/2 + (A00*B21)/2],
		[                   A00*B11,                   A11*B11,                   A11*B22, (A11*B01)/2 + (A11*B10)/2, (A11*B02)/2 + (A11*B20)/2, (A11*B12)/2 + (A11*B21)/2],
		[                   A00*B22,                   A11*B22,                   A22*B22, (A22*B01)/2 + (A22*B10)/2, (A22*B02)/2 + (A22*B20)/2, (A22*B12)/2 + (A22*B21)/2],
		[ (A00*B01)/2 + (A00*B10)/2, (A11*B01)/2 + (A11*B10)/2, (A22*B01)/2 + (A22*B10)/2, (A01*B01)/2 + (A01*B10)/2, (A01*B02)/2 + (A01*B20)/2, (A01*B12)/2 + (A01*B21)/2],
		[ (A00*B02)/2 + (A00*B20)/2, (A11*B02)/2 + (A11*B20)/2, (A22*B02)/2 + (A22*B20)/2, (A01*B02)/2 + (A01*B20)/2, (A02*B02)/2 + (A02*B20)/2, (A02*B12)/2 + (A02*B21)/2],
		[ (A00*B12)/2 + (A00*B21)/2, (A11*B12)/2 + (A11*B21)/2, (A22*B12)/2 + (A22*B21)/2, (A01*B12)/2 + (A01*B21)/2, (A02*B12)/2 + (A02*B21)/2, (A12*B12)/2 + (A12*B21)/2]
		])
 

	# This approach gives a non-symmetric 6x6 tensor
	# Tens = []
	# if A.shape[0]==3:
	# 	if np.allclose(A.T, A, rtol=1e-12,atol=1e-15) and np.allclose(B.T, B, rtol=1e-12,atol=1e-15):
	# 		Tens = np.dot(SecondTensor2Vector(A).reshape(6,1),SecondTensor2Vector(B).reshape(1,6))
	# 	else:
	# 		Tens = np.dot(SecondTensor2Vector(A).reshape(9,1),SecondTensor2Vector(B).reshape(1,9))
	# elif A.shape[0]==2:
	# 	if np.allclose(A.T, A, rtol=1e-12,atol=1e-15) and np.allclose(B.T, B, rtol=1e-12,atol=1e-15):
	# 		Tens = np.dot(SecondTensor2Vector(A).reshape(3,1),SecondTensor2Vector(B).reshape(1,3))
	# 	else:
	# 		Tens = np.dot(SecondTensor2Vector(A).reshape(4,1),SecondTensor2Vector(B).reshape(1,4))

	return Tens 


def AikBjl(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

	Tens = 1.0*np.array([
		[                   A00*B00,                   A01*B01,                   A02*B02, (A00*B01)/2 + (A01*B00)/2, (A00*B02)/2 + (A02*B00)/2, (A01*B02)/2 + (A02*B01)/2],
		[                   A01*B01,                   A11*B11,                   A12*B12, (A10*B11)/2 + (A11*B10)/2, (A10*B12)/2 + (A12*B10)/2, (A11*B12)/2 + (A12*B11)/2],
		[                   A02*B02,                   A12*B12,                   A22*B22, (A20*B21)/2 + (A21*B20)/2, (A20*B22)/2 + (A22*B20)/2, (A21*B22)/2 + (A22*B21)/2],
		[ (A00*B01)/2 + (A01*B00)/2, (A10*B11)/2 + (A11*B10)/2, (A20*B21)/2 + (A21*B20)/2, (A00*B11)/2 + (A01*B10)/2, (A00*B12)/2 + (A02*B10)/2, (A01*B12)/2 + (A02*B11)/2],
		[ (A00*B02)/2 + (A02*B00)/2, (A10*B12)/2 + (A12*B10)/2, (A20*B22)/2 + (A22*B20)/2, (A00*B12)/2 + (A02*B10)/2, (A00*B22)/2 + (A02*B20)/2, (A01*B22)/2 + (A02*B21)/2],
		[ (A01*B02)/2 + (A02*B01)/2, (A11*B12)/2 + (A12*B11)/2, (A21*B22)/2 + (A22*B21)/2, (A01*B12)/2 + (A02*B11)/2, (A01*B22)/2 + (A02*B21)/2, (A11*B22)/2 + (A12*B21)/2]
		])


	return Tens


def AilBjk(A,B):

	A=1.0*A; B=1.0*B
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

	Tens = 1.0*np.array([
		[                   A00*B00,                   A01*B01,                   A02*B02, (A00*B01)/2 + (A01*B00)/2, (A00*B02)/2 + (A02*B00)/2, (A01*B02)/2 + (A02*B01)/2],
		[                   A01*B01,                   A11*B11,                   A12*B12, (A10*B11)/2 + (A11*B10)/2, (A10*B12)/2 + (A12*B10)/2, (A11*B12)/2 + (A12*B11)/2],
		[                   A02*B02,                   A12*B12,                   A22*B22, (A20*B21)/2 + (A21*B20)/2, (A20*B22)/2 + (A22*B20)/2, (A21*B22)/2 + (A22*B21)/2],
		[ (A00*B01)/2 + (A01*B00)/2, (A10*B11)/2 + (A11*B10)/2, (A20*B21)/2 + (A21*B20)/2, (A00*B11)/2 + (A01*B10)/2, (A00*B12)/2 + (A02*B10)/2, (A01*B12)/2 + (A02*B11)/2],
		[ (A00*B02)/2 + (A02*B00)/2, (A10*B12)/2 + (A12*B10)/2, (A20*B22)/2 + (A22*B20)/2, (A00*B12)/2 + (A02*B10)/2, (A00*B22)/2 + (A02*B20)/2, (A01*B22)/2 + (A02*B21)/2],
		[ (A01*B02)/2 + (A02*B01)/2, (A11*B12)/2 + (A12*B11)/2, (A21*B22)/2 + (A22*B21)/2, (A01*B12)/2 + (A02*B11)/2, (A01*B22)/2 + (A02*B21)/2, (A11*B22)/2 + (A12*B21)/2]
		])


	return Tens


# A TENSOR AND TWO VECTORS
def AijUkVl(A,U,V):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A00*U1*V1,                     A00*U2*V2, (A00*U0*V0)/2 + (A00*U1*V1)/2, (A00*U0*V0)/2 + (A00*U2*V2)/2, (A00*U1*V1)/2 + (A00*U2*V2)/2],
		[                     A00*U1*V1,                     A11*U1*V1,                     A11*U2*V2, (A11*U0*V0)/2 + (A11*U1*V1)/2, (A11*U0*V0)/2 + (A11*U2*V2)/2, (A11*U1*V1)/2 + (A11*U2*V2)/2],
		[                     A00*U2*V2,                     A11*U2*V2,                     A22*U2*V2, (A22*U0*V0)/2 + (A22*U1*V1)/2, (A22*U0*V0)/2 + (A22*U2*V2)/2, (A22*U1*V1)/2 + (A22*U2*V2)/2],
		[ (A00*U0*V0)/2 + (A00*U1*V1)/2, (A11*U0*V0)/2 + (A11*U1*V1)/2, (A22*U0*V0)/2 + (A22*U1*V1)/2, (A01*U0*V0)/2 + (A01*U1*V1)/2, (A01*U0*V0)/2 + (A01*U2*V2)/2, (A01*U1*V1)/2 + (A01*U2*V2)/2],
		[ (A00*U0*V0)/2 + (A00*U2*V2)/2, (A11*U0*V0)/2 + (A11*U2*V2)/2, (A22*U0*V0)/2 + (A22*U2*V2)/2, (A01*U0*V0)/2 + (A01*U2*V2)/2, (A02*U0*V0)/2 + (A02*U2*V2)/2, (A02*U1*V1)/2 + (A02*U2*V2)/2],
		[ (A00*U1*V1)/2 + (A00*U2*V2)/2, (A11*U1*V1)/2 + (A11*U2*V2)/2, (A22*U1*V1)/2 + (A22*U2*V2)/2, (A01*U1*V1)/2 + (A01*U2*V2)/2, (A02*U1*V1)/2 + (A02*U2*V2)/2, (A12*U1*V1)/2 + (A12*U2*V2)/2]
		])


	return Tens


def UiVjAkl(U,V,A):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A11*U0*V0,                     A22*U0*V0, (A01*U0*V0)/2 + (A10*U0*V0)/2, (A02*U0*V0)/2 + (A20*U0*V0)/2, (A12*U0*V0)/2 + (A21*U0*V0)/2],
		[                     A11*U0*V0,                     A11*U1*V1,                     A22*U1*V1, (A01*U1*V1)/2 + (A10*U1*V1)/2, (A02*U1*V1)/2 + (A20*U1*V1)/2, (A12*U1*V1)/2 + (A21*U1*V1)/2],
		[                     A22*U0*V0,                     A22*U1*V1,                     A22*U2*V2, (A01*U2*V2)/2 + (A10*U2*V2)/2, (A02*U2*V2)/2 + (A20*U2*V2)/2, (A12*U2*V2)/2 + (A21*U2*V2)/2],
		[ (A01*U0*V0)/2 + (A10*U0*V0)/2, (A01*U1*V1)/2 + (A10*U1*V1)/2, (A01*U2*V2)/2 + (A10*U2*V2)/2, (A01*U0*V1)/2 + (A10*U0*V1)/2, (A02*U0*V1)/2 + (A20*U0*V1)/2, (A12*U0*V1)/2 + (A21*U0*V1)/2],
		[ (A02*U0*V0)/2 + (A20*U0*V0)/2, (A02*U1*V1)/2 + (A20*U1*V1)/2, (A02*U2*V2)/2 + (A20*U2*V2)/2, (A02*U0*V1)/2 + (A20*U0*V1)/2, (A02*U0*V2)/2 + (A20*U0*V2)/2, (A12*U0*V2)/2 + (A21*U0*V2)/2],
		[ (A12*U0*V0)/2 + (A21*U0*V0)/2, (A12*U1*V1)/2 + (A21*U1*V1)/2, (A12*U2*V2)/2 + (A21*U2*V2)/2, (A12*U0*V1)/2 + (A21*U0*V1)/2, (A12*U0*V2)/2 + (A21*U0*V2)/2, (A12*U1*V2)/2 + (A21*U1*V2)/2]
		])


	return Tens

def AikUjVl(A,U,V):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A01*U0*V1,                     A02*U0*V2, (A00*U0*V1)/2 + (A01*U0*V0)/2, (A00*U0*V2)/2 + (A02*U0*V0)/2, (A01*U0*V2)/2 + (A02*U0*V1)/2],
		[                     A01*U0*V1,                     A11*U1*V1,                     A12*U1*V2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2],
		[                     A02*U0*V2,                     A12*U1*V2,                     A22*U2*V2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2],
		[ (A00*U0*V1)/2 + (A01*U0*V0)/2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A00*U1*V1)/2 + (A01*U1*V0)/2, (A00*U1*V2)/2 + (A02*U1*V0)/2, (A01*U1*V2)/2 + (A02*U1*V1)/2],
		[ (A00*U0*V2)/2 + (A02*U0*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A00*U1*V2)/2 + (A02*U1*V0)/2, (A00*U2*V2)/2 + (A02*U2*V0)/2, (A01*U2*V2)/2 + (A02*U2*V1)/2],
		[ (A01*U0*V2)/2 + (A02*U0*V1)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2, (A01*U1*V2)/2 + (A02*U1*V1)/2, (A01*U2*V2)/2 + (A02*U2*V1)/2, (A11*U2*V2)/2 + (A12*U2*V1)/2]
		])


	return Tens


def UiVkAjl(U,V,A):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A01*U0*V1,                     A02*U0*V2, (A00*U0*V1)/2 + (A01*U0*V0)/2, (A00*U0*V2)/2 + (A02*U0*V0)/2, (A01*U0*V2)/2 + (A02*U0*V1)/2],
		[                     A01*U0*V1,                     A11*U1*V1,                     A12*U1*V2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2],
		[                     A02*U0*V2,                     A12*U1*V2,                     A22*U2*V2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2],
		[ (A00*U0*V1)/2 + (A01*U0*V0)/2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A10*U0*V1)/2 + (A11*U0*V0)/2, (A10*U0*V2)/2 + (A12*U0*V0)/2, (A11*U0*V2)/2 + (A12*U0*V1)/2],
		[ (A00*U0*V2)/2 + (A02*U0*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A10*U0*V2)/2 + (A12*U0*V0)/2, (A20*U0*V2)/2 + (A22*U0*V0)/2, (A21*U0*V2)/2 + (A22*U0*V1)/2],
		[ (A01*U0*V2)/2 + (A02*U0*V1)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2, (A11*U0*V2)/2 + (A12*U0*V1)/2, (A21*U0*V2)/2 + (A22*U0*V1)/2, (A21*U1*V2)/2 + (A22*U1*V1)/2] 
		])


	return Tens


def AilUjVk(A,U,V):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A01*U0*V1,                     A02*U0*V2, (A00*U0*V1)/2 + (A01*U0*V0)/2, (A00*U0*V2)/2 + (A02*U0*V0)/2, (A01*U0*V2)/2 + (A02*U0*V1)/2],
		[                     A01*U0*V1,                     A11*U1*V1,                     A12*U1*V2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2],
		[                     A02*U0*V2,                     A12*U1*V2,                     A22*U2*V2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2],
		[ (A00*U0*V1)/2 + (A01*U0*V0)/2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A00*U1*V1)/2 + (A01*U1*V0)/2, (A00*U1*V2)/2 + (A02*U1*V0)/2, (A01*U1*V2)/2 + (A02*U1*V1)/2],
		[ (A00*U0*V2)/2 + (A02*U0*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A00*U1*V2)/2 + (A02*U1*V0)/2, (A00*U2*V2)/2 + (A02*U2*V0)/2, (A01*U2*V2)/2 + (A02*U2*V1)/2],
		[ (A01*U0*V2)/2 + (A02*U0*V1)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2, (A01*U1*V2)/2 + (A02*U1*V1)/2, (A01*U2*V2)/2 + (A02*U2*V1)/2, (A11*U2*V2)/2 + (A12*U2*V1)/2]
		])


	return Tens


def UiVlAjk(U,V,A):

	A=1.0*A; U=1.0*U; V=1.0*V
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]; V0=V[0]; V1=V[1]; V2=V[2];

	Tens = 1.0*np.array([
		[                     A00*U0*V0,                     A01*U0*V1,                     A02*U0*V2, (A00*U0*V1)/2 + (A01*U0*V0)/2, (A00*U0*V2)/2 + (A02*U0*V0)/2, (A01*U0*V2)/2 + (A02*U0*V1)/2],
		[                     A01*U0*V1,                     A11*U1*V1,                     A12*U1*V2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2],
		[                     A02*U0*V2,                     A12*U1*V2,                     A22*U2*V2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2],
		[ (A00*U0*V1)/2 + (A01*U0*V0)/2, (A10*U1*V1)/2 + (A11*U1*V0)/2, (A20*U2*V1)/2 + (A21*U2*V0)/2, (A10*U0*V1)/2 + (A11*U0*V0)/2, (A10*U0*V2)/2 + (A12*U0*V0)/2, (A11*U0*V2)/2 + (A12*U0*V1)/2],
		[ (A00*U0*V2)/2 + (A02*U0*V0)/2, (A10*U1*V2)/2 + (A12*U1*V0)/2, (A20*U2*V2)/2 + (A22*U2*V0)/2, (A10*U0*V2)/2 + (A12*U0*V0)/2, (A20*U0*V2)/2 + (A22*U0*V0)/2, (A21*U0*V2)/2 + (A22*U0*V1)/2],
		[ (A01*U0*V2)/2 + (A02*U0*V1)/2, (A11*U1*V2)/2 + (A12*U1*V1)/2, (A21*U2*V2)/2 + (A22*U2*V1)/2, (A11*U0*V2)/2 + (A12*U0*V1)/2, (A21*U0*V2)/2 + (A22*U0*V1)/2, (A21*U1*V2)/2 + (A22*U1*V1)/2]
		])


	return Tens



# A TENSOR AND A VECTOR
def AijUk(A,U):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[                  A00*U0,                  A10*U0,                  A20*U0],
		[                  A01*U1,                  A11*U1,                  A21*U1],
		[                  A02*U2,                  A12*U2,                  A22*U2],
		[ (A00*U1)/2 + (A01*U0)/2, (A10*U1)/2 + (A11*U0)/2, (A20*U1)/2 + (A21*U0)/2],
		[ (A00*U2)/2 + (A02*U0)/2, (A10*U2)/2 + (A12*U0)/2, (A20*U2)/2 + (A22*U0)/2],
		[ (A01*U2)/2 + (A02*U1)/2, (A11*U2)/2 + (A12*U1)/2, (A21*U2)/2 + (A22*U1)/2]
		])

	return Tens


def AikUj(A,U):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[                  A00*U0,                  A10*U0,                  A20*U0],
		[                  A01*U1,                  A11*U1,                  A21*U1],
		[                  A02*U2,                  A12*U2,                  A22*U2],
		[ (A00*U1)/2 + (A01*U0)/2, (A10*U1)/2 + (A11*U0)/2, (A20*U1)/2 + (A21*U0)/2],
		[ (A00*U2)/2 + (A02*U0)/2, (A10*U2)/2 + (A12*U0)/2, (A20*U2)/2 + (A22*U0)/2],
		[ (A01*U2)/2 + (A02*U1)/2, (A11*U2)/2 + (A12*U1)/2, (A21*U2)/2 + (A22*U1)/2]
		])


	return Tens


def UiAjk(A,U):

	A=1.0*A; U=1.0*U
	A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
	U0=U[0]; U1=U[1]; U2=U[2]

	Tens = 1.0*np.array([
		[                  A00*U0,                  A00*U1,                  A00*U2],
		[                  A11*U0,                  A11*U1,                  A11*U2],
		[                  A22*U0,                  A22*U1,                  A22*U2],
		[ (A01*U0)/2 + (A10*U0)/2, (A01*U1)/2 + (A10*U1)/2, (A01*U2)/2 + (A10*U2)/2],
		[ (A02*U0)/2 + (A20*U0)/2, (A02*U1)/2 + (A20*U1)/2, (A02*U2)/2 + (A20*U2)/2],
		[ (A12*U0)/2 + (A21*U0)/2, (A12*U1)/2 + (A21*U1)/2, (A12*U2)/2 + (A21*U2)/2]
		])


	return Tens



# FOUR VECTORS MAKING A 4TH ORDER TENSOR
def UiVjXkYl(U,V,X,Y):
	# There is only one combination of this and the way this works is you just arrange 
	# your input vectors accrodingly such that it is always Ui-Vj-Xk-Yl


	U=1.0*U; V=1.0*V; X=1.0*X; Y=1.0*Y
	U0=U[0]; U1=U[1]; U2=U[2]
	V0=V[0]; V1=V[1]; V2=V[2]
	X0=X[0]; X1=X[1]; X2=X[2]
	Y0=Y[0]; Y1=Y[1]; Y2=Y[2]

	Tens = 1.0*np.array([
		[                       U0*V0*X0*Y0,                       U0*V0*X1*Y1,                       U0*V0*X2*Y2, (U0*V0*X0*Y1)/2 + (U0*V0*X1*Y0)/2, (U0*V0*X0*Y2)/2 + (U0*V0*X2*Y0)/2, (U0*V0*X1*Y2)/2 + (U0*V0*X2*Y1)/2],
		[                       U0*V0*X1*Y1,                       U1*V1*X1*Y1,                       U1*V1*X2*Y2, (U1*V1*X0*Y1)/2 + (U1*V1*X1*Y0)/2, (U1*V1*X0*Y2)/2 + (U1*V1*X2*Y0)/2, (U1*V1*X1*Y2)/2 + (U1*V1*X2*Y1)/2],
		[                       U0*V0*X2*Y2,                       U1*V1*X2*Y2,                       U2*V2*X2*Y2, (U2*V2*X0*Y1)/2 + (U2*V2*X1*Y0)/2, (U2*V2*X0*Y2)/2 + (U2*V2*X2*Y0)/2, (U2*V2*X1*Y2)/2 + (U2*V2*X2*Y1)/2],
		[ (U0*V0*X0*Y1)/2 + (U0*V0*X1*Y0)/2, (U1*V1*X0*Y1)/2 + (U1*V1*X1*Y0)/2, (U2*V2*X0*Y1)/2 + (U2*V2*X1*Y0)/2, (U0*V1*X0*Y1)/2 + (U0*V1*X1*Y0)/2, (U0*V1*X0*Y2)/2 + (U0*V1*X2*Y0)/2, (U0*V1*X1*Y2)/2 + (U0*V1*X2*Y1)/2],
		[ (U0*V0*X0*Y2)/2 + (U0*V0*X2*Y0)/2, (U1*V1*X0*Y2)/2 + (U1*V1*X2*Y0)/2, (U2*V2*X0*Y2)/2 + (U2*V2*X2*Y0)/2, (U0*V1*X0*Y2)/2 + (U0*V1*X2*Y0)/2, (U0*V2*X0*Y2)/2 + (U0*V2*X2*Y0)/2, (U0*V2*X1*Y2)/2 + (U0*V2*X2*Y1)/2],
		[ (U0*V0*X1*Y2)/2 + (U0*V0*X2*Y1)/2, (U1*V1*X1*Y2)/2 + (U1*V1*X2*Y1)/2, (U2*V2*X1*Y2)/2 + (U2*V2*X2*Y1)/2, (U0*V1*X1*Y2)/2 + (U0*V1*X2*Y1)/2, (U0*V2*X1*Y2)/2 + (U0*V2*X2*Y1)/2, (U1*V2*X1*Y2)/2 + (U1*V2*X2*Y1)/2]
		])


	return Tens


# THREE VECTORS MAKING A 3RD ORDER TENSOR
def UiVjXk(U,V,X):
	# There is only one combination of this and the way this works is you just arrange 
	# your input vectors accrodingly such that it is always Ui-Vj-Xk

	U=1.0*U; V=1.0*V; X=1.0*X
	U0=U[0]; U1=U[1]; U2=U[2]
	V0=V[0]; V1=V[1]; V2=V[2]
	X0=X[0]; X1=X[1]; X2=X[2]

	Tens = 1.0*np.array([
		[                    U0*V0*X0,                    U1*V0*X0,                    U2*V0*X0],
		[                    U0*V1*X1,                    U1*V1*X1,                    U2*V1*X1],
		[                    U0*V2*X2,                    U1*V2*X2,                    U2*V2*X2],
		[ (U0*V0*X1)/2 + (U0*V1*X0)/2, (U1*V0*X1)/2 + (U1*V1*X0)/2, (U2*V0*X1)/2 + (U2*V1*X0)/2],
		[ (U0*V0*X2)/2 + (U0*V2*X0)/2, (U1*V0*X2)/2 + (U1*V2*X0)/2, (U2*V0*X2)/2 + (U2*V2*X0)/2],
		[ (U0*V1*X2)/2 + (U0*V2*X1)/2, (U1*V1*X2)/2 + (U1*V2*X1)/2, (U2*V1*X2)/2 + (U2*V2*X1)/2]
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


# CHECK
###############################################
# A = np.array([
	# [6,1,2],
	# [1,7,3],
	# [2,3,8]
	# ])
# B = np.array([
	# [6,4,1],
	# [4,12,3],
	# [1,3,9]
	# ])

# I = np.eye(3,3)
# U=np.array([1.,2,3]) 
# V=np.array([4.,5,6])
# print U[2]

# print UiVjXk(V,U,U)

############################################### 
# dynamic types check
# print B
# C=1.0*np.array([
# 	[B[0,0],B[0,1],B[0,2]],
# 	[B[2,2]/2+B[1,1]/2,B[1,1],B[1,2]],
# 	[B[0,0]/2+B[0,0]/2,B[0,0],B[2,2]]
# 	])
# print C
# print 1.0*C
############################################

# FourthTensor2SecondTensor(A,B,(i,j,k,l))		