import numpy as np 
cimport numpy as np
import cython 
import imp, os 
# pwd = os.path.abspath(os.path.join(os.path.dirname( __file__ )))
pwd = os.getcwd()
mod_nurbs = imp.load_source('ProblemData',pwd+'/nurbs.py')
# from nurbs import CurveLengthAdaptive

DTYPE = np.float64
DTYPEI = np.int64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPEI_t

@cython.wraparound(False)
@cython.boundscheck(False)
def CurveEquallySpacedPoints_c(dict aNurbs, DTYPE_t u1, DTYPE_t u2, nPoints, DTYPE_t lengthTOL=1e-06):

	cdef DTYPEI_t nOfMaxIterations = 1000
	cdef np.ndarray uEq = np.zeros(nPoints,dtype=np.float64)
	
	uEq[0] = u1
	uEq[nPoints-1] = u2
	
	cdef DTYPE_t lengthU = np.abs(u2-u1)
	cdef DTYPE_t length = mod_nurbs.CurveLengthAdaptive(aNurbs, u1, u2, lengthTOL)
	cdef DTYPE_t lengthSub = 1.0*length/(nPoints-1.0)


	cdef np.ndarray[DTYPE_t,ndim=1] uGuess = np.linspace(u1,u2,nPoints)
	cdef DTYPEI_t nOfIntPoints = nPoints-2

	cdef DTYPEI_t iIntPoints, niter
	cdef DTYPE_t a,b,uOld,f,errU,errF

	for iIntPoints in range(nOfIntPoints):
		currentLength = (iIntPoints+1)*lengthSub
		a = u1
		b = u2
		# INITIAL GUESS 
		uOld = 0.5*(uGuess[iIntPoints] + uGuess[iIntPoints+2])
		length = mod_nurbs.CurveLengthAdaptive(aNurbs, u1 , uOld, lengthTOL)
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

			length = mod_nurbs.CurveLengthAdaptive(aNurbs, u1 , uEq[iIntPoints+1], lengthTOL)
			errU = np.abs(uOld-uEq[iIntPoints+1])/lengthU
			errF = np.abs(currentLength - length)/currentLength
			if errU<lengthTOL and errF < lengthTOL:
				break
			uOld = uEq[iIntPoints+1]
			# print niter

	return uEq