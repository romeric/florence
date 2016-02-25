import numpy as np
cimport numpy as np 
from scipy.stats import itemfreq
import cython 
from Core.Supplementary.Where import *


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

DTYPEI = np.int64
ctypedef np.int64_t DTYPEI_t

@cython.boundscheck(False)
@cython.wraparound(False)
def NodeLoopTriNPSP_Cython(np.ndarray[DTYPE_t, ndim=2] sorted_repoints,np.ndarray[DTYPE_t, ndim=2] Xs,np.ndarray[DTYPEI_t, ndim=1] invX,
	np.ndarray[DTYPEI_t, ndim=1] iSortX,np.ndarray[DTYPEI_t, ndim=2] duplicates, DTYPEI_t Decimals,DTYPE_t tol):

	cdef DTYPEI_t counter = 0
	cdef DTYPEI_t i,j
	cdef DTYPEI_t Xs_shape = Xs.shape[0]
	cdef DTYPEI_t Ys_shape, Ysy_shape, YsYs_shape 
	cdef np.ndarray Ys = np.zeros([],dtype=DTYPE)
	cdef np.ndarray Ysy = np.zeros([],dtype=DTYPE)
	cdef np.ndarray YsYs = np.zeros([],dtype=DTYPE)
	cdef np.ndarray dups = np.zeros([],dtype=DTYPE)
	# LOOP OVER POINTS
	for i in range(0,Xs_shape):
		# IF THE MULITPLICITY OF A GIVEN X-VALUE IS 1 THEN INGONRE
		if Xs[i,1]!=1:
			# IF THE MULTIPLICITY IS MORE THAN 1, THEN FIND WHERE ALL IN THE SORTED ARRAY THIS X-VALUE THEY OCCURS
			# dups =  np.where(i==invX)[0]
			dups = np.asarray(whereEQ(invX.reshape(invX.shape[0],1),i)[0])

			# FIND THE Y-COORDINATE VALUES OF THESE MULTIPLICITIES 
			Ys = sorted_repoints[dups,:][:,1]
			Ys_shape = Ys.shape[0]
			# IF MULTIPLICITY IS 2 THEN FIND IF THEY ARE Y-VALUES ARE EQUAL  
			if Ys_shape == 2:
				if np.abs(Ys[1]-Ys[0]) < tol:
					# IF EQUAL MARK THIS POINT AS DUPLICATE
					duplicates[counter,:] = dups
					# INCREASE THE COUNTER
					counter += 1
			# MULTIPLICITY CAN BE GREATER THAN 2, IN WHICH CASE FIND MULTIPLICITY OF Ys
			else:
				Ysy=itemfreq(np.round(Ys,decimals=Decimals))
				Ysy_shape = Ysy.shape[0]
				# IF itemfreq GIVES THE SAME LENGTH ARRAY, MEANS ALL VALUES ARE UNIQUE/DISTINCT AND WE DON'T HAVE TO CHECK
				if Ysy_shape!=Ys_shape:
					# OTHERWISE LOOP OVER THE ARRAY AND
					for j in range(0,Ysy.shape[0]):
						# FIND WHERE THE VALUES OCCUR
						YsYs = np.where(Ysy[j,0]==np.round(Ys,decimals=Decimals))[0]
						YsYs_shape = dups[YsYs].shape[0]
						# THIS LEADS TO A SITUATION WHERE SAY 3 NODES HAVE THE SAME X-VALUE, BUT TWO OF THEIR Y-VALUES ARE THE
						# SAME AND ONE IS UNIQUE. CHECK IF THIS IS JUST A NODE WITH NO Y-MULTIPLICITY
						if YsYs_shape!=1:
							# IF NOT THEN MARK AS DUPLICATE
							duplicates[counter,:] = dups[YsYs]
							# INCREASE COUNTER
							counter += 1

	# RE-ASSIGN DUPLICATE
	duplicates = duplicates[:counter,:]
	# BASED ON THE DUPLICATES OCCURING IN THE SORTED ARRAY sorted_repoints, FIND THE ACTUAL DUPLICATES OCCURING IN repoints
	duplicates = np.asarray([iSortX[duplicates[:,0]],iSortX[duplicates[:,1]] ]).T
	# SORT THE ACTUAL DUPLICATE ROW-WISE SO THAT THE FIRST COLUMN IS ALWAYS SMALLER THAN THE SECOND COLUMN
	duplicates = np.sort(duplicates,axis=1)

	return duplicates