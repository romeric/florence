import numpy as np

def summ(A):
	sums=0.
	for i in xrange(A.shape[0]):
		for j in xrange(A.shape[1]):
			sums += A[i,j]
	return sums