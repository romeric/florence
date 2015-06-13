import numpy as np 
# from Core.Supplementary.Where import *

def SparseAssemblySmall(i,j,coeff,I,J,V,elem,nvar,nodeperelem,elements,sort=0):

	# current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
	# for iter in range(0,nodeperelem):
	# 	for niter in range(0,nvar):
	# 		current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

	# full_current_row = np.copy(i); full_current_column = np.copy(j);
	# for iter in range(0,nvar*nodeperelem):
		# full_current_row[np.where(i==iter)[0]]=current_row_column[iter]
		# full_current_column[np.where(j==iter)[0]]=current_row_column[iter]


	current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
	niter = np.arange(nvar)
	for iter in range(0,nodeperelem):
		current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

	# from time import time 
	full_current_row = np.copy(i); full_current_column = np.copy(j)
	for iter in range(0,nvar*nodeperelem):
		# t1=time()
		# full_current_row[np.asarray(whereEQ1d(i,iter))]=current_row_column[iter]
		# full_current_column[np.asarray(whereEQ1d(j,iter))]=current_row_column[iter]

		# full_current_row[np.where(i==iter)[0]]=current_row_column[iter]
		# full_current_column[np.where(j==iter)[0]]=current_row_column[iter]
		
		full_current_row[i==iter]=current_row_column[iter]
		full_current_column[j==iter]=current_row_column[iter]
		# print time()-t1

	# t2=time()
	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff
	# print time()-t2

	# temp = np.arange( (nvar*nodeperelem)**2*elem,(nvar*nodeperelem)**2*(elem+1))
	# I.take(temp,mode='clip')

	return I, J, V












#-------------------------------------------------------------------------------------------------------------
# OLD ROUTINE

# def FindIndices(A):
# 	# NEW FASTER APPROACH - NO TEMPORARY
# 	return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()



# def SparseAssembly(stiffness,I,J,V,elem,nvar,nodeperelem,elements,sort=0):

# 	# Sparse Assembly
# 	# i, j = sp.nonzero(stiffness); coeff = k[sp.nonzero(k)] # DONT USE THIS
# 	i, j, coeff = FindIndices(stiffness) 	# User-defined		

# 	current_row_column = np.zeros(nvar*nodeperelem)
# 	for iter in range(0,nodeperelem):
# 		for niter in range(0,nvar):
# 			current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

# 	full_current_row = np.copy(i); full_current_column = np.copy(j);
# 	for iter in range(0,nvar*nodeperelem):
# 		full_current_row[np.where(i==iter)[0]]=current_row_column[iter]
# 		full_current_column[np.where(j==iter)[0]]=current_row_column[iter]

# 	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
# 	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
# 	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
# 	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff

# 	return I, J, V
