import numpy as np 
# from Core.Supplementary.Where import *

# NOTE THAT THIS ASSEMBLY ROUTINE IS SPLIT INTO TWO FUNCTIONS, ONE WHICH IS PARALLELISABLE AND ONE 
# WHICH IS NOT MEMORY-EFFICIENT FOR PYTHONS MULTI-PROCESSING POOL. SECOND FUNCTION CAN BE EMBEDDED 
# WITHIN THE ASSEMBLY ROUTINE TO AVOID PASSING OFF HUGE ARRAYS (I,J,V) TO ANOTHER FUNCTION WHICH
# HAPPENS WITHIN AN ELEMENT HOWEVER THEY ARE KEPT LIKE TO BE GENERIC IN USAGE FOR STIFFNESS OR MASS
# ASSEMBLY

# NOTE THAT THIS METHOD IS ONLY FASTER FOR LARGER MATRICES E.G. NELEM > 100000


# PARALLELISABLE ROUTINE
def SparseAssembly_Step_1(i,j,nvar,nodeperelem,elem,elements):

	current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
	niter = np.arange(nvar)
	for iter in range(0,nodeperelem):
		current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter


	full_current_row = np.copy(i); full_current_column = np.copy(j)
	for iter in range(0,nvar*nodeperelem):
		full_current_row[i==iter]=current_row_column[iter]
		full_current_column[j==iter]=current_row_column[iter]

	return full_current_row, full_current_column


# NOT PARALLELASIBLE ROUTINE
def SparseAssembly_Step_2(I,J,V,full_current_row,full_current_column,coeff,nvar,nodeperelem,elem):
	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff

	return I, J, V












# OLDER VERSION 
#------------------------------------------------------------------------------------------------#
# def SparseAssembly(i,j,coeff,I,J,V,elem,nvar,nodeperelem,elements,sort=0):

# 	# current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
# 	# for iter in range(0,nodeperelem):
# 	# 	for niter in range(0,nvar):
# 	# 		current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

# 	# full_current_row = np.copy(i); full_current_column = np.copy(j);
# 	# for iter in range(0,nvar*nodeperelem):
# 		# full_current_row[np.where(i==iter)[0]]=current_row_column[iter]
# 		# full_current_column[np.where(j==iter)[0]]=current_row_column[iter]


# 	current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
# 	niter = np.arange(nvar)
# 	for iter in range(0,nodeperelem):
# 		current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

# 	# from time import time 
# 	full_current_row = np.copy(i); full_current_column = np.copy(j)
# 	for iter in range(0,nvar*nodeperelem):
# 		full_current_row[i==iter]=current_row_column[iter]
# 		full_current_column[j==iter]=current_row_column[iter]

# 	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
# 	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
# 	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
# 	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff

# 	return I, J, V
#------------------------------------------------------------------------------------------------#











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
