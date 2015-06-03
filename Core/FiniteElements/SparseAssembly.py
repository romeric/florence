import numpy as np 


def SparseAssembly(i,j,coeff,I,J,V,elem,nvar,nodeperelem,elements,sort=0):

	current_row_column = np.zeros(nvar*nodeperelem)
	for iter in range(0,nodeperelem):
		for niter in range(0,nvar):
			current_row_column[nvar*iter+niter] = nvar*elements[elem,iter]+niter

	full_current_row = np.copy(i); full_current_column = np.copy(j);
	for iter in range(0,nvar*nodeperelem):
		full_current_row[np.where(i==iter)[0]]=current_row_column[iter]
		full_current_column[np.where(j==iter)[0]]=current_row_column[iter]

	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff

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
