import numpy as np 

def SparseAssemblySmall(i,j,coeff,I,J,V,elem,nvar,nodeperelem,elements):

	current_row_column = np.zeros((nvar*nodeperelem),dtype=np.int64)
	ncounter = np.arange(nvar)
	for counter in range(0,nodeperelem):
		current_row_column[nvar*counter+ncounter] = nvar*elements[elem,counter]+ncounter


	full_current_row = np.copy(i); full_current_column = np.copy(j)
	for counter in range(0,nvar*nodeperelem):
		full_current_row[i==counter]=current_row_column[counter]
		full_current_column[j==counter]=current_row_column[counter]


	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff



#----------------------------------------------------------------------------------------------------------
# OLD ROUTINE

# def FindIndices(A):
# 	# NEW FASTER APPROACH - NO TEMPORARY
# 	return np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0), np.tile(np.arange(0,A.shape[0]),A.shape[0]), A.ravel()



# def SparseAssembly(stiffness,I,J,V,elem,nvar,nodeperelem,elements,sort=0):

# 	# Sparse Assembly
# 	# i, j = sp.nonzero(stiffness); coeff = k[sp.nonzero(k)] # DONT USE THIS
# 	i, j, coeff = FindIndices(stiffness) 	# User-defined		

# 	current_row_column = np.zeros(nvar*nodeperelem)
# 	for counter in range(0,nodeperelem):
# 		for ncounter in range(0,nvar):
# 			current_row_column[nvar*counter+ncounter] = nvar*elements[elem,counter]+ncounter

# 	full_current_row = np.copy(i); full_current_column = np.copy(j);
# 	for counter in range(0,nvar*nodeperelem):
# 		full_current_row[np.where(i==counter)[0]]=current_row_column[counter]
# 		full_current_column[np.where(j==counter)[0]]=current_row_column[counter]

# 	# STORE INDICES AND COEFFICIENTS IN I, J AND V VECTORS
# 	I[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_row
# 	J[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = full_current_column
# 	V[(nvar*nodeperelem)**2*elem:(nvar*nodeperelem)**2*(elem+1)] = coeff

# 	return I, J, V
