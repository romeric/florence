
import numpy as np 

def SemiEmptyArrays(length,bc_indices,bc):

	# length - length of 1D array
	# bc_indices - indices where boundary conditions should be placed
	# bc - actual bc
	################################################################

	# Pure Python (lists)
	b = []
	for i in range(0,length):
		b.append([])

	bc_indices = np.array(bc_indices)
	print b

	for i in range(0,length):
		x = np.where(bc_indices==i)[0]
		if x.shape[0]!=0:
			b[i] = bc[x[0]]
	
	return b


b = SemiEmptyArrays(5,(0,3),(7.5,9))
print b
