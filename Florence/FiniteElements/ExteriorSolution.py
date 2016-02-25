import numpy as np 

def ExteriorSolution(mesh,total_sol,nvar):

	unique_nodes = np.unique(mesh.elements)

	exteriors = []
	for i in range(0,unique_nodes.shape[0]):
		for j in range(0,nvar):
			exteriors = np.append(exteriors,nvar*unique_nodes[i]+j)

	exteriors = np.array(exteriors, dtype=int)

	return total_sol[exteriors]
