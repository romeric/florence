import numpy as np 

def GetInteriorNodesCoordinates(xycoord,MeshType,elem,eps,Neval):

	if MeshType=='quad':
		xycoord_higher = np.zeros((1,2))
		Nshape0 = Neval.shape[0]
		for i in range(0,eps.shape[0]):
			xy = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)
			xycoord_higher = np.append(xycoord_higher,xy,axis=0)

		xycoord_higher = xycoord_higher[1:,:]


	if MeshType =='tri':
		Nshape0 = Neval.shape[0]
		xycoord_higher = np.zeros((eps.shape[0],2))
		for i in range(3,eps.shape[0]):
			xycoord_higher[i,:] = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)
			
		xycoord_higher[:3,:]=xycoord[:,:2]


	if MeshType == 'tet':
		xycoord_higher = np.zeros((eps.shape[0],3))
		Nshape0 = Neval.shape[0]
		for i in range(4,eps.shape[0]):
			xycoord_higher[i,:] = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)

			# if elem==0:
				# print i 
				# print Neval[:,i]
				# print Neval.shape

		xycoord_higher[:4,:]=xycoord

		# if elem==0:
			# print Neval[:,i]
			# print eps.shape

			# print xycoord_higher
			# print xycoord


	return xycoord_higher