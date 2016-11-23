import numpy as np 

def GetInteriorNodesCoordinates(xycoord,MeshType,elem,eps,Neval):


    if MeshType =='tri':
        xycoord_higher = np.zeros((eps.shape[0],2))
        # Nshape0 = Neval.shape[0]
        # for i in range(3,eps.shape[0]):
            # xycoord_higher[i,:] = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)
            
        xycoord_higher[:3,:]=xycoord[:,:2]
        xycoord_higher[3:,:]=np.dot(Neval[:,3:].T,xycoord)


    if MeshType == 'tet':
        xycoord_higher = np.zeros((eps.shape[0],3))
        # Nshape0 = Neval.shape[0]
        # for i in range(4,eps.shape[0]):
            # xycoord_higher[i,:] = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)

        xycoord_higher[:4,:]=xycoord
        xycoord_higher[4:,:]=np.dot(Neval[:,4:].T,xycoord)


    elif MeshType=='quad':
        # xycoord_higher = np.zeros((1,2))
        # Nshape0 = Neval.shape[0]
        # for i in range(0,eps.shape[0]):
        #     xy = np.dot(Neval[:,i].reshape(1,Nshape0),xycoord)
        #     xycoord_higher = np.append(xycoord_higher,xy,axis=0)

        # xycoord_higher = xycoord_higher[1:,:]

        xycoord_higher = np.zeros((eps.shape[0],2))
        xycoord_higher[:4,:]=xycoord
        xycoord_higher[4:,:]=np.dot(Neval[:,4:].T,xycoord)
        # print xycoord.shape, Neval[:,4:].shape
        # print np.dot(Neval[:,4:].T,xycoord)
        # print np.dot(xycoord.T,Neval[:,4:]).T
        # print xycoord
        # exit()


    return xycoord_higher