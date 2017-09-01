import numpy as np

def GetInteriorNodesCoordinates(xycoord, element_type, elem, eps, Neval):

    xycoord_higher              = np.zeros((eps.shape[0],xycoord.shape[1]))

    if element_type == 'hex':
        xycoord_higher[:8,:]    = xycoord
        xycoord_higher[8:,:]    = np.dot(Neval[:,8:].T,xycoord)

    elif element_type == 'tet':
        xycoord_higher[:4,:]    = xycoord
        xycoord_higher[4:,:]    = np.dot(Neval[:,4:].T,xycoord)

    elif element_type == 'quad':
        xycoord_higher[:4,:]    = xycoord
        xycoord_higher[4:,:]    = np.dot(Neval[:,4:].T,xycoord)

    elif element_type == 'tri':
        xycoord_higher[:3,:]    = xycoord
        xycoord_higher[3:,:]    = np.dot(Neval[:,3:].T,xycoord)

    elif element_type == 'line':
        xycoord_higher[:2,:]    = xycoord
        xycoord_higher[2:,:]    = np.dot(Neval[:,2:].T,xycoord)


    return xycoord_higher