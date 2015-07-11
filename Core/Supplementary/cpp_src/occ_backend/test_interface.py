import numpy as np
import main_interface 

C=1
p=C+1
nsize = (p+1)*(p+2)/2
nelem=25
elements = np.arange(nelem*(p+2)).reshape(25,p+2)
points = np.random.rand(np.max(elements)+1,2)
faces = np.zeros((1,4),dtype=np.int64)
edges = np.arange(elements.shape[0]*3*(p+1)).reshape(75,p+1)

# from Core import Mesh 

import main_interface
main_interface.main_interface(points,elements,edges,faces)