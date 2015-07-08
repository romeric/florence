import numpy as np
import main_interface 


elements = np.arange(100).reshape(25,4)
points = np.random.rand(np.max(elements)+1,2)

# print elements
# print points

import main_interface

# print dir(main_interface)
main_interface.main_interface(elements.astype(np.int32),points.astype(np.float64))