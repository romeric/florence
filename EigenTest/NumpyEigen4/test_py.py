#!/usr/bin/env python

"""
simple test of the multiply.pyx and c_multiply.c test code
"""


import numpy as np
import numpy_eigen

# a = np.arange(12, dtype=np.float64).reshape((3,4))
n=1400000
a=np.random.rand(n,20)
# print a
# print dir(numpy_eigen)
from time import time

t1 = time() 
numpy_eigen.pass_to_eigen(a)
print time()-t1
# print a
# print b