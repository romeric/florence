# distutils: language = c++
# distutils: include_dirs = /home/roman/Desktop/EigenTest
# distutils: sources = cy_cpp.cpp


#from cy_cpp cimport addtwo

cdef extern from "cy_cpp.h":
	double addtwo(double x, double y)

def pytest():
	cdef double x=2.5
	cdef double y=4.6
	cdef double z
	z = addtwo(x,y)

	print z