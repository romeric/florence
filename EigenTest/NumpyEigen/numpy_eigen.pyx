from libcpp.vector cimport vector
cimport numpy as np
import numpy as np

# cdef extern from "Eigen/Dense" namespace "Eigen":
	# cdef cppclass MatrixXd
cdef extern from "cpp_backend.h":
	vector[vector[double]] manipulate(vector[vector[int]] elements_std,vector[vector[double]] points_std)

def copy_to_std_vector(np.ndarray[np.int64_t,ndim=2] elements,np.ndarray[np.float64_t,ndim=2] points):
	
	cdef int i,j 

	cdef vector[vector[int]] elements_std
	cdef vector[vector[double]] points_std

	cdef vector[int] dummy_1
	cdef vector[double] dummy_2
	dummy_1.clear(); dummy_2.clear()

	for i in range(elements.shape[0]):
		dummy_1.clear()
		for j in range(elements.shape[1]):
			dummy_1.push_back(elements[i,j])
		elements_std.push_back(dummy_1)

	for i in range(points.shape[0]):
		dummy_2.clear()
		for j in range(points.shape[1]):
			dummy_2.push_back(points[i,j])
		points_std.push_back(dummy_2)  


	cdef vector[vector[double]] dirichletbc_std = manipulate(elements_std,points_std) 

	cdef np.ndarray dirichletbc = np.zeros((dirichletbc_std.size(), dirichletbc_std[0].size()),dtype=np.float64) 
	for i in range(dirichletbc_std.size()):
		for j in range(dirichletbc_std[0].size()):
			dirichletbc[i,j] = dirichletbc_std[i][j]


	return dirichletbc


#-----------------------------------
	# # cdef vector[double] x(10); 
	# # cdef vector[int] vect
	# cdef vector[double] vect
	# # cdef vector[int]* x = new vector[int](10)
	# cdef int i;
	# for i in range(10):
	# 	vect.push_back(i);

	# cdef double dd; 
	# cdef np.ndarray y=np.ones((10),dtype=np.float64);

	# for i in range(0,10):
	# 	y[i] = vect[i]

	# # print y