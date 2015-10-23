import numpy as np

def NodeArrangement(C):

	# Traversing the tetrahedral only via edges - used for plotting
	a1,a2,a3,a4 = [],[],[],[]
	if C==1:
		a1 = [1, 5, 2, 9, 4, 8, 1]
		a2 = [1, 5, 2, 7, 3, 6, 1]
		a3 = [1, 6, 3, 10, 4, 8, 1]
		a4 = [2, 7, 3, 10, 4, 9, 2]
	elif C==2:
		a1 = [1, 5, 6, 2, 9, 11, 3, 10, 7, 1]
		a2 = [1, 5, 6, 2, 14, 19, 4, 18, 12, 1]
		a3 = [2, 9, 11, 3, 17, 20, 4, 19, 14, 2]
		a4 = [1, 12, 18, 4, 20, 17, 3, 10, 7, 1]
	elif C==3:
		a1 = [1, 5, 6, 7, 2, 20, 29, 34, 4, 33, 27, 17, 1]
		a2 = [1, 8, 12, 15, 3, 16, 14, 11, 2, 7, 6, 5, 1]
		a3 = [2, 11, 14, 16, 3, 26, 32, 35, 4, 34, 29, 20, 2]
		a4 = [1, 8, 12, 15, 3, 26, 32, 35, 4, 33, 27, 17, 1]
	elif C==4:
		a1 = [1, 5, 6, 7, 8, 2, 27, 41, 50, 55, 4, 54, 48, 38, 23, 1]
		a2 = [1, 9, 14, 18, 21, 3, 22, 20, 17, 13, 2, 8, 7, 6, 5, 1]
		a3 = [2, 13, 17, 20, 22, 3, 37, 47, 53, 56, 4, 55, 50, 41, 27, 2]
		a4 = [1, 9, 14, 18, 21, 3, 37, 47, 53, 56, 4, 54, 48, 38, 23, 1]

	a1 = np.asarray(a1)
	a2 = np.asarray(a2)
	a3 = np.asarray(a3)
	a4 = np.asarray(a4)

	a1 -= 1
	a2 -= 1
	a3 -= 1
	a4 -= 1

	traversed_edge_numbering = [a1,a2,a3,a4]


	# Getting face numbering order from a tetrahedral element
	face_1,face_2,face_3,face_4 = [],[],[],[]
	if C==1:
		face_1 = [0,1,2,4,5,6]
		face_2 = [0,1,3,4,7,8]
		face_3 = [0,2,3,5,7,9]
		face_4 = [1,2,3,6,8,9]
	elif C==2:
		face_1 = [0,1,2,4,5,6,7,8,9,10]
		face_2 = [0,1,3,4,5,11,12,13,17,18]
		face_3 = [0,2,3,6,9,11,14,16,17,19]
		face_4 = [1,2,3,8,10,13,15,16,18,19]


	face_numbering = np.array([face_1,face_2,face_3,face_4])


	return face_numbering, traversed_edge_numbering