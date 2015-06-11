import numpy as np 
from Core.Supplementary.Tensors import itemfreq_py

def duplicate(ar,decimals=10,tol=1e-14):
	""" FIND DUPLICATED ROWS IN A 2D FLOATING POINT ARRAY WITH SHAPE NX2.
		NOTE THAT IF THE ARRAY IS INTEGER A FLOATING POINT ARRAY IS RETURNED.
		input:	
			ar:				2D ARRAY
			Decimals:	 	UP TO WHICH DECIMAL THE ELEMENTS OF THE ARRAY NEEDS TO CHECKED FOR DUPLICATES
			tol:			THE TOLERANCE PRESCIRBED FOR TWO FLOATING POINTS TO BE CONSIDERED EQUAL

		returns:
			duplicates:		A TUPLE OF NUMPY ARRAYS WITH EACH ARRAY CONTAINING THE DUPLICATED ROWS"""


	# duplicates = []
	duplicates = ()

	iSortY = np.argsort(ar[:,0])
	# BASED ON THE SORTED Y-VALUES, PUT THEIR CORRESPONDING Z-VALUES NEXT TO THEM I.E. SORT Ys BASED ON Y-VALUES
	sorted_Ys = ar[iSortY,:]
	# NOW LETS FIND THE UNIQUE VALUES OF THIS SORTED FLOATING POINTS ARRAY
	# NOTE THAT FROM THE INVERSE INDICES OF A UNIQUE ARRAY WE CAN CONSTRUCT THE ACTUAL ARRAY 
	unique_Ys,invY =np.unique(np.round(sorted_Ys[:,0],decimals=decimals),return_inverse=True)
	# NOW FIND THE MULTIPLICITY OF EACH UNIQUE Y-VALUE 
	Yss = itemfreq_py(un_arr=unique_Ys,inv_arr=invY)

	counter = 0
	for k in range(0,Yss.shape[0]):
		if Yss[k,1]!=1:
			# dupsy = np.asarray(whereEQ(invY.reshape(invY.shape[0],1),k)[0]) 
			dupsy = np.arange(np.sum(np.int64(Yss[:k,1])),np.sum(np.int64(Yss[:k+1,1])))
			Zs = sorted_Ys[dupsy,:][:,1]
			# IF MULTIPLICITY IS 2 THEN FIND IF THEIR Z-VALUES ARE EQUAL  
			if Zs.shape[0]==2:
				if np.abs(Zs[1]-Zs[0]) < tol:
					# IF EQUAL MARK THIS POINT AS DUPLICATE
					# duplicates.append(np.sort(iSortY[dupsy]))
					duplicates = duplicates + (np.sort(iSortY[dupsy]),)
					# INCREASE THE COUNTER
					counter += 1
			# MULTIPLICITY CAN BE GREATER THAN 2, IN WHICH CASE FIND MULTIPLICITY OF Ys
			else:
				Zsy = itemfreq_py(Zs,decimals=decimals)
				# IF itemfreq GIVES THE SAME LENGTH ARRAY, MEANS ALL VALUES ARE UNIQUE/DISTINCT AND WE DON'T HAVE TO CHECK
				if Zsy.shape[0]!=Zs.shape[0]:
					# OTHERWISE LOOP OVER THE ARRAY AND
					for j in range(0,Zsy.shape[0]):
						# FIND WHERE THE VALUES OCCUR
						ZsZs = np.where(Zsy[j,0]==np.round(Zs,decimals=decimals))[0]
						# THIS LEADS TO A SITUATION WHERE SAY 3 NODES HAVE THE SAME X-VALUE, BUT TWO OF THEIR Y-VALUES ARE THE
						# SAME AND ONE IS UNIQUE. CHECK IF THIS IS JUST A NODE WITH NO Y-MULTIPLICITY
						if dupsy[ZsZs].shape[0]!=1:
							# IF NOT THEN MARK AS DUPLICATE
							# duplicates.append( np.sort(iSortY[dupsy[ZsZs]]) )
							duplicates = duplicates + ( np.sort(iSortY[dupsy[ZsZs]]),)
							# INCREASE COUNTER
							counter += 1


	return duplicates
