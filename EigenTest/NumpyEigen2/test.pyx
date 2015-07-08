include "matrices.h"
import numpy


def getNDArray(self, int d1, int d2): 
    cdef MatrixXdPy me = self.thisptr.returnMatrixXd(d1,d2) # get MatrixXdPy object
     
    result = numpy.zeros((me.rows(),me.cols())) # create nd array 
    # Fill out the nd array with MatrixXf elements 
    for row in range(me.rows()): 
        for col in range(me.cols()): 
            result[row, col] = me.coeff(row, col)   
    return result 