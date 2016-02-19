import numpy as np 
from Core.Supplementary.CythonBuilds.voigt_sym.voigt_sym import voigt_sym



def Cross(A,B):

    A=1.0*A; B=1.0*B
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

    AB = np.zeros((3,3),dtype=np.float64)
    if A.shape==(3,3) and B.shape==(3,3):
        AB = np.array([
            [ A11*B22 - A12*B21 - A21*B12 + A22*B11, A12*B20 - A10*B22 + A20*B12 - A22*B10, A10*B21 - A11*B20 - A20*B11 + A21*B10],
            [ A02*B21 - A01*B22 + A21*B02 - A22*B01, A00*B22 - A02*B20 - A20*B02 + A22*B00, A01*B20 - A00*B21 + A20*B01 - A21*B00],
            [ A01*B12 - A02*B11 - A11*B02 + A12*B01, A02*B10 - A00*B12 + A10*B02 - A12*B00, A00*B11 - A01*B10 - A10*B01 + A11*B00]
            ])
 
    elif A.shape==(2,2) and B.shape==(2,2):
        AB[2,2] = A00*B11 - A01*B10 - A10*B01 + A11*B00

    else:
        raise ValueError('Incompatible dimensions. Input matrices must be either 3x3 or 2x2')

    return AB


def SecondTensor2Vector(A):

    # Check size of the matrix
    if A.shape[0]>3 or A.shape[0]==1 or A.shape[0]!=A.shape[1]:
        raise ValueError('Only square 2x2 and 3x3 matrices can be transformed to Voigt vector form')
    if A.shape[0]==3:
        # Matrix is symmetric
        vecA = np.array([
            A[0,0],A[1,1],A[2,2],A[0,1],A[0,2],A[1,2]
            ])
        # Check for symmetry
        if ~np.allclose(A.T, A, rtol=1e-12,atol=1e-15):
            # Matrix is non-symmetric
            vecA = np.array([
                A[0,0],A[1,1],A[2,2],A[0,1],A[0,2],A[1,2],A[1,0],A[2,0],A[2,1]
                ])
    elif A.shape[0]==2:
        # Matrix is symmetric
        vecA = np.array([
            A[0,0],A[1,1],A[0,1]
            ])
        # Check for symmetry
        if ~np.allclose(A.T, A, rtol=1e-12,atol=1e-15):
            # Matrix is non-symmetric
            vecA = np.array([
                A[0,0],A[1,1],A[0,1],A[1,0]
                ])

    return vecA


def Voigt(A,sym=0):
    # Given a 4th order tensor A, puts it in 6x6 format
    # Given a 3rd order tensor A, puts it in 3x6 format (symmetric wrt the first two indices)
    # Given a 2nd order tensor A, puts it in 1x6 format

    # sym returns the symmetrised tensor (only for 3rd and 4th order). Switched off by default.

    # GET THE DIMESNIONS
    if len(A.shape)==4:
        # GIVEN TENSOR IS FOURTH ORDER
        # C=A
        if sym:
            VoigtA = voigt_sym(A)

        #   if C.shape[0]==3:
        #       VoigtA = 0.5*np.array([
        #           [2*C[0,0,0,0],2*C[0,0,1,1],2*C[0,0,2,2],C[0,0,0,1]+C[0,0,1,0],C[0,0,0,2]+C[0,0,2,0],C[0,0,1,2]+C[0,0,2,1]],
        #           [0           ,2*C[1,1,1,1],2*C[1,1,2,2],C[1,1,0,1]+C[1,1,1,0],C[1,1,0,2]+C[1,1,2,0],C[1,1,1,2]+C[1,1,2,1]],
        #           [0           ,0           ,2*C[2,2,2,2],C[2,2,0,1]+C[2,2,1,0],C[2,2,0,2]+C[2,2,2,0],C[2,2,1,2]+C[2,2,2,1]],
        #           [0           ,0           ,0           ,C[0,1,0,1]+C[0,1,1,0],C[0,1,0,2]+C[0,1,2,0],C[0,1,1,2]+C[0,1,2,1]],
        #           [0           ,0           ,0           ,0                    ,C[0,2,0,2]+C[0,2,2,0],C[0,2,1,2]+C[0,2,2,1]],
        #           [0           ,0           ,0           ,0                    ,0                    ,C[1,2,1,2]+C[1,2,2,1]]
        #           ])

        #   else:
        #       VoigtA = 0.5*np.array([
        #           [2*C[0,0,0,0],2*C[0,0,1,1],C[0,0,0,1]+C[0,0,1,0]],
        #           [0           ,2*C[1,1,1,1],C[1,1,0,1]+C[1,1,1,0]],
        #           [0           ,0           ,C[0,1,0,1]+C[0,1,1,0]]
        #           ])

        #   VoigtA = VoigtA+VoigtA.T 
        #   for i in range(0,VoigtA.shape[0]):
        #       VoigtA[i,i] = VoigtA[i,i]/2.0

            

        else:
            C=A
            if C.shape[0]==3:
                VoigtA = np.array([
                    [C[0,0,0,0],C[0,0,1,1],C[0,0,2,2],C[0,0,0,1],C[0,0,0,2],C[0,0,1,2]],
                    [C[1,1,0,0],C[1,1,1,1],C[1,1,2,2],C[1,1,0,1],C[1,1,0,2],C[1,1,1,2]],
                    [C[2,2,0,0],C[2,2,1,1],C[2,2,2,2],C[2,2,0,1],C[2,2,0,2],C[2,2,1,2]],
                    [C[0,1,0,0],C[0,1,1,1],C[0,1,2,2],C[0,1,0,1],C[0,1,0,2],C[0,1,1,2]],
                    [C[0,2,0,0],C[0,2,1,1],C[0,2,2,2],C[0,2,0,1],C[0,2,0,2],C[0,2,1,2]],
                    [C[1,2,0,0],C[1,2,1,1],C[1,2,2,2],C[1,2,0,1],C[1,2,0,2],C[1,2,1,2]]
                    ])
            elif C.shape[0]==2:
                VoigtA = np.array([
                    [C[0,0,0,0],C[0,0,1,1],C[0,0,0,1]],
                    [C[1,1,0,0],C[1,1,1,1],C[1,1,0,1]],
                    [C[0,1,0,0],C[0,1,1,1],C[0,1,0,1]]
                    ])

    elif len(A.shape)==3:
        e=A
        if e.shape[0]==3:
            if ~sym:
                VoigtA = 0.5*np.array([
                    [2.*e[0,0,0],2.*e[1,0,0],2.*e[2,0,0]],
                    [2.*e[0,1,1],2.*e[1,1,1],2.*e[2,1,1]],
                    [2.*e[0,2,2],2.*e[1,2,2],2.*e[2,2,2]],
                    [e[0,0,1]+e[0,1,0],e[1,0,1]+e[1,1,0],e[2,0,1]+e[2,1,0]],
                    [e[0,0,2]+e[0,2,0],e[1,0,2]+e[1,2,0],e[2,0,2]+e[2,2,0]],
                    [e[0,1,2]+e[0,2,1],e[1,1,2]+e[1,2,1],e[2,1,2]+e[2,2,1]]
                    ])
            else:
                VoigtA = np.array([
                    [e[0,0,0],e[1,0,0],e[2,0,0]],
                    [e[0,1,1],e[1,1,1],e[2,1,1]],
                    [e[0,2,2],e[1,2,2],e[2,2,2]],
                    [e[0,0,1],e[1,0,1],e[2,0,1]],
                    [e[0,0,2],e[1,0,2],e[2,0,2]],
                    [e[0,1,2],e[1,1,2],e[2,1,2]]
                    ])
        elif e.shape[0]==2:
            if ~sym:
                VoigtA = 0.5*np.array([
                    [2.*e[0,0,0],2.*e[1,0,0]],
                    [2.*e[0,1,1],2.*e[1,1,1]],
                    [e[0,0,1]+e[0,1,0],e[1,0,1]+e[1,1,0]]
                    ])
            else:
                VoigtA = np.array([
                    [e[0,0,0],e[1,0,0]],
                    [e[0,1,1],e[1,1,1]],
                    [e[0,0,1],e[1,0,1]]
                    ])

    elif len(A.shape)==2:
        VoigtA = SecondTensor2Vector(A)

    else:
        VoigtA = np.array([])

    return VoigtA




def UnVoigt(v):
    A = []
    if len(v.shape)==2:
        if v.shape[1]>1:
            pass
            # DO IT LATER FOR shape>1
        elif v.shape[1]==1:
            # VECTORS TO SYMMETRIC 2ND ORDER TENSORS
            if v.shape[0]==6:
                A = np.array([
                    [v[0,0],v[3,0],v[4,0]],
                    [v[3,0],v[1,0],v[5,0]],
                    [v[4,0],v[5,0],v[2,0]]
                    ])
            elif v.shape[0]==3:
                A = np.array([
                    [v[0,0],v[2,0]],
                    [v[2,0],v[1,0]]
                    ])

    elif len(v.shape)==1:
        # VECTORS TO SYMMETRIC 2ND ORDER TENSORS
        if v.shape[0]==6:
            A = np.array([
                [v[0],v[3],v[4]],
                [v[3],v[1],v[5]],
                [v[4],v[5],v[2]]
                ])
        elif v.shape[0]==3:
            A = np.array([
                [v[0],v[2]],
                [v[2],v[1]]
                ])

    return A 






def IncrementallyLinearisedStress(Stress_k,H_Voigt_k,I,strain,Gradu):
    # IN PRINCIPLE WE NEED GRADU AND NOT STRAIN FOR V_strain
    V_strain =  Voigt(strain)[:,None]
                    # STRESS                        HESSSIAN 'I_W:GRADU'
    return np.dot(Stress_k,(I+strain)) + UnVoigt( np.dot(H_Voigt_k,V_strain) )
    









########################################################################################################
# Note that these matrices are all symmetrised
def AijBkl(A,B):

    A=1.0*A; B=1.0*B
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]
    
    Tens = 1.0*np.array([ 
        [ A00*B00, A00*B11, A00*B22, A00*B01, A00*B02, A00*B12],
        [ A11*B00, A11*B11, A11*B22, A11*B01, A11*B02, A11*B12],
        [ A22*B00, A22*B11, A22*B22, A22*B01, A22*B02, A22*B12],
        [ A01*B00, A01*B11, A01*B22, A01*B01, A01*B02, A01*B12],
        [ A02*B00, A02*B11, A02*B22, A02*B01, A02*B02, A02*B12],
        [ A12*B00, A12*B11, A12*B22, A12*B01, A12*B02, A12*B12]
        ])


    return Tens 


def AikBjl(A,B):

    A=1.0*A; B=1.0*B
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

    Tens = 1.0*np.array([
        [ A00*B00, A01*B01, A02*B02, A00*B01, A00*B02, A01*B02],
        [ A10*B10, A11*B11, A12*B12, A10*B11, A10*B12, A11*B12],
        [ A20*B20, A21*B21, A22*B22, A20*B21, A20*B22, A21*B22],
        [ A00*B10, A01*B11, A02*B12, A00*B11, A00*B12, A01*B12],
        [ A00*B20, A01*B21, A02*B22, A00*B21, A00*B22, A01*B22],
        [ A10*B20, A11*B21, A12*B22, A10*B21, A10*B22, A11*B22]
        ])


    return Tens


def AilBjk(A,B):

    A=1.0*A; B=1.0*B
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    B00=B[0,0]; B11=B[1,1]; B22=B[2,2]; B01=B[0,1]; B02=B[0,2]; B12=B[1,2]; B10=B[1,0]; B20=B[2,0]; B21=B[2,1]

    Tens = 1.0*np.array([
        [ A00*B00, A01*B01, A02*B02, A01*B00, A02*B00, A02*B01],
        [ A10*B10, A11*B11, A12*B12, A11*B10, A12*B10, A12*B11],
        [ A20*B20, A21*B21, A22*B22, A21*B20, A22*B20, A22*B21],
        [ A00*B10, A01*B11, A02*B12, A01*B10, A02*B10, A02*B11],
        [ A00*B20, A01*B21, A02*B22, A01*B20, A02*B20, A02*B21],
        [ A10*B20, A11*B21, A12*B22, A11*B20, A12*B20, A12*B21]
        ])


    return Tens


# A TENSOR AND A VECTOR
def AijUk(A,U):

    A=1.0*A; U=1.0*U
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    U0=U[0]; U1=U[1]; U2=U[2]

    Tens = 1.0*np.array([
        [ A00*U0, A01*U1, A02*U2, A00*U1, A00*U2, A01*U2],
        [ A10*U0, A11*U1, A12*U2, A10*U1, A10*U2, A11*U2],
        [ A20*U0, A21*U1, A22*U2, A20*U1, A20*U2, A21*U2]
        ])

    return Tens


def AikUj(A,U):

    A=1.0*A; U=1.0*U
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    U0=U[0]; U1=U[1]; U2=U[2]

    Tens = 1.0*np.array([
        [ A00*U0, A01*U1, A02*U2, A01*U0, A02*U0, A02*U1],
        [ A10*U0, A11*U1, A12*U2, A11*U0, A12*U0, A12*U1],
        [ A20*U0, A21*U1, A22*U2, A21*U0, A22*U0, A22*U1]
        ])


    return Tens


def UiAjk(U,A):

    A=1.0*A; U=1.0*U
    A00=A[0,0]; A11=A[1,1]; A22=A[2,2]; A01=A[0,1]; A02=A[0,2]; A12=A[1,2]; A10=A[1,0]; A20=A[2,0]; A21=A[2,1]
    U0=U[0]; U1=U[1]; U2=U[2]

    Tens = 1.0*np.array([
        [ A00*U0, A11*U0, A22*U0, A01*U0, A02*U0, A12*U0],
        [ A00*U1, A11*U1, A22*U1, A01*U1, A02*U1, A12*U1],
        [ A00*U2, A11*U2, A22*U2, A01*U2, A02*U2, A12*U2]
        ])


    return Tens






#-------------------------------------------------------------------------#
# UTILITY FUNCTIONS

def unique2d(arr,axis=1,consider_sort=False,order=True,return_index=False,return_inverse=False):
    """Get unique values along an axis for a 2D array.
        see: http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array

        input:

            arr:
                2D array

            axis:
                Axis along which to take unique values, for instance unique 
                rows (axis=1) or unique columns (axis=0). The axis ordering 
                should not be confused with the usual numpy style axis argument, 
                as here since unique values of a 1D type array is finally computed, 
                numpy style axis hence becomes meaningless. Hence, in this context 
                axis=0 implies finding unique rows of an array (lumping the rows) and 
                axis=1 implies finding unique columns of an array (lumping the columns)

            consider_sort:
                Does permutation of the values in row/column matter. Two rows/columns
                can have the same elements but with different arrangements. If consider_sort 
                is True then those rows/columns would be considered equal

            order:
                Similar to 1D unique in numpy, wherein the unique values are always sorted,
                if order is True, unique2d will also sort the values

            return_index:
                Similar to numpy unique. If order is True the indices would be sorted

            return_inverse:
                Similar to numpy unique

        returns:

            2D array of unique values
            If return_index is True also returns indices
            If return_inverse is True also returns the inverse array 

            """

    if axis == 0:
        arr = np.copy(arr.T,order='C')

    if consider_sort is True:
        a = np.sort(arr,axis=1)
    else:
        a = arr
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))

    if return_inverse is False:
        _, idx = np.unique(b, return_index=True)
    else:
        _, idx, inv = np.unique(b, return_index=True, return_inverse=True)

    if order is True:
        idx.sort()

    if return_index == False and return_inverse == False:
        return arr[idx]
    elif return_index == True and return_inverse == False:
        return arr[idx], idx
    elif return_index == False and return_inverse == True:
        return arr[idx], inv
    elif return_index == True and return_inverse == True:
        return arr[idx], idx, inv
    else:
        return arr[idx]



def in2d(arr1, arr2, axis=1, consider_sort=False):
    """Generalisation of numpy.in1d to 2D arrays

        NOTE: arr_1 and arr_2 should have the same dtype

        input:

            arr1:
                2D array

            arr2:
                2D array

            axis:
                Axis along which to take unique values, for instance unique 
                rows (axis=1) or unique columns (axis=0). The axis ordering 
                should not be confused with the usual numpy style axis argument, 
                as here since unique values of a 1D type array is finally computed, 
                numpy style axis hence becomes meaningless. Hence, in this context 
                axis=0 implies finding unique rows of an array (lumping the rows) and 
                axis=1 implies finding unique columns of an array (lumping the columns)

            consider_sort:
                Does permutation of the values in row/column matter. Two rows/columns
                can have the same elements but with different arrangements. If consider_sort 
                is True then those rows/columns would be considered equal

        returns:

            1D boolean array of the same length as arr1

            """

    assert arr1.dtype == arr2.dtype

    if axis == 0:
        arr1 = np.copy(arr1.T,order='C')
        arr2 = np.copy(arr2.T,order='C')
    
    if consider_sort is True:
        sorter_arr1 = np.argsort(arr1)
        arr1 = arr1[np.arange(arr1.shape[0])[:,None],sorter_arr1]
        sorter_arr2 = np.argsort(arr2)
        arr2 = arr2[np.arange(arr2.shape[0])[:,None],sorter_arr2]
        
    arr1_view = np.ascontiguousarray(arr1).view(np.dtype((np.void, arr1.dtype.itemsize * arr1.shape[1])))
    arr2_view = np.ascontiguousarray(arr2).view(np.dtype((np.void, arr2.dtype.itemsize * arr2.shape[1])))
    intersected = np.in1d(arr1_view, arr2_view)
    return intersected.view(np.bool).reshape(-1)


def intersect2d(arr1, arr2,axis=1, consider_sort=False):
    """Generalisation of numpy.intersect1d to 2D arrays

        NOTE: arr_1 and arr_2 should have the same dtype
    
        input:

            arr1:
                2D array

            arr2:
                2D array

            axis:
                Axis along which to take unique values, for instance unique 
                rows (axis=1) or unique columns (axis=0). The axis ordering 
                should not be confused with the usual numpy style axis argument, 
                as here since unique values of a 1D type array is finally computed, 
                numpy style axis hence becomes meaningless. Hence, in this context 
                axis=0 implies finding unique rows of an array (lumping the rows) and 
                axis=1 implies finding unique columns of an array (lumping the columns)

            consider_sort:
                Does permutation of the values in row/column matter. Two rows/columns
                can have the same elements but with different arrangements. If consider_sort 
                is True then those rows/columns would be considered equal

        returns:

            2D array of intersection
            
            """

    assert arr1.dtype == arr2.dtype
    
    if axis == 0:
        arr1 = np.copy(arr1.T,order='C')
        arr2 = np.copy(arr2.T,order='C')
    
    if consider_sort is True:
        sorter_arr1 = np.argsort(arr1)
        arr1 = arr1[np.arange(arr1.shape[0])[:,None],sorter_arr1]
        sorter_arr2 = np.argsort(arr2)
        arr2 = arr2[np.arange(arr2.shape[0])[:,None],sorter_arr2]

    arr1_view = np.ascontiguousarray(arr1).view(np.dtype((np.void, arr1.dtype.itemsize * arr1.shape[1])))
    arr2_view = np.ascontiguousarray(arr2).view(np.dtype((np.void, arr2.dtype.itemsize * arr2.shape[1])))
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])


def shuffle_along_axis(A,B,axis=1,consider_sort=False):
    """Given to 2D arrays A and B, shuffles B along an specified
        axis (rows or columns) such that B==A 

        input:
            A:
                2D array

            B:
                2D array to shuffle

            axis:
                Axis along which to take unique values, for instance unique 
                rows (axis=1) or unique columns (axis=0). The axis ordering 
                should not be confused with the usual numpy style axis argument, 
                as here since unique values of a 1D type array is finally computed, 
                numpy style axis hence becomes meaningless. Hence, in this context 
                axis=0 implies finding unique rows of an array (lumping the rows) and 
                axis=1 implies finding unique columns of an array (lumping the columns)

            consider_sort:
                Does permutation of the values in row/column matter. Two rows/columns
                can have the same elements but with different arrangements. If consider_sort 
                is True then those rows/columns would be considered equal


        returns:
            A_to_B: 
                A mapper such that A[A_to_B] == B 
    """

    assert A.dtype == B.dtype
    assert A.shape == B.shape

    if axis == 0:
        A = np.copy(A.T,order='C')
        B = np.copy(B.T,order='C')

    if consider_sort is True:
        sorter_arr1 = np.argsort(A)
        A = A[np.arange(A.shape[0])[:,None],sorter_arr1]
        sorter_arr2 = np.argsort(B)
        B = B[np.arange(B.shape[0])[:,None],sorter_arr2]


    rowtype = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    a = np.ascontiguousarray(A).view(rowtype).ravel()
    b = np.ascontiguousarray(B).view(rowtype).ravel()

    A_to_As = np.argsort(a)

    As_to_B = a[A_to_As].searchsorted(b)
    A_to_B = A_to_As[As_to_B]

    return A_to_B