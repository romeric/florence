import numpy as np 

def itemfreq_py(arr=None,un_arr=None,inv_arr=None,decimals=None):

    if (arr is None) and (un_arr is None):
        raise ValueError('No input array to work with. Either the array or its unique with inverse should be provided')
    if un_arr is None:
        if decimals is not None:
            un_arr, inv_arr = np.unique(np.round(arr,decimals=decimals),return_inverse=True)
        else:
            un_arr, inv_arr = np.unique(arr,return_inverse=True)

    if arr is not None:
        if len(arr.shape) > 1:
            dtype = type(arr[0,0])
        else:
            dtype = type(arr[0])
    else:
        if len(un_arr.shape) > 1:
            dtype = type(un_arr[0,0])
        else:
            dtype = type(un_arr[0])

    unf_arr = np.zeros((un_arr.shape[0],2),dtype=dtype)
    unf_arr[:,0] = un_arr
    unf_arr[:,1] = np.bincount(inv_arr)

    return unf_arr