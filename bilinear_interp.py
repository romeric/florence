import numpy as np

def bilinear_interp(data, xp_rows, xp_cols, ninterp_x=50, ninterp_y=15, nfill_rows=3):
    """ Manual bilinear interpolation """

    scaledA = data

    print scaledA.shape, xp_rows.shape, xp_cols.shape, ninterp_x

    cols = scaledA.shape[1]*ninterp_x
    new_scaledA = np.zeros((scaledA.shape[0],cols))
    for i in range(scaledA.shape[0]):
        yp_rows = scaledA[i,:]
        new_scaledA[i,:] = np.interp(np.linspace(0.001,0.5,cols),xp_rows,yp_rows)
    scaledA = new_scaledA

    # return scaledA

    # rows = scaledA.shape[0]*ninterp_y
    # new_scaledA = np.zeros((rows,scaledA.shape[1]))
    # for i in range(scaledA.shape[1]):
    #     yp_cols = scaledA[:,i]
    #     new_scaledA[:,i] = np.interp(np.linspace(1,50,rows),xp_cols,yp_cols)
    #     # new_scaledA[:,i] = np.interp(interpolated_xs,xp,yp)
    # scaledA = new_scaledA

    # for i in range(1,nfill_rows):
    #     # scaledA[:,i] = scaledA[:,0]
    #     scaledA[-i,:] = scaledA[-1,:] 


    return scaledA