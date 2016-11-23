import numpy as np
def GetCounterClockwiseIndices(C):
    # Returns indices of a matrix counterclockwise for any matrix 
    # Specify the value of C such that C+2 is the matrix size   
    if C>0:
        zeta_index=[]; eta_index=np.zeros((2))
        go_in = C & 0x1
        for m in range(0,C+go_in):
            # print m
            for k in range(0,4):
                if k==0:
                    minlim = m; maxlim = (C+2)-m
                elif k==1:
                    minlim = m+1; maxlim = (C+2)-m
                elif k==2:
                    minlim = (C+2)-(m+1)-1; maxlim = m-1
                elif k==3:
                    minlim = (C+2)-(m+1)-1; maxlim = (m+1)-1
                # if minlim==maxlim:
                # print minlim,maxlim
                if maxlim>minlim:
                    # if m==1:
                        # print range(minlim,maxlim)
                        # pass
                    for i in range(minlim,maxlim):
                        if k==0 or k==3:
                            zeta_index = np.append(zeta_index,i)
                            eta_index = np.append(eta_index,eta_index[-1])
                        elif k==1 or k==3:
                            zeta_index = np.append(zeta_index,zeta_index[-1])
                            eta_index = np.append(eta_index,i)
                if maxlim<minlim:
                    if m==1:
                        # print range(minlim,maxlim,-1)
                        pass
                    for i in range(minlim,maxlim,-1):
                        if k==0 or k==2:
                            zeta_index = np.append(zeta_index,i)
                            eta_index = np.append(eta_index,eta_index[-1])
                        elif k==1 or k==3:
                            zeta_index = np.append(zeta_index,zeta_index[-1])
                            eta_index = np.append(eta_index,i)

        eta_index =  np.delete(eta_index,np.array([0,1]))

        # Erase all the un-necessary entries from the end
        eleminate = zeta_index.shape[0]-(C+2)**2
        if eleminate!=0:
            for i in range(0,eleminate):
                zeta_index = np.delete(zeta_index,-1)
                eta_index = np.delete(eta_index,-1)

    # C=0 case
    elif C==0:
        zeta_index = np.array([0,1,1,0])
        eta_index = np.array([0,0,1,1])


    return np.array(zeta_index,dtype=int), np.array(eta_index,dtype=int)
