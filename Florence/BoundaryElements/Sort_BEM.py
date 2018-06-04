from __future__ import print_function, division
import numpy as np

def Sort_BEM(boundary_data,stiffness_K1,stiffness_K2):

    # SORTKERNEL MATRICES
    LHS2LHS=[]; LHS2RHS=[]; RHS2RHS=[]; RHS2LHS=[]
    # Count number of columns going back and forth
    for i in range(0,boundary_data.shape[0]):
        # Known variables going to RHS
        if boundary_data[i,0]!=-1:
            LHS2RHS=np.append(LHS2RHS,i)
        # Unknown variables remaining at LHS
        else:
            LHS2LHS = np.append(LHS2LHS,i)
        # Unknown variables going to LHS
        if boundary_data[i,1]==-1:
            RHS2LHS = np.append(RHS2LHS,i)
        # Known variables remaining at RHS
        else:
            RHS2RHS = np.append(RHS2RHS,i)

    LHS2RHS = np.copy(LHS2RHS).astype(np.int64)
    RHS2RHS = np.copy(RHS2RHS).astype(np.int64)


    # Make total LHS and RHS sizes and which columns comes from where
    total_LHS = np.zeros((LHS2LHS.shape[0]+RHS2LHS.shape[0],2),dtype=np.int64)
    total_RHS = np.zeros((LHS2RHS.shape[0]+RHS2RHS.shape[0],2),dtype=np.int64)

    # In second columns of total_LHS and total_RHS a 1 means in-place columns and a 0 means
    # columns coming from the other side

    # Sort LHS
    total_LHS[0:,0] = np.concatenate((LHS2LHS,RHS2LHS))
    for i in range(0,LHS2LHS.shape[0]):
        total_LHS[i,1] = 1
    sort_indices_LHS = np.argsort(total_LHS[0:,0])
    total_LHS[0:,0] = np.sort(total_LHS[0:,0])
    total_LHS[0:,1] = total_LHS[sort_indices_LHS,1]


    # Sort RHS
    total_RHS[0:,0] = np.concatenate((RHS2RHS,LHS2RHS))
    for i in range(0,RHS2RHS.shape[0]):
        total_RHS[i,1] = 1
    sort_indices_RHS = np.argsort(total_RHS[0:,0])
    total_RHS[0:,0] = np.sort(total_RHS[0:,0])
    total_RHS[0:,1] = total_RHS[sort_indices_RHS,1]

    # Now build global kernel matrices
    # Make two other kernel matrices knowing the dimensions from LHS2RHS, RHS2LHS and so on
    global_K1 = np.zeros((total_LHS.shape[0],total_LHS.shape[0]))
    global_K2 = np.zeros((total_RHS.shape[0],total_RHS.shape[0]))

    for i in range(0,total_LHS.shape[0]):
        if total_LHS[i,1]==1:
            global_K1[0:,i] = stiffness_K1[0:,total_LHS[i,0]]
        else:
            global_K1[0:,i] = -stiffness_K2[0:,total_LHS[i,0]]

    for i in range(0,total_RHS.shape[0]):
        if total_RHS[i,1]==1:
            global_K2[0:,i] = stiffness_K2[0:,total_RHS[i,0]]
        else:
            global_K2[0:,i] = -stiffness_K1[0:,total_RHS[i,0]]


    # Make the RHS vector (vector of known variables) to be multiplied with RHS matrix
    RHS_vector = np.zeros((total_RHS.shape[0],1))
    for i in range(0,total_RHS.shape[0]):
        if total_RHS[i,1]==1:
            RHS_vector[i,0] = boundary_data[total_RHS[i,0],1]
        else:
            RHS_vector[i,0] = boundary_data[total_RHS[i,0],0]

    mm=global_K2.dot(RHS_vector) # usual matrix-vector product (checked)

    return global_K1,mm, total_LHS,total_RHS, LHS2LHS, LHS2RHS, RHS2LHS, RHS2RHS