from __future__ import print_function, division
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, lil_matrix


def AssemblyBEM2D(C, global_coord, boundary_elements, element_connectivity, dN, Basis, w, z, Jacobian, nx, ny, XCO, YCO, geo_args):

    # Allocate the two kernels
    stiffness_K1 = np.zeros((global_coord.shape[0],global_coord.shape[0]))
    stiffness_K2 = np.zeros((global_coord.shape[0],global_coord.shape[0]))

    # Loop over collocation points
    for j in range(0,global_coord.shape[0]):
        XP = global_coord[j,0];     YP = global_coord[j,1]
        # Loop over elements
        for elem in range(0,boundary_elements.shape[0]):
            # Carry out usual Gaussian integration
            for g in range(0,w.shape[0]):
                # Compute the radial distance
                RA = np.sqrt((XCO[g,elem]-XP)**2+(YCO[g,elem]-YP)**2)
                # Compute Kernels - Assuming both sides are multiplied by 2pi
                K1 = (-1.0/(RA**2))*((XP-XCO[g,elem])*nx[g,elem]+(YP-YCO[g,elem])*ny[g,elem])
                # K2 = np.log(1.0/RA)
                K2 = -np.log(RA)
                # Fill Kernel Matrices
                stiffness_K1[j,element_connectivity[elem,0:]]+= K1*Basis[0:,g]*Jacobian[g,elem]*w[g]
                stiffness_K2[j,element_connectivity[elem,0:]]+= K2*Basis[0:,g]*Jacobian[g,elem]*w[g]


    # Implicit integration for diagonal components of K1
    for cols in range(0,stiffness_K1.shape[0]):
        stiffness_K1[cols,cols] = 1.0*(0-np.sum(stiffness_K1[cols,0:])-stiffness_K1[cols,cols])

    # Make modified stiffness matrix
    mod_stiffness_K1 = np.zeros((stiffness_K1.shape[0]+4,stiffness_K1.shape[1]+4))
    mod_stiffness_K2 = np.zeros((stiffness_K2.shape[0]+4,stiffness_K2.shape[1]+4))

    if geo_args.Lagrange_Multipliers == 'activated':
        # Normal BE blocks
        mod_stiffness_K1[0:global_coord.shape[0],0:global_coord.shape[0]] = stiffness_K1
        mod_stiffness_K2[0:global_coord.shape[0],0:global_coord.shape[0]] = stiffness_K2

        # Off diagonal blocks
        for i in range(0,global_coord.shape[0]):
            if global_coord[i,2]==1 or global_coord[i,2]==2 or global_coord[i,2]==3 or global_coord[i,2]==4:
                mod_stiffness_K1[i,global_coord.shape[0]+np.abs(global_coord[i,2])-1] = 1
            if global_coord[i,2]==-1 or global_coord[i,2]==-2 or global_coord[i,2]==-3 or global_coord[i,2]==-4:
                mod_stiffness_K1[i,global_coord.shape[0]+np.abs(global_coord[i,2])-1] = -1

        mod_stiffness_K1[global_coord.shape[0]:,0:]=mod_stiffness_K1[0:,global_coord.shape[0]:].T

        stiffness_K1 = mod_stiffness_K1
        stiffness_K2 = mod_stiffness_K2


    # # Make dense matrix a sparse matrix as sparse assembly is not efficient
    # stiffness_K1_sparse = lil_matrix((stiffness_K1.shape[0],stiffness_K1.shape[1]))
    # stiffness_K2_sparse = lil_matrix((stiffness_K1.shape[0],stiffness_K1.shape[1]))
    # # kk[:,0]=stiffness_K1[:,0]
    # for i in range(0,16):
    #   for j in range(0,16):
    #       stiffness_K1_sparse[i,j] = stiffness_K1[i,j]
    #       stiffness_K2_sparse[i,j] = stiffness_K2[i,j]


    return stiffness_K1, stiffness_K2






def AssemblyBEM2D_Sparse(C,global_coord,boundary_elements,element_connectivity,dN,Basis,w,z,Jacobian,nx,ny,XCO,YCO,geo_args):


    I_k1 = np.zeros((global_coord.shape[0]*global_coord.shape[0])); J_k1 = np.zeros((global_coord.shape[0]*global_coord.shape[0]));
    V_k1 = np.zeros((global_coord.shape[0]*global_coord.shape[0]))
    # I_k2 = np.zeros((global_coord.shape[0]*global_coord.shape[0]));   J_k2 = np.zeros((global_coord.shape[0]*global_coord.shape[0]));
    V_k2 = np.zeros((global_coord.shape[0]*global_coord.shape[0]))


    # Loop over collocation points
    for j in range(0,global_coord.shape[0]):
        XP = global_coord[j,0];     YP = global_coord[j,1]
        # Loop over elements
        for elem in range(0,boundary_elements.shape[0]):
            for k in range(0,element_connectivity.shape[1]):
                # Carry out usual Gaussian integration
                for g in range(0,w.shape[0]):
                    # Compute the radial distance
                    RA = np.sqrt((XCO[g,elem]-XP)**2+(YCO[g,elem]-YP)**2)
                    # Compute Kernels - Assuming both sides are multiplied by 2pi
                    K1 = (-1.0/(RA**2))*((XP-XCO[g,elem])*nx[g,elem]+(YP-YCO[g,elem])*ny[g,elem])
                    K2 = np.log(1.0/RA)


                    # Fill Kernel Matrices
                    I_k1[j*global_coord.shape[0]+j] = j
                    J_k1[element_connectivity[elem,k]*global_coord.shape[0]+element_connectivity[elem,k]] = element_connectivity[elem,k]
                    V_k1[j*global_coord.shape[0]+j] += K1*Basis[k,g]*Jacobian[g,elem]*w[g]
                    V_k2[j*global_coord.shape[0]+j] += K2*Basis[k,g]*Jacobian[g,elem]*w[g]


    stiffness_K1 = coo_matrix((V_k1,(I_k1,J_k1)),shape=((global_coord.shape[0],global_coord.shape[0]))).tocsc()
    stiffness_K2 = coo_matrix((V_k2,(I_k1,J_k1)),shape=((global_coord.shape[0],global_coord.shape[0]))).tocsc()


    # # Make modified stiffness matrix
    # mod_stiffness_K1 = csc_matrix((stiffness_K1.shape[0]+4,stiffness_K1.shape[1]+4))
    # mod_stiffness_K2 = csc_matrix((stiffness_K1.shape[0]+4,stiffness_K1.shape[1]+4))
    mod_stiffness_K1 = lil_matrix((stiffness_K1.shape[0]+4,stiffness_K1.shape[1]+4))
    mod_stiffness_K2 = lil_matrix((stiffness_K1.shape[0]+4,stiffness_K1.shape[1]+4))

    if geo_args.Lagrange_Multipliers=='activated':
        # Normal BE blocks
        mod_stiffness_K1[0:global_coord.shape[0],0:global_coord.shape[0]] = stiffness_K1
        mod_stiffness_K2[0:global_coord.shape[0],0:global_coord.shape[0]] = stiffness_K2

        # Off diagonal blocks
        for i in range(0,global_coord.shape[0]):
            if global_coord[i,2]==1 or global_coord[i,2]==2 or global_coord[i,2]==3 or global_coord[i,2]==4:
                mod_stiffness_K1[i,global_coord.shape[0]+np.abs(global_coord[i,2])-1] = 1
            if global_coord[i,2]==-1 or global_coord[i,2]==-2 or global_coord[i,2]==-3 or global_coord[i,2]==-4:
                mod_stiffness_K1[i,global_coord.shape[0]+np.abs(global_coord[i,2])-1] = -1

        mod_stiffness_K1[global_coord.shape[0]:,0:]=mod_stiffness_K1[0:,global_coord.shape[0]:].T

        stiffness_K1 = mod_stiffness_K1
        stiffness_K2 = mod_stiffness_K2


    return stiffness_K1, stiffness_K2


