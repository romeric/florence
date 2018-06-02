import numpy as np

def InteriorPostProcess(total_sol,internal_points,global_coord,element_connectivity,w,z,boundary_elements,C,dN,Basis,Jacobian, nx, ny, XCO, YCO):
    # Computing potential and flux - Interiors

    POT = np.zeros((internal_points.shape[0],1))
    FLUX1 = np.zeros((internal_points.shape[0],1))
    FLUX2 = np.zeros((internal_points.shape[0],1))

    # Loop over collocation points
    for j in range(0,internal_points.shape[0]):
        XP = internal_points[j,0];      YP = internal_points[j,1]
        # Loop over elements
        # for elem in range(0,len(boundary_elements)):
        for elem in range(0,boundary_elements.shape[0]):
            # Loop over nodes of the element
            for i in range(0,C+2):
                # Carry out usual Gaussian integration
                A=0; B=0
                DU1 = 0; DU2=0; DQ1=0; DQ2=0
                for g in range(0,w.shape[0]):
                    # Compute the radial distance
                    RA = np.sqrt((XCO[g,elem]-XP)**2+(YCO[g,elem]-YP)**2)

                    # Compute Kernels - Assuming both sides are multiplied by 2pi
                    K1 = (-1.0/(RA**2))*((XP-XCO[g,elem])*nx[g,elem]+(YP-YCO[g,elem])*ny[g,elem])
                    K2 = np.log(1.0/RA)

                    RD1 = (XCO[g,elem]-XP)/RA
                    RD2 = (YCO[g,elem]-YP)/RA

                    # For potential
                    A+= K1*Basis[i,g]*Jacobian[g,elem]*w[g]
                    B+= K2*Basis[i,g]*Jacobian[g,elem]*w[g]

                    # Derivatives of potential along x and y
                    DU1 +=(1.0/RA**2)*(XCO[g,elem]-XP)*Basis[i,g]*Jacobian[g,elem]*w[g]
                    DU2 +=(1.0/RA**2)*(YCO[g,elem]-YP)*Basis[i,g]*Jacobian[g,elem]*w[g]

                    # Derivatives of flux along x and y
                    DQ1 += -((2.0*(RD1**2)-1.0)*nx[g,elem]+2.0*RD1*RD2*ny[g,elem])*Basis[i,g]*w[g]*Jacobian[g,elem]/(RA**2)
                    DQ2 += -((2.0*(RD2**2)-1.0)*ny[g,elem]+2.0*RD1*RD2*nx[g,elem])*Basis[i,g]*w[g]*Jacobian[g,elem]/(RA**2)

                POT[j] += total_sol[element_connectivity[elem,i],0]*A-total_sol[element_connectivity[elem,i],1]*B
                FLUX1[j] += total_sol[element_connectivity[elem,i],1]*DU1-total_sol[element_connectivity[elem,i],0]*DQ1
                FLUX2[j] += total_sol[element_connectivity[elem,i],1]*DU2-total_sol[element_connectivity[elem,i],0]*DQ2

        # Divide by 2pi
        POT[j] = POT[j]/2.0/np.pi
        FLUX1[j] = FLUX1[j]/2.0/np.pi
        FLUX2[j] = FLUX2[j]/2.0/np.pi


    return POT, FLUX1, FLUX2





def GetTotalSolution(sol,boundary_data,LHS2LHS,RHS2LHS):

    total_sol = np.copy(boundary_data)
    total_sol[np.array(LHS2LHS,dtype=int),0] = sol[np.array(LHS2LHS,dtype=int),0]
    total_sol[np.array(RHS2LHS,dtype=int),1] = sol[np.array(RHS2LHS,dtype=int),0]

    return total_sol