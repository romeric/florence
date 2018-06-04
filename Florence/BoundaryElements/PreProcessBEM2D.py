from __future__ import print_function, division
import numpy as np
import scipy as sp
import warnings, sys

def GenerateCoordinates(boundary_elements,boundary_points,C,z):
    # Generate the entire coordinate system for higher order elements

    from Florence.FunctionSpace.OneDimensional.Line import LagrangeGaussLobatto, Lagrange
    # Get nodal position at the parent coordinate
    # eps = Lagrange(C,0)[2]
    eps = LagrangeGaussLobatto(C,0)[2]
    # Check if Gauss point and interior nodes coincide
    if eps[C]==z[int(z.shape[0]/2)]:
        sys.exit("Error: At least one Gauss point and an interior node coincide. Change the order of integration to one order higher or lower")

    # Allocate
    global_coord = np.zeros((boundary_points.shape[0]+C*boundary_elements.shape[0],2))
    for elem in range(0,len(boundary_elements)):
        # Get the two edge nodes of the element
        node_i = boundary_elements[elem,0]
        node_j = boundary_elements[elem,1]
        # Get their coordinates
        node_i_coord = boundary_points[node_i,0:2]
        node_j_coord = boundary_points[node_j,0:2]
        # Get length of the element
        le = sp.linalg.norm(node_j_coord-node_i_coord)
        angle = np.arccos((node_j_coord[0]-node_i_coord[0])/le)
        # for higher order elements divide the boundary element into corresponding pieces (i.e. interior nodes)
        eps_physical = le*(eps+1)/2.0  # eps_physical contains le, so don't include it

        coord = np.zeros((len(eps_physical),2))
        for i in range(0,len(eps_physical)):
            x_coord = node_i_coord[0]+eps_physical[i]*np.cos(angle)
            if node_j_coord[1] > node_i_coord[1]:
                y_coord = node_i_coord[1]+eps_physical[i]*np.sin(angle)
            else:
                y_coord = node_i_coord[1]-eps_physical[i]*np.sin(angle)

            coord[i,0] = x_coord
            coord[i,1] = y_coord

        global_coord[(C+1)*elem:(C+1)*elem+(C+1),0:] = coord[0:(C+1),0:]

    return global_coord


def CoordsJacobianRadiusatGaussPoints(boundary_elements,global_coord,C,Basis,dN,w):

    # Allocate
    XCO = np.zeros((w.shape[0],boundary_elements.shape[0])); YCO = np.zeros((w.shape[0],boundary_elements.shape[0]))
    nx = np.zeros((w.shape[0],boundary_elements.shape[0])); ny = np.zeros((w.shape[0],boundary_elements.shape[0]))
    Jacobian = np.zeros((w.shape[0],boundary_elements.shape[0]))

    # Loop over elements
    for elem in range(0,boundary_elements.shape[0]):
        # Loop over Gauss points
        for g in range(0,w.shape[0]):
            if elem != boundary_elements[-1,0]:
                XCO[g,elem] = np.dot(Basis[0:,g],global_coord[(C+1)*elem:(C+1)*(elem+1)+1,0])
                YCO[g,elem] = np.dot(Basis[0:,g],global_coord[(C+1)*elem:(C+1)*(elem+1)+1,1])
            elif elem == boundary_elements[-1,0]:
                XCO[g,elem] = np.dot(Basis[0:,g],np.append(global_coord[(C+1)*elem:(C+1)*(elem+1)+1,0],global_coord[0,0]))
                YCO[g,elem] = np.dot(Basis[0:,g],np.append(global_coord[(C+1)*elem:(C+1)*(elem+1)+1,1],global_coord[0,1]))

            # Get the derivatives at Gauss points
            if elem != boundary_elements[-1,0]:
                dx_by_dxi = np.dot(dN[0:,g],global_coord[(C+1)*elem:(C+1)*(elem+1)+1,0])
                dy_by_dxi = np.dot(dN[0:,g],global_coord[(C+1)*elem:(C+1)*(elem+1)+1,1])
            elif elem == boundary_elements[-1,0]:
                dx_by_dxi = np.dot(dN[0:,g],np.append(global_coord[(C+1)*elem:(C+1)*(elem+1)+1,0],global_coord[0,0]))
                dy_by_dxi = np.dot(dN[0:,g],np.append(global_coord[(C+1)*elem:(C+1)*(elem+1)+1,1],global_coord[0,1]))


            # dx_dy_by_dxi = np.dot(dN,coord) # vectorised version
            # Compute Jacobian
            Jacobian[g,elem] = np.sqrt(dx_by_dxi**2+dy_by_dxi**2)
            if Jacobian[g,elem]==0:
                warnings.warn("Jacobian is zero!")

            # Compute unit normal
            nx[g,elem] = 1.0/Jacobian[g,elem]*dy_by_dxi
            ny[g,elem] = -1.0/Jacobian[g,elem]*dx_by_dxi


    return Jacobian, nx, ny, XCO, YCO


def CoordsJacobianRadiusatGaussPoints_LM(boundary_elements,global_coord,C,Basis,dN,w,element_connectivity):

    # Allocate
    XCO = np.zeros((w.shape[0],boundary_elements.shape[0])); YCO = np.zeros((w.shape[0],boundary_elements.shape[0]))
    nx = np.zeros((w.shape[0],boundary_elements.shape[0])); ny = np.zeros((w.shape[0],boundary_elements.shape[0]))
    Jacobian = np.zeros((w.shape[0],boundary_elements.shape[0]))

    # Loop over elements
    for elem in range(0,boundary_elements.shape[0]):
        coord = global_coord[element_connectivity[elem]]
        coord = coord[:,0:-1]
        # Loop over Gauss points
        for g in range(0,w.shape[0]):
            if elem != boundary_elements[-1,0]:
                XCO[g,elem] = np.dot(Basis[0:,g],coord[:,0])
                YCO[g,elem] = np.dot(Basis[0:,g],coord[:,1])
            elif elem == boundary_elements[-1,0]:
                XCO[g,elem] = np.dot(Basis[0:,g],np.append(coord[:,0],coord[0,0]))
                YCO[g,elem] = np.dot(Basis[0:,g],np.append(coord[:,1],coord[0,1]))

            # Get the derivatives at Gauss points
            if elem != boundary_elements[-1,0]:
                dx_by_dxi = np.dot(dN[0:,g],coord[:,0])
                dy_by_dxi = np.dot(dN[0:,g],coord[:,1])
            elif elem == boundary_elements[-1,0]:
                dx_by_dxi = np.dot(dN[0:,g],np.append(coord[:,0],global_coord[0,0]))
                dy_by_dxi = np.dot(dN[0:,g],np.append(coord[:,1],global_coord[0,1]))


            # dx_dy_by_dxi = np.dot(dN,coord) # vectorised version
            # Compute Jacobian
            Jacobian[g,elem] = np.sqrt(dx_by_dxi**2+dy_by_dxi**2)
            if Jacobian[g,elem]==0:
                warnings.warn("Jacobian is zero!")

            # Compute unit normal
            nx[g,elem] = 1.0/Jacobian[g,elem]*dy_by_dxi
            ny[g,elem] = -1.0/Jacobian[g,elem]*dx_by_dxi



    return Jacobian, nx, ny, XCO, YCO