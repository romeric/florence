import numpy as np 
import imp, os


def ApplyNeumannBoundaryConditions3D(general_data,mesh,w,elem,xycoord):

    BoundaryData = general_data.BoundaryData
    Domain = general_data.Domain
    Boundary = general_data.Boundary

    ndim = general_data.ndim
    nvar = general_data.nvar
    C = general_data.C
    nodeperface = mesh.faces.shape[1]
    nodeperelem = mesh.elements.shape[1]
    traction_force = np.zeros(((C+2)**ndim*nvar,1))


    # Find centriod of the element
    centriod_coord = np.zeros(3)
    for i in range(0,ndim):
        centriod_coord[i] = np.sum(xycoord[:,i])/(C+2)**ndim 


    # Identify if element has an edge at the physical boundary
    node_counter=0
    node_coords = np.zeros((1,3))
    node_number = []
    for i in range(0,mesh.elements.shape[1]):
        x = np.where(mesh.faces==mesh.elements[elem,i])[0]
        if x.shape[0]!=0:
            node_counter+=1
            node_coords = np.append(node_coords,mesh.points[mesh.elements[elem,i],:].reshape(1,ndim),axis=0)
            node_number = np.append(node_number,mesh.elements[elem,i])
    node_coords = node_coords[1:,:]
    # node_counter counts the number of free nodes of an element at the boundary
    # if node_counter = 0 -> element is not at the boundary
    # if node_counter = 4 -> element has 1 face at the boundary
    # if node_counter = 6 -> element has 2 face at the boundary
    # if node_counter = 7 -> element has 3 face at the boundary 

    if node_counter!=0:
        for iface in range(0,mesh.faces.shape[0]):
            # This is for selecting individual faces
            face_node_counter = 0
            for i in range(0,mesh.faces.shape[1]):
                for j in range(0,node_coords.shape[0]):
                    if np.array_equal(mesh.points[mesh.faces[iface,i],:],node_coords[j]):
                        face_node_counter+=1
            if face_node_counter==4:
                # Now we are on an specific boundary face of the element

                # Based on numbering of the element find which face are we at
                local_element = []
                for j in range(0,node_number.shape[0]):
                    x1 = np.where(node_number[j]==mesh.elements[elem,:])
                    local_element = np.append(local_element,x1)
                # Now what are the numbers of the current face
                local_face=[]
                for j in range(0, mesh.faces.shape[1]):
                    x2 = np.where( mesh.elements[elem,:]== mesh.faces[iface, j])
                    local_face = np.append(local_face,x2)
                # print local_element, elem
                # print local_face, iface, elem

                xycoord_face = np.zeros((nodeperface, ndim))
                for i in range(0, nodeperface):
                    xycoord_face[i, :] = mesh.points[mesh.faces[iface, i], :]


                if np.where(local_face==0)[0].shape[0]!=0 and np.where(local_face==1)[0].shape[0]!=0 and \
                                np.where(local_face==2)[0].shape[0]!=0 and np.where(local_face==3)[0].shape[0]!=0:
                    # At this face [0 1 2 3] -> beta=-1
                    # Remove dependency on z to integrate within the quad plane (i.e. x-y)
                    index = 4
                    face_index = [0, 1]
                elif np.where(local_face == 0)[0].shape[0] != 0 and np.where(local_face == 1)[0].shape[0] != 0 and \
                                np.where(local_face == 5)[0].shape[0] != 0 and np.where(local_face == 4)[0].shape[
                    0] != 0:
                    # At this face [0 1 5 4] -> eta=-1
                    index = 2
                    face_index = [0, 2]
                elif np.where(local_face == 0)[0].shape[0] != 0 and np.where(local_face == 3)[0].shape[0] != 0 and \
                                np.where(local_face == 7)[0].shape[0] != 0 and np.where(local_face == 4)[0].shape[
                    0] != 0:
                    # At this face [0 3 7 4] -> zeta=-1
                    index = 0
                    face_index = [1, 2]
                elif np.where(local_face == 4)[0].shape[0] != 0 and np.where(local_face == 5)[0].shape[0] != 0 and \
                                np.where(local_face == 6)[0].shape[0] != 0 and np.where(local_face == 7)[0].shape[
                    0] != 0:
                    # At this face [4 5 6 7] -> beta=1
                    index = 5
                    face_index = [0, 1]
                elif np.where(local_face == 6)[0].shape[0] != 0 and np.where(local_face == 7)[0].shape[0] != 0 and \
                                np.where(local_face == 3)[0].shape[0] != 0 and np.where(local_face == 2)[0].shape[
                    0] != 0:
                    # At this face [6 7 3 2] -> eta=1
                    index = 3
                    face_index = [0, 2]
                elif np.where(local_face == 1)[0].shape[0] != 0 and np.where(local_face == 5)[0].shape[0] != 0 and \
                                np.where(local_face == 6)[0].shape[0] != 0 and np.where(local_face == 2)[0].shape[
                    0] != 0:
                    # At this face [1 5 6 2] -> zeta=1
                    index = 1
                    face_index = [1, 2]

                # Call the Neumann Boundary File
                BoundaryData.NeuArgs.points = xycoord_face
                t = BoundaryData().NeumannCriterion(BoundaryData.NeuArgs)

                # Apply Neumann boundary conditions only if traction vector is non-zero
                if np.count_nonzero(t)!=0:

                    B = np.zeros((Domain.Bases.shape[0]*nvar,nvar))
                    counter=0
                    for i in range(0,w.shape[0]):
                        for j in range(0,w.shape[0]):
                            # Get coordinate of Gauss point
                            Coordg = np.dot(Boundary.Basis[:,counter,index].reshape(Boundary.Basis.shape[0],1).T,xycoord)
                            # Gradient matrix of in parent element
                            Jm = np.zeros((ndim, Boundary.Basis[:,:,index].shape[0]))
                            Jm[0, :] = Boundary.gBasisx[:, counter,index]
                            Jm[1, :] = Boundary.gBasisy[:, counter,index]
                            Jm[2, :] = Boundary.gBasisz[:, counter,index]
                            # Jacobian
                            J=np.dot(Jm,xycoord)
                            # Gradient matrix in physical element
                            Jb = np.dot(np.linalg.inv(J),Jm)

                            # Unit normal
                            unit_normal = np.cross(J[:,face_index[0]],
                                J[:,face_index[1]])/np.linalg.norm(np.cross(J[:,face_index[0]],J[:,face_index[1]]))
                            # Just a reformat based on constitutive equation 
                            for k in range(0,nvar):
                                B[k:B.shape[0]:nvar,k] = Boundary.Basis[:,counter,index]

                            # Find unit (outward) normal
                            difference_vector = Coordg.reshape(Coordg.shape[1])-centriod_coord
                            unit_normal = np.sign(np.dot(difference_vector,unit_normal))*unit_normal

                            # Now determine whether traction type or pressure type Neumann BC is applied 

                            # Compute traction force
                            traction_force += np.dot(B,t.T)*w[i]*w[j]*np.abs(np.linalg.det(J))
                            counter+=1



    # p = np.zeros((C+2)**ndim*nvar)
    # return p
    return traction_force[:,0]






# Electromechanical Neumann boundary condition
def ApplyNeumannBoundaryConditions(C,mesh,nmesh,BoundaryData,nvar,Basis,w,z,elem):
    # VERY IMPORTANT TO NOTE THAT THE EDGES GIVEN AS INPUT ARGUMENT TO THIS FUNCTION MUST BE FREE (BOUNDARY) EDGES

    #######################################################
    # Applying Neumann boundary conditions
    P = np.zeros((C+2)**2*nvar)
    # First off - check if it is a boundary edge (An edge where you apply Neumann)

    mesh_points = np.array(mesh.points)
    mesh_elements = np.array(mesh.elements)
    # Get free edges (edges at the boundary) - Check if this is available for all meshes used
    mesh_edges = np.array(mesh.edges)       
    

    counter=0; m=np.zeros((2,1)); n=np.zeros((2,1))
    edge_node_1 = []; edge_node_2=[];
    edge_nodes = []
    for i in range(0,mesh_elements.shape[1]):
        x= np.where(mesh_elements[elem,i]==mesh_edges)
        # Get free edge nodes at this element
        if x[0].shape[0]!=0:
            edge_nodes = np.append(edge_nodes,int(mesh_edges[x[0][0],x[1][0]])) 
    # print edge_nodes

    
    if np.array(edge_nodes).shape[0]!=0:
        # If they exist the perform computation

        edge_nodes = np.array(edge_nodes,dtype=int)
        if edge_nodes.shape[0]==2:
            # You are good to go
            edge = edge_nodes.reshape(edge_nodes.shape[0],1).T
            edge = np.sort(edge)


        elif edge_nodes.shape[0]==3:
            # This element has 3 nodes at boundary - corner element
            # Now find how are these edge nodes connected
            edge = np.zeros((2,1))
            for i in range(0,edge_nodes.shape[0]):
                y = np.where(mesh_edges==edge_nodes[i])
                yy = np.copy(y)
                for j in range(0,2):
                    if y[1][j]==0:
                        yy[1][j]=1
                    elif y[1][j]==1:
                        yy[1][j]=0
                    m = np.where(mesh_edges[y[0][j],yy[1][j]]==edge_nodes)[0]
                    if m.shape[0]!=0:
                        edge = np.append(edge,np.array([[mesh_edges[y[0][j],yy[1][j]], mesh_edges[y[0][j],y[1][j]]]]).T,axis=1)
            edge=edge[:,1:].T
            edge = np.sort(edge)
            edge_1 = edge[0,0]; edge_2=edge[0,1]
            # Romove duplicates
            # Since at the edge corner of a 2D mesh one is common and after sort that node 
            # is going to comprise one column of 'edge' here - print edge to check (right here)
            # Carefull
            c1=0; c2=0; i1=edge[0,1]; i2=edge[0,0];
            for k in range(1,edge.shape[0]):
                if edge[k,0]==edge_1:
                    c1+=1; i1=np.append(i1,edge[k,1])
                if edge[k,1]==edge_2:
                    c2+=1; i2=np.append(i2,edge[k,0])
            if c1==3:
                edge = np.append(i2.reshape(2,1),np.unique(i1).reshape(2,1),axis=1)
            if c2==3:
                edge = np.append(i1.reshape(2,1),np.unique(i2).reshape(2,1),axis=1)
                 
        if mesh_elements.shape[0]==1:
            edge = mesh_edges

        
        # Get OneD Basis
        # Basis_boundary = np.zeros((C+2,w.shape[0])); dBasis_boundary = np.zeros((C+2,w.shape[0]))
        # for g in range(0,w.shape[0]):
        #   Basis_boundary[:,g],dBasis_boundary[:,g],eps = LagrangeGaussLobatto(C,z[g])
        Basis_boundary = np.zeros((2,w.shape[0])); dBasis_boundary = np.zeros((2,w.shape[0]))
        for g in range(0,w.shape[0]):
            Basis_boundary[:,g],dBasis_boundary[:,g],eps = LagrangeGaussLobatto(0,z[g])
            
        # _, nx, ny,_,_ = CoordsJacobianRadiusatGaussPoints(mesh_edges,mesh_points,C,Basis_boundary,dBasis_boundary,w)
        # print edge
        # print nmesh.edges
        # Get coordinates of each edge
        for iedge in range(0,edge.shape[0]):
            coord_node_1 = mesh_points[edge[iedge,0]]
            coord_node_2 = mesh_points[edge[iedge,1]]
            # print coord_node_2
            edge_length = np.linalg.norm(coord_node_2-coord_node_1)

            # Get the position of nodes in the current element (locally)
            local_element_nodes = []
            local_element_nodes = np.append(local_element_nodes,np.where(mesh_elements[elem]==edge[iedge,0]))
            local_element_nodes = np.append(local_element_nodes,np.where(mesh_elements[elem]==edge[iedge,1]))


            # Find the unit normal
            X = np.append(coord_node_1[0],coord_node_2[0])
            Y = np.append(coord_node_1[1],coord_node_2[1])
            # print dBasis_boundary[:,g],X
            # print mesh_points[edge[iedge,0]]
            # print edge[iedge]

            # Get the center of the element
            X_element_center = np.sum(mesh_points[mesh_elements[elem,:],0])/4.0
            Y_element_center = np.sum(mesh_points[mesh_elements[elem,:],1])/4.0

            # Loop over Gauss points
            for g in range(0,w.shape[0]):
                # Get the derivatives at Gauss points
                dx_by_dxi = np.dot(dBasis_boundary[:,g],X)
                dy_by_dxi = np.dot(dBasis_boundary[:,g],Y)
                J = np.sqrt(dx_by_dxi**2+dy_by_dxi**2)

                # Compute unit normal
                nx = 1.0/J*dy_by_dxi
                ny = -1.0/J*dx_by_dxi
                unit_normal = np.append(nx,ny)
                
                # Get coordinates of Gauss point
                X_gauss = np.dot(Basis_boundary[:,g],X)
                Y_gauss = np.dot(Basis_boundary[:,g],Y)

                # Compute the vector going from the center of the elements to the Gauss point
                dx = X_gauss - X_element_center
                dy = Y_gauss - Y_element_center
                vector = np.append(dx,dy)

                # Assign the correct direction to unit normal
                unit_normal = np.sign(np.dot(vector,unit_normal))*unit_normal
                # print np.linalg.norm(unit_normal)
                # print unit_normal, X,Y



            local_element_nodes = np.sort(local_element_nodes)

            BoundaryData().NeuArgs.node1 = coord_node_1
            BoundaryData().NeuArgs.node2 = coord_node_2

            # Okay now that we have edges of the element, apply Neumann boundary criterion
            force = BoundaryData().NeumannCriterion(BoundaryData().NeuArgs)
            # print force

            # For reduced computational cost, compute force vector if force!=0
            if np.count_nonzero(force)!=0:
                
                if C>0:
                    local_element_nodes = np.linspace((C+1)*local_element_nodes[0],
                        (C+1)*local_element_nodes[1],(C+1)*(local_element_nodes[1]-local_element_nodes[0])+1)
                for g in range(0,w.shape[0]):
                    for k in range(0,nvar):
                        P[np.array(local_element_nodes*nvar+k,dtype=int)]+=force[k]*Basis_boundary[:,g]*w[g]*edge_length/2
                        # P[np.array(local_element_nodes*nvar+k,dtype=int)]+=Basis_boundary[:,g]*w[g]*edge_length/2
                        # print nx[g,iedge],ny[g,iedge]

                # for k in range(0,nvar):
                        # P[np.array(local_element_nodes*nvar+k,dtype=int)]*=force[k]

    return P 



















def ApplyNeumannBoundaryConditions_Mesh(C,mesh,boundary_data,nvar,Basis,w,z,elem):
    # VERY IMPORTANT TO NOTE THAT THE EDGES GIVEN AS INPUT ARGUMENT TO THIS FUNCTION MUST BE FREE EDGES

    #######################################################
    # Applying Neumann boundary conditions
    P = np.zeros((C+2)**2*nvar)
    # First off - check if it is a boundary edge (An edge where you apply Neumann)

    mesh_points = np.array(mesh.points)
    mesh_elements = np.array(mesh.elements)
    # Get free edges (edges at the boundary) - Check if this is available for all meshes used
    mesh_edges = np.array(mesh.edges)       


    counter=0; m=np.zeros((2,1)); n=np.zeros((2,1))
    edge_node_1 = []; edge_node_2=[];
    edge_nodes = []
    for i in range(0,mesh_elements.shape[1]):
        x= np.where(mesh_elements[elem,i]==mesh_edges)
        # Get free edge nodes at this element
        if x[0].shape[0]!=0:
            edge_nodes = np.append(edge_nodes,int(mesh_edges[x[0][0],x[1][0]])) 

    if np.array(edge_nodes).shape[0]!=0:
        # If they exist the perform computation

        edge_nodes = np.array(edge_nodes,dtype=int)
        if edge_nodes.shape[0]==2:
            # You are good to go
            edge = edge_nodes.reshape(edge_nodes.shape[0],1).T
            edge = np.sort(edge)


        elif edge_nodes.shape[0]==3:
            # This element has 3 nodes at boundary - corner element
            # Now find how are these edge nodes connected
            edge = np.zeros((2,1))
            for i in range(0,edge_nodes.shape[0]):
                y = np.where(mesh_edges==edge_nodes[i])
                yy = np.copy(y)
                for j in range(0,2):
                    if y[1][j]==0:
                        yy[1][j]=1
                    elif y[1][j]==1:
                        yy[1][j]=0
                    m = np.where(mesh_edges[y[0][j],yy[1][j]]==edge_nodes)[0]
                    if m.shape[0]!=0:
                        edge = np.append(edge,np.array([[mesh_edges[y[0][j],yy[1][j]], 
                            mesh_edges[y[0][j],y[1][j]]]]).T,axis=1)
            edge=edge[:,1:].T
            edge = np.sort(edge)
            edge_1 = edge[0,0]; edge_2=edge[0,1]
            # Romove duplicates
            # Since at the edge corner of a 2D mesh one is common and after sort that node 
            # is going to comprise one column of 'edge' here - print edge to check (right here)
            # Carefull
            c1=0; c2=0; i1=edge[0,1]; i2=edge[0,0];
            for k in range(1,edge.shape[0]):
                if edge[k,0]==edge_1:
                    c1+=1; i1=np.append(i1,edge[k,1])
                if edge[k,1]==edge_2:
                    c2+=1; i2=np.append(i2,edge[k,0])
            if c1==3:
                edge = np.append(i2.reshape(2,1),np.unique(i1).reshape(2,1),axis=1)
            if c2==3:
                edge = np.append(i1.reshape(2,1),np.unique(i2).reshape(2,1),axis=1)
                 
        if mesh_elements.shape[0]==1:
            edge = mesh_edges


        # Get OneD Basis
        Basis_boundary = np.zeros((C+2,w.shape[0])); dBasis_boundary = np.zeros((C+2,w.shape[0]))
        for g in range(0,w.shape[0]):
            Basis_boundary[:,g],dBasis_boundary[:,g],eps = LagrangeGaussLobatto(C,z[g])


        # # for iedge in range(0,edge.shape[0]):
        #   # coord_node_1 = mesh_points[edge[iedge,0]]
        #   # coord_node_2 = mesh_points[edge[iedge,1]]
        #   edge_length = np.linalg.norm(coord_node_2-coord_node_1)

        # unique_edge_nodes = np.unique(edge)
        # for inode in range(0,unique_edge_nodes.shape[0]):
        #   if edge.shape[0]>1:






                




        for iedge in range(0,edge.shape[0]):
            # unique_edge_nodes = np.unique(edge[iedge,:])

            coord_node_1 = mesh_points[edge[iedge,0]]
            coord_node_2 = mesh_points[edge[iedge,1]]
            edge_length = np.linalg.norm(coord_node_2-coord_node_1)

            # Get the position of nodes in the current element (locally)
            local_element_nodes = []
            local_element_nodes = np.append(local_element_nodes,np.where(mesh_elements[elem]==edge[iedge,0]))
            local_element_nodes = np.append(local_element_nodes,np.where(mesh_elements[elem]==edge[iedge,1]))
            # print local_element_nodes


            for inode in range(0,edge.shape[1]):
                coord_node = mesh_points[edge[iedge,inode],:]
                boundary_data().NeuArgs.node = coord_node
                # Okay now that we have edges of the element, apply Neumann boundary criterion
                force = boundary_data().NeumannCriterion(boundary_data().NeuArgs)
                # print force, elem


                # For reduced computational cost, compute force vector if force!=0
                if np.count_nonzero(force)!=0:
                
                    if C>0:
                        local_element_nodes = np.linspace((C+1)*local_element_nodes[0],(C+1)*local_element_nodes[1],
                            (C+1)*(local_element_nodes[1]-local_element_nodes[0])+1)
                    for g in range(0,w.shape[0]):
                        for k in range(0,nvar):
                            P[np.array(local_element_nodes[inode]*nvar+k,dtype=int)]+=force[k]*Basis_boundary[:,g]*w[g]*edge_length/2

    return P 





