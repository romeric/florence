import numpy as np

def NodeArrangementTet(C):

    # NUMBERING IS NOT CONSISTENT WITH FACE NUMBERS
    # # Traversing the tetrahedral only via edges - used for plotting
    # a1,a2,a3,a4 = [],[],[],[]
    # if C==0:
    #     a1 = [1,2,4]
    #     a2 = [1,2,3]
    #     a3 = [1,3,4]
    #     a4 = [2,3,4]
    # elif C==1:
    #     a1 = [1, 5, 2, 9, 4, 8, 1]
    #     a2 = [1, 5, 2, 7, 3, 6, 1]
    #     a3 = [1, 6, 3, 10, 4, 8, 1]
    #     a4 = [2, 7, 3, 10, 4, 9, 2]
    # elif C==2:
    #     a1 = [1, 5, 6, 2, 9, 11, 3, 10, 7, 1]
    #     a2 = [1, 5, 6, 2, 14, 19, 4, 18, 12, 1]
    #     a3 = [2, 9, 11, 3, 17, 20, 4, 19, 14, 2]
    #     a4 = [1, 12, 18, 4, 20, 17, 3, 10, 7, 1]
    # elif C==3:
    #     a1 = [1, 5, 6, 7, 2, 20, 29, 34, 4, 33, 27, 17, 1]
    #     a2 = [1, 8, 12, 15, 3, 16, 14, 11, 2, 7, 6, 5, 1]
    #     a3 = [2, 11, 14, 16, 3, 26, 32, 35, 4, 34, 29, 20, 2]
    #     a4 = [1, 8, 12, 15, 3, 26, 32, 35, 4, 33, 27, 17, 1]
    # elif C==4:
    #     a1 = [1, 5, 6, 7, 8, 2, 27, 41, 50, 55, 4, 54, 48, 38, 23, 1]
    #     a2 = [1, 9, 14, 18, 21, 3, 22, 20, 17, 13, 2, 8, 7, 6, 5, 1]
    #     a3 = [2, 13, 17, 20, 22, 3, 37, 47, 53, 56, 4, 55, 50, 41, 27, 2]
    #     a4 = [1, 9, 14, 18, 21, 3, 37, 47, 53, 56, 4, 54, 48, 38, 23, 1]

    # a1 = np.asarray(a1)
    # a2 = np.asarray(a2)
    # a3 = np.asarray(a3)
    # a4 = np.asarray(a4)

    # a1 -= 1
    # a2 -= 1
    # a3 -= 1
    # a4 -= 1

    # # traversed_edge_numbering_tet = [a1,a2,a3,a4]
    # traversed_edge_numbering_tet = np.array([a2,a1,a3,a4])
    traversed_edge_numbering_tet = None


    # GET FACE NUMBERING ORDER FROM TETRAHEDRAL ELEMENT
    face_0,face_1,face_2,face_3 = [],[],[],[]
    if C==0:
        face_0 = [0,1,2]
        face_1 = [0,1,3]
        face_2 = [0,2,3]
        face_3 = [1,2,3]
    elif C==1:
        face_0 = [0,1,2,4,5,6]
        face_1 = [0,1,3,4,7,8]
        face_2 = [0,2,3,5,7,9]
        face_3 = [1,2,3,6,8,9]
    elif C==2:
        face_0 = [0,1,2,4,5,6,7,8,9,10]
        face_1 = [0,1,3,4,5,11,12,13,17,18]
        face_2 = [0,2,3,6,9,11,14,16,17,19]
        face_3 = [1,2,3,8,10,13,15,16,18,19]
    elif C==3:
        face_0 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15]
        face_1 = [0,1,3,4,5,6,16,17,18,19,26,27,28,32,33]
        face_2 = [0,2,3,7,11,14,16,20,23,25,26,29,31,32,34]
        face_3 = [1,2,3,10,13,15,19,22,24,25,28,30,31,33,34]
    elif C==4:
        face_0 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        face_1 = [0,1,3,4,5,6,7,22,23,24,25,26,37,38,39,40,47,48,49,53,54]
        face_2 = [0,2,3,8,13,17,20,22,27,31,34,36,37,41,44,46,47,50,52,53,55]
        face_3 = [1,2,3,12,16,19,21,26,30,33,35,36,40,43,45,46,49,51,52,54,55]
    elif C==5:
        face_0 = [0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]
        face_1 = [0,1,3,4,5,6,7,8,29,30,31,32,33,34,50,51,52,53,54,65,66,67,68,75,76,77,81,82]
        face_2 = [0,2,3,9,15,20,24,27,29,35,40,44,47,49,50,55,59,62,64,65,69,72,74,75,78,80,81,83]
        face_3 = [1,2,3,14,19,23,26,28,34,39,43,46,48,49,54,58,61,63,64,68,71,73,74,77,79,80,82,83]
    else:

        # THIS IS A FLOATING POINT BASED ALGORITHM
        from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet
        tol=1e-12
        nsize = int((C+2)*(C+3)*(C+4)/6)

        fekete = FeketePointsTet(C)
        all_nodes = np.arange(nsize)
        face_0 = all_nodes[np.where(np.abs(fekete[:,2]+1.)<tol)].tolist()
        face_1 = all_nodes[np.where(np.abs(fekete[:,1]+1.)<tol)].tolist()
        face_2 = all_nodes[np.where(np.abs(fekete[:,0]+1.)<tol)].tolist()
        # THE AREA OF FACE_3 TRIANGLE
        area = np.sqrt(12) 
        # FIND ALL THE NODES LYING ON THIS FACE
        l12 = np.repeat((fekete[1,:] - fekete[2,:])[:,None].T,fekete.shape[0],axis=0)
        l13 = np.repeat((fekete[1,:] - fekete[3,:])[:,None].T,fekete.shape[0],axis=0)
        l23 = np.repeat((fekete[2,:] - fekete[3,:])[:,None].T,fekete.shape[0],axis=0)
        l1p = np.repeat(fekete[1,:][:,None].T,fekete.shape[0],axis=0) - fekete
        l2p = np.repeat(fekete[2,:][:,None].T,fekete.shape[0],axis=0) - fekete
        area_1p2 = np.linalg.norm(np.cross(l12,l1p),axis=1)/2.
        area_1p3 = np.linalg.norm(np.cross(l13,l1p),axis=1)/2.
        area_2p3 = np.linalg.norm(np.cross(l23,l2p),axis=1)/2.
        areas = area_1p2 + area_1p3 + area_2p3

        # GET ORDER OF THE FACE
        face_3 = all_nodes[np.where(np.abs(areas-area)<tol)].tolist()

        # SANITY CHECK
        assert len(face_0) == len(face_1)
        assert len(face_1) == len(face_2)
        assert len(face_2) == len(face_3)


    face_numbering = np.array([face_0,face_1,face_2,face_3])


    return face_numbering, traversed_edge_numbering_tet



def NodeArrangementTri(C):

    # GET EDGE NUMBERING ORDER FROM TRIANGULAR ELEMENT
    edge0 = []; edge1 = []; edge2 = []
    for i in range(0,C):
        edge0 = np.append(edge0,i+3)
        edge1 = np.append(edge1, 2*C+3 +i*C -i*int((i-1)/2)  )
        edge2 = np.append(edge2,C+3 +i*(C+1) -i*int((i-1)/2) )

    # TRAVERSING TRIANGULAR ELEMENT VIA EDGES
    traversed_edge_numbering_tri = np.concatenate(([0],edge0,[1],edge1,[2],edge2,[0])).astype(np.int64)

    edge0 = np.append(np.append(0,1),edge0)
    edge1 = np.append(np.append(1,2),edge1)
    edge2 = np.append(np.append(2,0),edge2[::-1])
    edge_numbering = np.concatenate((edge0[None,:],edge1[None,:],edge2[None,:]),axis=0).astype(np.int64)
  

    return edge_numbering, traversed_edge_numbering_tri



def NodeArrangementQuad(C):
    """Edge node arrangement for quads

                     edge 2
                 _______________
                |               |
                |               |
        edge 3  |               | edge 1
                |               |
                |_______________|

                    edge 0
    """

    # ELEMENT NODE ARRANGEMENT
    linear_bases_idx = np.array([0,(C+1),(C+2)**2-(C+2),(C+2)**2-1])
    element_numbering = np.concatenate((linear_bases_idx, np.delete(np.arange((C+2)**2),linear_bases_idx)))

    # TRAVERSING QUAD ELEMENT VIA EDGES
    traversed_edge_numbering_quad = None

    # EDGE ARRANGEMENT
    # GET EDGE NUMBERING ORDER FROM QUAD ELEMENT
    edge0, edge1, edge2, edge3 = [], [], [], []
    for i in range(C):
        edge0 = np.append(edge0,i+4)
        edge1 = np.append(edge1, 2*C+5 + i*(C+2))
        edge3 = np.append(edge3, C + 4 + i*(C+2))
    
    edge2 = np.arange((C+2)**2-C,(C+2)**2)

    edge0 = np.append(np.append(0,1),edge0)
    edge1 = np.append(np.append(1,2),edge1)
    edge2 = np.append(np.append(2,3),edge2[::-1])
    edge3 = np.append(np.append(3,0),edge3[::-1])

    edge_numbering = np.concatenate((edge0[None,:],edge1[None,:],edge2[None,:],edge3[None,:]),axis=0).astype(np.int64)  

    return edge_numbering, traversed_edge_numbering_quad, element_numbering

