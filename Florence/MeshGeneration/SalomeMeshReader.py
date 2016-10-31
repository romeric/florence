from __future__ import with_statement
import imp
import numpy as np 

from _fromfile_reader import NPFROMFILE_LOOP_Cython


#--------------------------------------------------------------------------------------------------#
""" WHICH MESH READER TO CHOOSE:

1. THE PURE PYTHON MESH READER IS REASONABLY FAST FOR VERY SMALL MESH SIZES (E.G. 5000 TRIS, 3000 TETS).

2. THE NUMPY (NPLOADTXT) MESH READER IS MUCH FASTER THAN PURE PYTHON ROUTINE FOR LARGE 
MESHES/FILE SIZES. 1-3 ORDERS MAGNITUDE SPEED UP IS NOT UNCOMMON USING THE NUMPY (NPLOADTXT) MESH READER. 
FOR INSTANCE, FOR A MESH WITH 382526 TRIANGULAR ELEMENTS AND A FILE SIZE OF 20.8MB THE COMPARATIVE TIMING 
OF THE TWO MESH READER ARE APPROXIMATELY 15SECS AND 680SECS, WHICH MAKES THE NUMPY MESH READER 45 TIMES FASTER.

3. THE PANDAS MESH READER IS KEPT OPTIONAL DUE TO DEPENDENCY ISSUE. HOWEVER PANDAS MESH READER IS 2-3X
FASTER THAN NUMPY'S (NPLOADTXT) MESH READER. FOR A MESH SIZE OF 6135253 TRIANGULAR ELEMENTS AND A FILE SIZE OF 
365.6MB THE COMPARATIVE TIMING OF PANDAS VS NUMPY IS APPROXIMATELY 112SECS VS 240SECS. 

4. THE NUMPY (NPFROMFILE) MESH READER IS THE FASTEST OF THE LOT ACHIEVING ALMOST  2-3X 
(IN SIMPLE DATA FILES EVEN AN ORDER OF MAGNITUDE) SPEED-UP OVER THE ALREADY FAST PANDAS MESH READER.
ACCORDING TO NUMPY'S DOCUMENTAION NP.FROMFILE IS A HIGHLY EFFICIENT WAY OF READING BINARY DATA SINCE IT
DOES NOT SAVE ANY METADATA AND AS A RESULT (NP.FORMFILE & NP.TOFILE) SHOULD NOT BE RELIED UPON. 
NOTE THAT THIS MESH READER, READS THE DATA ONLY ONCE AND THEN RELIES ON A PYTHON LOOP TO FIGURE OUT THE 
REST. THE ALGORITHM IS SLOWED DOWN BY THIS PYTHON LOOP. 6135253 TRIANGULAR ELEMENTS AND A FILE SIZE OF 
365.6MB TAKES APPROXIMATELY 88SECS WITH THIS ALGORITHM. 

5. FINALLY THE NUMPY (NPFROMFILE) MESH READER IS ACCELERATED BY A CYTHON ALGORITHM (FOR THE PYTHON LOOP).
WITH THIS LEVEL OF OPTIMISATION UP TO 4 ORDERS OF MAGNITUDE SPEED-UP CAN BE OBTAINED (ONLY FOR THE PYTHON LOOP).
6135253 TRIANGULAR ELEMENTS AND A FILE SIZE OF 365.6MB TAKES APPROXIMATELY 27SECS WITH THIS ALGORITHM."""
#--------------------------------------------------------------------------------------------------#

def NPFROMFILE_LOOP(FileContent,nnode):
    # DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)
    cols1 = np.int64((FileContent[4*nnode+2+1] - 100) + 2); cols2=0
    counter_edge = 0
    for i in range(4*nnode+2+1,FileContent.shape[0]):
        if FileContent[4*nnode+2+1+cols1*counter_edge] == FileContent[4*nnode+2+1]:
            counter_edge +=1
        else:
            cols2 = np.int64((FileContent[4*nnode+2+1+cols1*counter_edge] - 200)+2)
            break
    counter_face = 0
    for i in range(4*nnode+2+1+cols1*counter_edge,FileContent.shape[0]):
        if 4*nnode+2+1+cols1*counter_edge+cols2*counter_face >= FileContent.shape[0]:
            break
        else:
            if FileContent[4*nnode+2+1+cols1*counter_edge+cols2*counter_face] == FileContent[4*nnode+2+1+cols1*counter_edge]:
                counter_face +=1
            else:
                break
    
    nelse_type = np.zeros((2,2),dtype=np.int64); 
    nelse_type[0,0] = np.int64(FileContent[4*nnode+2+1]); nelse_type[1,0] = np.int64(FileContent[4*nnode+2+1+cols1*counter_edge])
    nelse_type[0,1] = counter_edge; nelse_type[1,1]=counter_face

    return nelse_type 

def ReadMesh_NPFROMFILE(filename,MeshType,C=0):
    # GENERIC SALOME MESH READER
    # FILENAME SHOULD BE A STRING
    class mesh(object):
        """build a mesh class"""
        info = MeshType
            

    # READ THE WHOLE FILE
    FileContent = np.fromfile(filename,dtype=np.float64, sep=" ") 
    # GET NO OF NODES AND NELSE (NELES=NO OF FREE EDGES + NO OF FREE FACES + NO OF ELEMENTS)
    mesh.nnode = np.int64(FileContent[0]); nelse = np.int64(FileContent[1])
    # DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)

    # CALL THE NPFROMFILE_LOOP TO DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)
    # PURE PYTHON ALGORITHM
    # nelse_type = NPFROMFILE_LOOP(FileContent,mesh.nnode)
    # CYTHON ALGORITHM
    nelse_type = NPFROMFILE_LOOP_Cython(FileContent,mesh.nnode)
    

    # READ MESH POINTS WITH METADATA
    mesh.points = FileContent[2:4*mesh.nnode+2].reshape(mesh.nnode,4)

    
    # READ FREE EDGES & ELEMENTS FOR 2D MESHES 
    if MeshType=='tri' or MeshType=='quad':
        # READ POINTS
        if np.allclose(mesh.points[:,3],0.):
            mesh.points = mesh.points[:,1:3]
        elif np.allclose(mesh.points[:,1],0):
            mesh.points = mesh.points[:,2:4]
        elif np.allclose(mesh.points[:,2],0):
            mesh.points = mesh.points[:,[1,3]]
            
        # READ EDGES
        edge_cols = np.arange(2,np.int64(nelse_type[0,0]-100)+2)
        mesh.edges = FileContent[(4*mesh.nnode+2):(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1]
        ].reshape(nelse_type[0,1],2+edge_cols.shape[0])[:,2:].astype(np.int64) - 1
        # READ ELEMENTS
        elem_cols = np.arange(2,np.int64(nelse_type[1,0]-200)+2)
        mesh.elements = FileContent[(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1]
        :(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1] + (2+elem_cols.shape[0])*nelse_type[1,1]
        ].reshape(nelse_type[1,1],2+elem_cols.shape[0])[:,2:].astype(np.int64) - 1
        # READ NUMBER OF ELEMENTS IN 2D
        mesh.nelem = nelse_type[1,1]
        mesh.faces = []


    # READ FREE EDGES, FREE FACES & ELEMENTS FOR 3D MESHES
    elif MeshType=='tet' or MeshType=='hex':
        # READ POINTS
        mesh.points = mesh.points[:,1:]

        #------------------------------------------------------------------------------------------------------------------#
        # USAGE OF EDGES IN 3D IS UNCOMMON - SO THE ARE COMMENTED HERE
        # READ EDGES
        edge_cols = np.arange(2,np.int64(nelse_type[0,0]-100)+2)
        mesh.edges = FileContent[(4*mesh.nnode+2):(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1]
        ].reshape(nelse_type[0,1],2+edge_cols.shape[0])[:,2:].astype(np.int64) - 1
        #------------------------------------------------------------------------------------------------------------------#

        # READ FACES
        face_cols = np.arange(2,np.int64(nelse_type[1,0]-200)+2)
        mesh.faces = FileContent[(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1]
        :(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1] + (2+face_cols.shape[0])*nelse_type[1,1]
        ].reshape(nelse_type[1,1],2+face_cols.shape[0])[:,2:].astype(np.int64) - 1
        # GET NUMBER OF ROWS AND COLUMNS OF ELMENT ENTRIES
        cols = nelse - nelse_type[0,1] - nelse_type[1,1] #                                                                            v
        rows = np.int64(FileContent[(4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1] + (2+face_cols.shape[0])*nelse_type[1,1]+(1)])
        nelse_type = np.concatenate((nelse_type,np.array([[rows,cols]])),axis=0)
        # READ ELEMENTS
        elem_cols = np.arange(2,int(nelse_type[2,0]-300)+2)
        mesh.elements = FileContent[int((4*mesh.nnode+2)+(2+edge_cols.shape[0])*nelse_type[0,1] + (2+face_cols.shape[0])*nelse_type[1,1]):
        ].reshape(cols,2+elem_cols.shape[0])[:,2:].astype(np.int64) - 1
        # READ NUMBER OF FREE FACES & ELEMENTS IN 3D
        mesh.nface = nelse_type[1,1]
        mesh.nelem = nelse_type[2,1]
    # READ NUMBER OF FREE EDGES
    mesh.nedge = nelse_type[0,1]
    

    return mesh 






def ReadMesh_PANDAS(filename,MeshType,C=0):
    from scipy.stats import itemfreq
    try:
        imp.find_module('pandas')
        import pandas as pd 
        FOUND = True
    except ImportError:
        FOUND = False
    # GENERIC SALOME MESH READER
    # FILENAME SHOULD BE A STRING
    class mesh(object):
        """docstring for mesh"""
        info = MeshType

    # GET NO OF NODES AND NELSE (NELES=NO OF FREE EDGES + NO OF FREE FACES + NO OF ELEMENTS)
    mesh.nnode, nelse = np.fromfile(filename,dtype=np.int64,count=2, sep=" ") 
    # DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)
    nelse_type = (itemfreq(np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode,usecols=(1,)))).astype(np.int64)
    # READ FREE EDGES & ELEMENTS FOR 2D MESHES 
    if MeshType=='tri' or MeshType=='quad':
        # READ POINTS
        nodeRows = np.zeros((nelse+1),dtype=np.int64);  nodeRows[1:] = np.arange(mesh.nnode+1,mesh.nnode+nelse+1)
        mesh.points = pd.read_csv(filename,dtype=np.float64,sep=' ',skiprows=nodeRows,error_bad_lines=False,header=None,usecols=(1,2)).as_matrix()
        # READ EDGES
        edge_cols = np.arange(2,int(nelse_type[0,0]-100)+2).tolist()
        mesh.edges = pd.read_csv(filename,dtype=np.float64,sep=' ',skiprows=1+mesh.nnode,skipfooter=nelse_type[1,1],usecols=edge_cols,
            header=None, error_bad_lines=False).as_matrix() -1
        # READ ELEMENTS
        elem_cols = np.arange(2,int(nelse_type[1,0]-200)+2)
        mesh.elements = pd.read_csv(filename,dtype=np.int64,sep=' ',skiprows=1+mesh.nnode+nelse_type[0,1],usecols=elem_cols,
            header=None, error_bad_lines=False).as_matrix() -1
        # READ NUMBER OF ELEMENTS IN 2D
        mesh.nelem = nelse_type[1,1]
        mesh.faces = []


    # READ FREE EDGES, FREE FACES & ELEMENTS FOR 3D MESHES
    elif MeshType=='tet' or MeshType=='hex':
        # READ POINTS
        # mesh.points = np.loadtxt(filename,dtype=np.float64,skiprows=1,usecols=(1,2,3) )[:mesh.nnode,:]
        nodeRows = np.zeros((nelse+1),dtype=np.int64);  nodeRows[1:] = np.arange(mesh.nnode+1,mesh.nnode+nelse+1)
        mesh.points = pd.read_csv(filename,dtype=np.float64,sep=' ',skiprows=nodeRows,error_bad_lines=False,header=None,usecols=(1,2,3)).as_matrix()

        #------------------------------------------------------------------------------------------------------------------#
        # USAGE OF EDGES IN 3D IS UNCOMMON - SO THE ARE COMMENTED HERE
        # READ EDGES
        # edge_cols = np.arange(2,int(nelse_type[0,0]-100)+2).tolist()
        # mesh.edges = pd.read_csv(filename,dtype=np.int64,sep=' ',skiprows=1+mesh.nnode,skipfooter=nelse_type[1,1]+nelse_type[2,1],usecols=edge_cols,
        #   header=None, error_bad_lines=False).as_matrix() -1
        #------------------------------------------------------------------------------------------------------------------#

        # READ FACES
        face_cols = np.arange(2,int(nelse_type[1,0]-200)+2).tolist()
        mesh.faces = pd.read_csv(filename,dtype=np.int64,sep=' ',skiprows=1+mesh.nnode+nelse_type[0,1],skipfooter=nelse_type[2,1],usecols=face_cols,
            header=None, error_bad_lines=False).as_matrix() -1
        # READ ELEMENTS
        elem_cols = np.arange(2,int(nelse_type[2,0]-300)+2)
        mesh.elements = pd.read_csv(filename,dtype=np.int64,sep=' ',skiprows=1+mesh.nnode+nelse_type[0,1]+nelse_type[1,1],usecols=elem_cols,
            header=None, error_bad_lines=False).as_matrix() -1
        # READ NUMBER OF FREE FACES & ELEMENTS IN 3D
        mesh.nface = nelse_type[1,1]
        mesh.nelem = nelse_type[2,1]
    # READ NUMBER OF FREE EDGES
    mesh.nedge = nelse_type[0,1]
    

    return mesh 


        


def ReadMesh_NPLOADTXT(filename,MeshType,C=0):
    from scipy.stats import itemfreq
    
    # GENERIC SALOME MESH READER
    # FILENAME SHOULD BE A STRING
    class mesh(object):
        """docstring for mesh"""
        info = MeshType
            
    # GET NO OF NODES AND NELSE (NELES=NO OF FREE EDGES + NO OF FREE FACES + NO OF ELEMENTS)
    mesh.nnode, nelse = np.fromfile(filename,dtype=np.int64,count=2, sep=" ") 
    # DETERMINE MULTIPLICITY OF EACH TYPE (i.e. EDGES, FACES, ELEMENTS)
    nelse_type = (itemfreq(np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode,usecols=(1,)))).astype(int)
    # READ FREE EDGES & ELEMENTS FOR 2D MESHES 
    if MeshType=='tri' or MeshType=='quad':
        # READ POINTS
        mesh.points = np.loadtxt(filename,dtype=np.float64,skiprows=1,usecols=(1,2,3) )[:mesh.nnode,:2]
        # READ EDGES
        edge_cols = np.arange(2,int(nelse_type[0,0]-100)+2)
        mesh.edges = np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode,usecols=edge_cols )[:nelse_type[0,1],:] - 1
        # READ ELEMENTS
        elem_cols = np.arange(2,int(nelse_type[1,0]-200)+2)
        mesh.elements = np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode+nelse_type[0,1],usecols=elem_cols )[:nelse_type[1,1],:] - 1
        # READ NUMBER OF ELEMENTS IN 2D
        mesh.nelem = nelse_type[1,1]
        mesh.faces = []


    # READ FREE EDGES, FREE FACES & ELEMENTS FOR 3D MESHES
    elif MeshType=='tet' or MeshType=='hex':
        # READ POINTS
        mesh.points = np.loadtxt(filename,dtype=np.float64,skiprows=1,usecols=(1,2,3) )[:mesh.nnode,:]

        #------------------------------------------------------------------------------------------------------------------#
        # USAGE OF EDGES IN 3D IS UNCOMMON - SO THE ARE COMMENTED HERE
        # READ EDGES
        # edge_cols = np.arange(2,int(nelse_type[0,0]-100)+2)
        # mesh.edges = np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode,usecols=edge_cols )[:nelse_type[0,1],:] - 1
        #------------------------------------------------------------------------------------------------------------------#

        # READ FACES
        face_cols = np.arange(2,int(nelse_type[1,0]-200)+2)
        mesh.faces = np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode+nelse_type[0,1],usecols=face_cols )[:nelse_type[1,1],:] - 1
        # READ ELEMENTS
        elem_cols = np.arange(2,int(nelse_type[2,0]-300)+2)
        mesh.elements = np.loadtxt(filename,dtype=np.int64,skiprows=1+mesh.nnode+nelse_type[0,1]+nelse_type[1,1],usecols=elem_cols )[:nelse_type[2,1],:] - 1
        # READ NUMBER OF FREE FACES & ELEMENTS IN 3D
        mesh.nface = nelse_type[1,1]
        mesh.nelem = nelse_type[2,1]
    # READ NUMBER OF FREE EDGES
    mesh.nedge = nelse_type[0,1]
    

    return mesh 




#--------------------------------------------------------------------------------------------------#
#                   THIS MESH READER IS FAST FOR RELATIVELY SMALL MESHES/FILE SIZES
#--------------------------------------------------------------------------------------------------#
def ReadMesh_PYTHON(filename,MeshType,C=0):
    # GENERIC SALOME MESH READER
    # FILENAME SHOULD BE A STRING

    # ALTHOUGH THIS ROUTINE IS BUILT TO READ ANY HIGH P MESHES, IN GENERAL 
    # WE ASSUME THAT THE MESH GENERATOR IS GIVING AS A LINEAR MESH, HENCE 
    C = 0 

    nsize_tri = int((C+2.)*(C+3.)/2.)
    nsize_tet = int((C+2.)*(C+3.)*(C+4.)/6.)
    # Sizes
    nsize_1D = C+2; nsize_2D = []; nsize_3D = []
    if MeshType == 'tri':
        nsize_2D = nsize_tri
    if MeshType == 'quad': 
        nsize_2D = (C+2)**2 
    if MeshType == 'tet':
        nsize_2D = nsize_tri
        nsize_3D = nsize_tet
    if MeshType == 'hex': 
        nsize_2D = (C+2)**2 
        nsize_3D = (C+2)**3

    # ALLOCATE   
    mesh_nodes = np.zeros((1,3)); mesh_edges = np.zeros((1,nsize_1D))
    mesh_faces=[]; mesh_elems=[]
    if MeshType=='tri' or MeshType=='quad':
        mesh_elems = np.zeros((1,nsize_2D))
    elif MeshType=='tet' or MeshType=='hex':
        mesh_faces = np.zeros((1,nsize_2D))
        mesh_elems = np.zeros((1,nsize_3D))

    # READ THE FIRST LINE OF THE FILE
    nnode = []
    with open(filename, 'r') as fil:
        first_line = fil.readline()
    nnode = int(first_line.rstrip().split()[0])

    # READ DATA FILE
    line_counter = 0
    for line in open(filename):
        item = line.rstrip()
        plist = item.split()

        if line_counter > 0 and line_counter < nnode + 1:
            # Read the nodal coordinates
            nodes = np.zeros((1,3))
            for node in range(1,len(plist)):
                nodes[0,node-1] = float(plist[node])
            mesh_nodes = np.append(mesh_nodes,nodes,axis=0)
        else:
            # READ CONNECTIVITY AND REST
            # FREE EDGES
            if float(plist[1])==100+(C+2):
                edges = np.zeros((1,(C+2)))
                for edge in range(1,len(plist)):
                    edges[0,edge-2] = int(plist[edge])
                mesh_edges = np.append(mesh_edges,edges,axis=0)
            if MeshType == 'tri' or MeshType == 'quad': 
                # ELEMENTS
                if float(plist[1])==200+nsize_2D:
                    elems = np.zeros((1,nsize_2D))
                    for elem in range(1,len(plist)):
                        elems[0,elem-2] = int(plist[elem])
                    mesh_elems= np.append(mesh_elems,elems,axis=0)
            if MeshType == 'tet' or MeshType == 'hex': 
                # FREE FACES
                if float(plist[1])==200+nsize_2D:
                    faces = np.zeros((1,nsize_2D))
                    for face in range(1,len(plist)):
                        faces[0,face-2] = int(plist[face])
                    mesh_faces = np.append(mesh_faces,faces,axis=0)
                # ELEMENTS
                elif float(plist[1])==300+nsize_3D:
                    elems = np.zeros((1,nsize_3D))
                    for elem in range(1,len(plist)):
                        elems[0,elem-2] = int(plist[elem])
                    mesh_elems = np.append(mesh_elems,elems,axis=0)

        
        line_counter +=1

    mesh_elems = mesh_elems[1:,:]-1
    if MeshType == 'tet' or MeshType == 'hex': 
        mesh_faces = mesh_faces[1:,:]-1
    mesh_edges = mesh_edges[1:,:]-1
    if MeshType == 'tet' or MeshType == 'hex': 
        mesh_nodes = mesh_nodes[1:,:]
    elif MeshType == 'tri' or MeshType == 'quad':
        mesh_nodes = mesh_nodes[1:,:2] 

    class mesh(object):
        # """Construct Mesh"""
        points = mesh_nodes
        elements = np.array(mesh_elems,dtype=int)       
        edges = np.array(mesh_edges,dtype=int)
        faces = np.array(mesh_faces,dtype=int)
        nnode = mesh_nodes.shape[0]-1
        nelem = mesh_elems.shape[0]-1
        info = MeshType

        
    return mesh






#--------------------------------------------------------------------------------------------------#
#   THIS IS THE MAIN MESH READER FUNCTION WHERE A DECISION IS MADE ON WHICH MESH READER TO USE
#--------------------------------------------------------------------------------------------------#
def ReadMesh(filename,MeshType,C=0):
    # FOR THE REMAINING MESH READER ACTIVATE THIS
    # nnode = np.fromfile(filename,dtype=np.int64,count=1, sep=" ")
    # mesh = []
    # if (MeshType=='tri' or MeshType=='tet') and  nnode < 5000:
    #   mesh = ReadMesh_PYTHON(filename,MeshType,C)
    # elif (MeshType=='tri' or MeshType=='tet') and  nnode > 5000:
    #   if FOUND:
    #       mesh = ReadMesh_PANDAS(filename,MeshType,C)
    #   else:
    #       mesh = ReadMesh_NPLOADTXT(filename,MeshType,C)
        

    # if (MeshType=='quad' or MeshType=='hex') and  nnode < 3000:
    #   mesh = ReadMesh_PYTHON(filename,MeshType,C)
    # elif (MeshType=='quad' or MeshType=='hex') and  nnode > 3000:
    #   if FOUND:
    #       mesh = ReadMesh_PANDAS(filename,MeshType,C)
    #   else:
    #       mesh = ReadMesh_NPLOADTXT(filename,MeshType,C)

    # USE THE NPFROMFILE+CYTHON MESH READER
    mesh = ReadMesh_NPFROMFILE(filename,MeshType,C=0)

    # Unsigned 
    # mesh.elements = mesh.elements.astype(np.uint64)
    # mesh.edges = mesh.edges.astype(np.uint64)
    # if MeshType == "tet" or MeshType == "hex":
    #   mesh.faces = mesh.faces.astype(np.uint64)


    return mesh 
    
