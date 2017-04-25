from __future__ import division
import os, sys, warnings
from time import time
import numpy as np 
from scipy.io import loadmat, savemat
from Florence.Tensor import makezero, itemfreq, unique2d, in2d
from Florence.Utils import insensitive
from vtk_writer import write_vtu
from NormalDistance import NormalDistance
try:
    import meshpy.triangle as triangle
    has_meshpy = True
except ImportError:
    has_meshpy = False
from SalomeMeshReader import ReadMesh
from HigherOrderMeshing import *
from GeometricPath import *
from warnings import warn
from copy import deepcopy



"""
Mesh class providing most of the pre-processing functionalities of the Core module

Roman Poya - 13/06/2015
"""



class Mesh(object):

    """Mesh class provides the following functionalities:  
    1. Generating higher order meshes based on a linear mesh, for tris, tets, quads and hexes
    2. Generating linear tri and tet meshes based on meshpy back-end
    3. Generating linear tri meshes based on distmesh back-end
    4. Finding bounary edges and faces for tris and tets, in case they are not provided by the mesh generator
    5. Reading Salome meshes in binary (.dat/.txt/etc) format
    6. Reading gmsh files .msh 
    7. Checking for node numbering order of elements and fixing it if desired
    8. Writing meshes to unstructured vtk file format (.vtu) based on the original script by Luke Olson 
       (extended here to high order elements) 
    """

    def __init__(self, element_type=None):
        super(Mesh, self).__init__()
        # self.faces and self.edges ARE BOUNDARY FACES 
        # AND BOUNDARY EDGES, RESPECTIVELY
        
        self.degree = None
        self.ndim = None
        self.nelem = None
        self.nnode = None

        self.elements = None
        self.points = None
        self.edges = None
        self.faces = None
        self.element_type = element_type

        self.face_to_element = None
        self.edge_to_element = None
        self.boundary_edge_to_element = None
        self.boundary_face_to_element = None
        self.all_faces = None
        self.all_edges = None
        self.interior_faces = None
        self.interior_edges = None

        # FOR GEOMETRICAL CURVES/SURFACES
        self.edge_to_curve = None
        self.face_to_surface = None


        self.spatial_dimension = None
        self.reader_type = None
        self.reader_type_format = None
        self.reader_type_version = None
        self.writer_type = None

        self.filename = None
        # self.has_meshpy = has_meshpy



    def SetElements(self,arr):
        self.elements = arr

    def SetPoints(self,arr):
        self.points = arr

    def SetEdges(self,arr):
        self.edges = arr

    def SetFaces(self,arr):
        self.faces = arr

    def GetEdges(self):
        assert self.element_type is not None
        if self.element_type == "tri":
            self.GetEdgesTri()
        elif self.element_type == "quad":
            self.GetEdgesQuad()
        elif self.element_type == "tet":
            self.GetEdgesTet()
        elif self.element_type == "hex":
            self.GetEdgesHex()
        else:
            raise ValueError('Type of element not understood')

    def GetBoundaryEdges(self):
        assert self.element_type is not None
        if self.element_type == "tri":
            self.GetBoundaryEdgesTri()
        elif self.element_type == "quad":
            self.GetBoundaryEdgesQuad()
        elif self.element_type == "tet":
            self.GetBoundaryEdgesTet()
        elif self.element_type == "hex":
            self.GetBoundaryEdgesHex()
        else:
            raise ValueError('Type of element not understood')

    def GetInteriorEdges(self):
        assert self.element_type is not None
        if self.element_type == "tri":
            self.GetInteriorEdgesTri()
        elif self.element_type == "quad":
            self.GetInteriorEdgesQuad()
        elif self.element_type == "tet":
            self.GetInteriorEdgesTet()
        elif self.element_type == "hex":
            self.GetInteriorEdgesHex()
        else:
            raise ValueError('Type of element not understood')

    def GetFaces(self):
        assert self.element_type is not None
        if self.element_type == "tet":
            self.GetFacesTet()
        elif self.element_type == "hex":
            self.GetFacesHex()
        elif self.element_type=="tri" or self.element_type=="quad":
            raise ValueError("2D mesh does not have faces")
        else:
            raise ValueError('Type of element not understood')

    def GetBoundaryFaces(self):
        assert self.element_type is not None
        if self.element_type == "tet":
            self.GetBoundaryFacesTet()
        elif self.element_type == "hex":
            self.GetBoundaryFacesHex() 
        elif self.element_type=="tri" or self.element_type=="quad":
            raise ValueError("2D mesh does not have faces")
        else:
            raise ValueError('Type of element not understood')

    def GetInteriorFaces(self):
        assert self.element_type is not None
        if self.element_type == "tet":
            self.GetInteriorFacesTet()
        elif self.element_type == "hex":
            self.GetInteriorFacesHex() 
        elif self.element_type=="tri" or self.element_type=="quad":
            raise ValueError("2D mesh does not have faces")
        else:
            raise ValueError('Type of element not understood')

    def GetElementsEdgeNumbering(self):
        assert self.element_type is not None
        if self.element_type == "tri":
            return self.GetElementsEdgeNumberingTri()
        elif self.element_type == "quad":
            return self.GetElementsEdgeNumberingQuad()
        else:
            raise ValueError('Type of element not understood')

    def GetElementsWithBoundaryEdges(self):
        assert self.element_type is not None
        if self.element_type == "tri":
            return self.GetElementsWithBoundaryEdgesTri()
        elif self.element_type == "quad":
            return self.GetElementsWithBoundaryEdgesQuad()
        else:
            raise ValueError('Type of element not understood')

    def GetElementsFaceNumbering(self):
        assert self.element_type is not None
        if self.element_type == "tet":
            return self.GetElementsFaceNumberingTet()
        elif self.element_type == "hex":
            return self.GetElementsFaceNumberingHex()
        elif self.element_type=="tri" or self.element_type=="quad":
            raise ValueError("2D mesh does not have faces")
        else:
            raise ValueError('Type of element not understood')

    def GetElementsWithBoundaryFaces(self):
        assert self.element_type is not None
        if self.element_type == "tet":
            return self.GetElementsWithBoundaryFacesTet()
        elif self.element_type == "hex":
            return self.GetElementsWithBoundaryFacesHex()
        elif self.element_type=="tri" or self.element_type=="quad":
            raise ValueError("2D mesh does not have faces")
        else:
            raise ValueError('Type of element not understood')


    @property
    def Bounds(self):
        """Returns bounds of a mesh i.e. the minimum and maximum coordinate values
            in every direction
        """
        assert self.points is not None

        if self.points.shape[1] == 3:
            bounds = np.array([[np.min(self.points[:,0]),
                        np.min(self.points[:,1]),
                        np.min(self.points[:,2])],
                        [np.max(self.points[:,0]),
                        np.max(self.points[:,1]),
                        np.max(self.points[:,2])]])
            makezero(bounds)
            return bounds
        elif self.points.shape[1] == 2:
            bounds = np.array([[np.min(self.points[:,0]),
                        np.min(self.points[:,1])],
                        [np.max(self.points[:,0]),
                        np.max(self.points[:,1])]])
            makezero(bounds)
        else:
            raise ValueError("Invalid dimension for mesh coordinates")
            return bounds


    def GetEdgesTri(self):
        """Find the all edges of a triangular mesh.
            Sets all_edges property and returns it

        returns:            

            arr:            numpy ndarray of all edges"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_edges,np.ndarray):
            if self.all_edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_edges.shape[1]==2 and p > 1:
                    pass
                else:
                    return self.all_edges 

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri
        
        node_arranger = NodeArrangementTri(p-1)[0]

        # CHECK IF FACES ARE ALREADY AVAILABLE
        if isinstance(self.all_edges,np.ndarray):
            if self.all_edges.shape[0] > 1 and self.all_edges.shape[1] == p+1:
                warn("Mesh edges seem to be already computed. I am going to recompute them")


        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        edges = np.zeros((3*self.elements.shape[0],p+1),dtype=np.uint64)
        edges[:self.elements.shape[0],:] = self.elements[:,node_arranger[0,:]]
        edges[self.elements.shape[0]:2*self.elements.shape[0],:] = self.elements[:,node_arranger[1,:]]
        edges[2*self.elements.shape[0]:,:] = self.elements[:,node_arranger[2,:]]

        # REMOVE DUPLICATES
        edges, idx = unique2d(edges,consider_sort=True,order=False,return_index=True)

        edge_to_element = np.zeros((edges.shape[0],2),np.int64)
        edge_to_element[:,0] =  idx % self.elements.shape[0]
        edge_to_element[:,1] =  idx // self.elements.shape[0]

        self.edge_to_element = edge_to_element


        # DO NOT SET all_edges IF THE CALLER FUNCTION IS GetBoundaryEdgesTet
        import inspect
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)[1][3]
        
        if calframe != "GetBoundaryEdgesTet":
            self.all_edges = edges

        return edges


    def GetBoundaryEdgesTri(self):
        """Find boundary edges (lines) of triangular mesh"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.edges,np.ndarray):
            if self.edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return


        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTri

        node_arranger = NodeArrangementTri(p-1)[0]

        # CONCATENATE ALL THE EDGES MADE FROM ELEMENTS
        all_edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
                             self.elements[:,node_arranger[2,:]]),axis=0)
        # GET UNIQUE ROWS 
        uniques, idx, inv = unique2d(all_edges,consider_sort=True,order=False,return_index=True,return_inverse=True)

        # ROWS THAT APPEAR ONLY ONCE CORRESPOND TO BOUNDARY EDGES
        freqs_inv = itemfreq(inv)
        edges_ext_flags = freqs_inv[freqs_inv[:,1]==1,0]
        # NOT ARRANGED
        self.edges = uniques[edges_ext_flags,:] 

        # DETERMINE WHICH FACE OF THE ELEMENT THEY ARE
        boundary_edge_to_element = np.zeros((edges_ext_flags.shape[0],2),dtype=np.int64)

        # FURTHER RE-ARRANGEMENT / ARANGE THE NODES BASED ON THE ORDER THEY APPEAR
        # IN ELEMENT CONNECTIVITY
        # THIS STEP IS NOT NECESSARY INDEED - ITS JUST FOR RE-ARANGMENT OF EDGES
        all_edges_in_edges = in2d(all_edges,self.edges,consider_sort=True)
        all_edges_in_edges = np.where(all_edges_in_edges==True)[0]

        boundary_edge_to_element[:,0] = all_edges_in_edges % self.elements.shape[0]
        boundary_edge_to_element[:,1] = all_edges_in_edges // self.elements.shape[0]

        # ARRANGE FOR ANY ORDER OF BASES/ELEMENTS AND ASSIGN DATA MEMBERS
        self.edges = self.elements[boundary_edge_to_element[:,0][:,None],node_arranger[boundary_edge_to_element[:,1],:]]
        self.edges = self.edges.astype(np.uint64)
        self.boundary_edge_to_element = boundary_edge_to_element

        return self.edges


    def GetInteriorEdgesTri(self):
        """Computes interior edges of a triangular mesh

            returns:        

                interior_faces          ndarray of interior edges
                edge_flags              ndarray of edge flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_edges,np.ndarray):
            self.GetEdgesTri()
        if not isinstance(self.edges,np.ndarray):
            self.GetBoundaryEdgesTri()

        sorted_all_edges = np.sort(self.all_edges,axis=1)
        sorted_boundary_edges = np.sort(self.edges,axis=1)

        x = []
        for i in range(self.edges.shape[0]):
            current_sorted_boundary_edge = np.tile(sorted_boundary_edges[i,:],
                self.all_edges.shape[0]).reshape(self.all_edges.shape[0],self.all_edges.shape[1])
            interior_edges = np.linalg.norm(current_sorted_boundary_edge - sorted_all_edges,axis=1)
            pos_interior_edges = np.where(interior_edges==0)[0]
            if pos_interior_edges.shape[0] != 0:
                x.append(pos_interior_edges)

        edge_aranger = np.arange(self.all_edges.shape[0])
        edge_aranger = np.setdiff1d(edge_aranger,np.array(x)[:,0])
        interior_edges = self.all_edges[edge_aranger,:]

        # GET FLAGS FOR BOUNDRAY AND INTERIOR
        edge_flags = np.ones(self.all_edges.shape[0],dtype=np.int64)
        edge_flags[edge_aranger] = 0

        self.interior_edges = interior_edges
        return interior_edges, edge_flags


    def GetFacesTet(self):
        """Find all faces (surfaces) in the tetrahedral mesh (boundary & interior).
            Sets all_faces property and returns it

        returns:            

            arr:            numpy ndarray of all faces

        """

        # DETERMINE DEGREE
        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_faces,np.ndarray):
            if self.all_faces.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_faces.shape[1] == 3 and p > 1:
                    pass
                else:
                    return self.all_faces 

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTet

        node_arranger = NodeArrangementTet(p-1)[0]
        fsize = int((p+1.)*(p+2.)/2.)

        # GET ALL FACES FROM THE ELEMENT CONNECTIVITY
        faces = np.zeros((4*self.elements.shape[0],fsize),dtype=np.uint64)
        faces[:self.elements.shape[0],:] = self.elements[:,node_arranger[0,:]]
        faces[self.elements.shape[0]:2*self.elements.shape[0],:] = self.elements[:,node_arranger[1,:]]
        faces[2*self.elements.shape[0]:3*self.elements.shape[0],:] = self.elements[:,node_arranger[2,:]]
        faces[3*self.elements.shape[0]:,:] = self.elements[:,node_arranger[3,:]]

        # REMOVE DUPLICATES
        self.all_faces, idx = unique2d(faces,consider_sort=True,order=False,return_index=True) 

        face_to_element = np.zeros((self.all_faces.shape[0],2),np.int64)
        face_to_element[:,0] =  idx % self.elements.shape[0]
        face_to_element[:,1] =  idx // self.elements.shape[0]

        self.face_to_element = face_to_element

        return self.all_faces 


    def GetEdgesTet(self):
        """Find all edges (lines) of tetrahedral mesh (boundary & interior)"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_edges,np.ndarray):
            if self.all_edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return self.all_edges 


        # FIRST GET BOUNDARY FACES
        if not isinstance(self.all_faces,np.ndarray):
            self.GetFacesTet()

        # BUILD A 2D MESH
        tmesh = Mesh()
        # tmesh = deepcopy(self) 
        tmesh.element_type = "tri"
        tmesh.elements = self.all_faces
        tmesh.nelem = tmesh.elements.shape[0]
        del tmesh.faces
        del tmesh.points
       
        # COMPUTE ALL EDGES
        self.all_edges = tmesh.GetEdgesTri()
        return self.all_edges



    def GetBoundaryFacesTet(self):
        """Find boundary faces (surfaces) of a tetrahedral mesh"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.faces,np.ndarray):
            if self.faces.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.faces.shape[1] == 3 and p > 1:
                    pass
                else:
                    return


        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTet
        node_arranger = NodeArrangementTet(p-1)[0]

        # CONCATENATE ALL THE FACES MADE FROM ELEMENTS
        all_faces = np.concatenate((self.elements[:,:3],self.elements[:,[0,1,3]],
                             self.elements[:,[0,2,3]],self.elements[:,[1,2,3]]),axis=0)
        # GET UNIQUE ROWS 
        uniques, idx, inv = unique2d(all_faces,consider_sort=True,order=False,return_index=True,return_inverse=True)

        # ROWS THAT APPEAR ONLY ONCE CORRESPOND TO BOUNDARY FACES
        freqs_inv = itemfreq(inv)
        faces_ext_flags = freqs_inv[freqs_inv[:,1]==1,0]
        # NOT ARRANGED
        self.faces = uniques[faces_ext_flags,:] 

        # DETERMINE WHICH FACE OF THE ELEMENT THEY ARE
        boundary_face_to_element = np.zeros((faces_ext_flags.shape[0],2),dtype=np.int64)
        # THE FOLLOWING WILL COMPUTE FACES BASED ON SORTING AND NOT TAKING INTO ACCOUNT 
        # THE ELEMENT CONNECTIVITY
        # boundary_face_to_element[:,0] = np.remainder(idx[faces_ext_flags],self.elements.shape[0])
        # boundary_face_to_element[:,1] = np.floor_divide(idx[faces_ext_flags],self.elements.shape[0])
        # OR EQUIVALENTLY
        # boundary_face_to_element[:,0] = idx[faces_ext_flags] % self.elements.shape[0]
        # boundary_face_to_element[:,1] = idx[faces_ext_flags] // self.elements.shape[0]

        # FURTHER RE-ARRANGEMENT / ARANGE THE NODES BASED ON THE ORDER THEY APPEAR
        # IN ELEMENT CONNECTIVITY
        # THIS STEP IS NOT NECESSARY INDEED - ITS JUST FOR RE-ARANGMENT OF FACES
        all_faces_in_faces = in2d(all_faces,self.faces,consider_sort=True)
        all_faces_in_faces = np.where(all_faces_in_faces==True)[0]

        # boundary_face_to_element = np.zeros((all_faces_in_faces.shape[0],2),dtype=np.int64)
        boundary_face_to_element[:,0] = all_faces_in_faces % self.elements.shape[0]
        boundary_face_to_element[:,1] = all_faces_in_faces // self.elements.shape[0]

        # ARRANGE FOR ANY ORDER OF BASES/ELEMENTS AND ASSIGN DATA MEMBERS
        self.faces = self.elements[boundary_face_to_element[:,0][:,None],node_arranger[boundary_face_to_element[:,1],:]]
        self.faces = self.faces.astype(np.uint64)
        self.boundary_face_to_element = boundary_face_to_element



    def GetBoundaryEdgesTet(self):
        """Find boundary edges (lines) of tetrahedral mesh.
            Note that for tetrahedrals this function is more robust than Salome's default edge generator
        """

        p = self.InferPolynomialDegree()
        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.edges,np.ndarray):
            if self.edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return

        # FIRST GET BOUNDARY FACES
        if not isinstance(self.faces,np.ndarray):
            self.GetBoundaryFacesTet()

        # BUILD A 2D MESH
        tmesh = Mesh()
        tmesh.element_type = "tri"
        tmesh.elements = self.faces
        tmesh.nelem = tmesh.elements.shape[0]
        del tmesh.faces
        del tmesh.points
        
        # ALL THE EDGES CORRESPONDING TO THESE BOUNDARY FACES ARE BOUNDARY EDGES
        self.edges =  tmesh.GetEdgesTri()



    def GetInteriorFacesTet(self):
        """Computes interior faces of a tetrahedral mesh

            returns:        

                interior_faces          ndarray of interior faces
                face_flags              1D array of face flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_faces,np.ndarray):
            self.GetFacesTet()
        if not isinstance(self.faces,np.ndarray):
            self.GetBoundaryFacesTet()

        face_flags = in2d(self.all_faces.astype(self.faces.dtype),self.faces,consider_sort=True)
        face_flags[face_flags==True] = 1
        face_flags[face_flags==False] = 0
        interior_faces = self.all_faces[face_flags==False,:]

        return interior_faces, face_flags


    def GetInteriorEdgesTet(self):
        """Computes interior faces of a tetrahedral mesh

            returns:        

                interior_edges          ndarray of interior edges
                edge_flags              1D array of edge flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_edges,np.ndarray):
            self.GetEdgesTet()
        if not isinstance(self.edges,np.ndarray):
            self.GetBoundaryEdgesTet()

        edge_flags = in2d(self.all_edges.astype(self.edges.dtype),self.edges,consider_sort=True)
        edge_flags[edge_flags==True] = 1
        edge_flags[edge_flags==False] = 0
        interior_edges = self.all_edges[edge_flags==False,:]

        return interior_edges, edge_flags


    def GetEdgesQuad(self):
        """Find the all edges of a quadrilateral mesh.
            Sets all_edges property and returns it

        returns:            

            arr:            numpy ndarray of all edges"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_edges,np.ndarray):
            if self.all_edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_edges.shape[1]==2 and p > 1:
                    pass
                else:
                    return self.all_edges 

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        node_arranger = NodeArrangementQuad(p-1)[0]

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],self.elements[:,node_arranger[3,:]]),axis=0).astype(np.uint64)

        # REMOVE DUPLICATES
        edges, idx = unique2d(edges,consider_sort=True,order=False,return_index=True)

        edge_to_element = np.zeros((edges.shape[0],2),np.int64)
        edge_to_element[:,0] =  idx % self.elements.shape[0]
        edge_to_element[:,1] =  idx // self.elements.shape[0]

        self.edge_to_element = edge_to_element

        # DO NOT SET all_edges IF THE CALLER FUNCTION IS GetBoundaryEdgesHex
        import inspect
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)[1][3]
        
        if calframe != "GetBoundaryEdgesHex":
            self.all_edges = edges

        return edges


    def GetBoundaryEdgesQuad(self):
        """Find boundary edges (lines) of a quadrilateral mesh"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.edges,np.ndarray):
            if self.edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        node_arranger = NodeArrangementQuad(p-1)[0]

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        all_edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],self.elements[:,node_arranger[3,:]]),axis=0).astype(np.uint64)

        # GET UNIQUE ROWS 
        uniques, idx, inv = unique2d(all_edges,consider_sort=True,order=False,return_index=True,return_inverse=True)

        # ROWS THAT APPEAR ONLY ONCE CORRESPOND TO BOUNDARY EDGES
        freqs_inv = itemfreq(inv)
        edges_ext_flags = freqs_inv[freqs_inv[:,1]==1,0]
        # NOT ARRANGED
        self.edges = uniques[edges_ext_flags,:] 

        # DETERMINE WHICH FACE OF THE ELEMENT THEY ARE
        boundary_edge_to_element = np.zeros((edges_ext_flags.shape[0],2),dtype=np.int64)

        # FURTHER RE-ARRANGEMENT / ARANGE THE NODES BASED ON THE ORDER THEY APPEAR
        # IN ELEMENT CONNECTIVITY
        # THIS STEP IS NOT NECESSARY INDEED - ITS JUST FOR RE-ARANGMENT OF EDGES
        all_edges_in_edges = in2d(all_edges,self.edges,consider_sort=True)
        all_edges_in_edges = np.where(all_edges_in_edges==True)[0]

        boundary_edge_to_element[:,0] = all_edges_in_edges % self.elements.shape[0]
        boundary_edge_to_element[:,1] = all_edges_in_edges // self.elements.shape[0]

        # ARRANGE FOR ANY ORDER OF BASES/ELEMENTS AND ASSIGN DATA MEMBERS
        self.edges = self.elements[boundary_edge_to_element[:,0][:,None],node_arranger[boundary_edge_to_element[:,1],:]]
        self.edges = self.edges.astype(np.uint64)
        self.boundary_edge_to_element = boundary_edge_to_element

        return self.edges


    def GetInteriorEdgesQuad(self):
        """Computes interior edges of a quadrilateral mesh

            returns:        

                interior_faces          ndarray of interior edges
                edge_flags              ndarray of edge flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_edges,np.ndarray):
            self.GetEdgesQuad()
        if not isinstance(self.edges,np.ndarray):
            self.GetBoundaryEdgesQuad()

        sorted_all_edges = np.sort(self.all_edges,axis=1)
        sorted_boundary_edges = np.sort(self.edges,axis=1)

        x = []
        for i in range(self.edges.shape[0]):
            current_sorted_boundary_edge = np.tile(sorted_boundary_edges[i,:],
                self.all_edges.shape[0]).reshape(self.all_edges.shape[0],self.all_edges.shape[1])
            interior_edges = np.linalg.norm(current_sorted_boundary_edge - sorted_all_edges,axis=1)
            pos_interior_edges = np.where(interior_edges==0)[0]
            if pos_interior_edges.shape[0] != 0:
                x.append(pos_interior_edges)

        edge_aranger = np.arange(self.all_edges.shape[0])
        edge_aranger = np.setdiff1d(edge_aranger,np.array(x)[:,0])
        interior_edges = self.all_edges[edge_aranger,:]

        # GET FLAGS FOR BOUNDRAY AND INTERIOR
        edge_flags = np.ones(self.all_edges.shape[0],dtype=np.int64)
        edge_flags[edge_aranger] = 0

        self.interior_edges = interior_edges
        return interior_edges, edge_flags


    def GetFacesHex(self):
        """Find all faces (surfaces) in the hexahedral mesh (boundary & interior).
            Sets all_faces property and returns it

        returns:            

            arr:            numpy ndarray of all faces

        """

        # DETERMINE DEGREE
        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_faces,np.ndarray):
            if self.all_faces.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_faces.shape[1] == 4 and p > 1:
                    pass
                else:
                    return self.all_faces 

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex

        node_arranger = NodeArrangementHex(p-1)[0]
        fsize = int((p+1)**3)

        # GET ALL FACES FROM THE ELEMENT CONNECTIVITY
        faces = np.concatenate((np.concatenate((
                np.concatenate((np.concatenate((np.concatenate((self.elements[:,node_arranger[0,:]],
                self.elements[:,node_arranger[1,:]]),axis=0),self.elements[:,node_arranger[2,:]]),axis=0),
                self.elements[:,node_arranger[3,:]]),axis=0),self.elements[:,node_arranger[4,:]]),axis=0),
                self.elements[:,node_arranger[5,:]]),axis=0).astype(np.int64)

        # REMOVE DUPLICATES
        self.all_faces, idx = unique2d(faces,consider_sort=True,order=False,return_index=True) 

        face_to_element = np.zeros((self.all_faces.shape[0],2),np.int64)
        face_to_element[:,0] =  idx % self.elements.shape[0]
        face_to_element[:,1] =  idx // self.elements.shape[0]

        self.face_to_element = face_to_element

        return self.all_faces 


    def GetEdgesHex(self):
        """Find all edges (lines) of tetrahedral mesh (boundary & interior)"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.all_edges,np.ndarray):
            if self.all_edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.all_edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return self.all_edges 


        # FIRST GET BOUNDARY FACES
        if not isinstance(self.all_faces,np.ndarray):
            self.GetFacesHex()

        # BUILD A 2D MESH
        tmesh = Mesh()
        # tmesh = deepcopy(self) 
        tmesh.element_type = "quad"
        tmesh.elements = self.all_faces
        tmesh.nelem = tmesh.elements.shape[0]
        del tmesh.faces
        del tmesh.points
       
        # COMPUTE ALL EDGES
        self.all_edges = tmesh.GetEdgesQuad()
        return self.all_edges


    def GetBoundaryFacesHex(self):
        """Find boundary faces (surfaces) of a hexahedral mesh"""

        p = self.InferPolynomialDegree()

        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.faces,np.ndarray):
            if self.faces.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.faces.shape[1] == 4 and p > 1:
                    pass
                else:
                    return


        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex
        node_arranger = NodeArrangementHex(p-1)[0]

        # CONCATENATE ALL THE FACES MADE FROM ELEMENTS
        all_faces = np.concatenate((np.concatenate((
                np.concatenate((np.concatenate((np.concatenate((self.elements[:,node_arranger[0,:]],
                self.elements[:,node_arranger[1,:]]),axis=0),self.elements[:,node_arranger[2,:]]),axis=0),
                self.elements[:,node_arranger[3,:]]),axis=0),self.elements[:,node_arranger[4,:]]),axis=0),
                self.elements[:,node_arranger[5,:]]),axis=0).astype(np.int64)
        # GET UNIQUE ROWS 
        uniques, idx, inv = unique2d(all_faces,consider_sort=True,order=False,return_index=True,return_inverse=True)

        # ROWS THAT APPEAR ONLY ONCE CORRESPOND TO BOUNDARY FACES
        freqs_inv = itemfreq(inv)
        faces_ext_flags = freqs_inv[freqs_inv[:,1]==1,0]
        # NOT ARRANGED
        self.faces = uniques[faces_ext_flags,:] 

        # DETERMINE WHICH FACE OF THE ELEMENT THEY ARE
        boundary_face_to_element = np.zeros((faces_ext_flags.shape[0],2),dtype=np.int64)

        # FURTHER RE-ARRANGEMENT / ARANGE THE NODES BASED ON THE ORDER THEY APPEAR
        # IN ELEMENT CONNECTIVITY
        # THIS STEP IS NOT NECESSARY INDEED - ITS JUST FOR RE-ARANGMENT OF FACES
        all_faces_in_faces = in2d(all_faces,self.faces,consider_sort=True)
        all_faces_in_faces = np.where(all_faces_in_faces==True)[0]

        # boundary_face_to_element = np.zeros((all_faces_in_faces.shape[0],2),dtype=np.int64)
        boundary_face_to_element[:,0] = all_faces_in_faces % self.elements.shape[0]
        boundary_face_to_element[:,1] = all_faces_in_faces // self.elements.shape[0]

        # ARRANGE FOR ANY ORDER OF BASES/ELEMENTS AND ASSIGN DATA MEMBERS
        self.faces = self.elements[boundary_face_to_element[:,0][:,None],node_arranger[boundary_face_to_element[:,1],:]]
        self.faces = self.faces.astype(np.uint64)
        self.boundary_face_to_element = boundary_face_to_element


    def GetBoundaryEdgesHex(self):
        """Find boundary edges (lines) of hexahedral mesh.
        """

        p = self.InferPolynomialDegree()
        # DO NOT COMPUTE IF ALREADY COMPUTED
        if isinstance(self.edges,np.ndarray):
            if self.edges.shape[0] > 1:
                # IF LINEAR VERSION IS COMPUTED, DO COMPUTE HIGHER VERSION
                if self.edges.shape[1] == 2 and p > 1:
                    pass
                else:
                    return


        # FIRST GET BOUNDARY FACES
        if not isinstance(self.faces,np.ndarray):
            self.GetBoundaryFacesHex()

        # BUILD A 2D MESH
        tmesh = Mesh()
        tmesh.element_type = "quad"
        tmesh.elements = self.faces
        tmesh.nelem = tmesh.elements.shape[0]
        del tmesh.faces
        del tmesh.points
        
        # ALL THE EDGES CORRESPONDING TO THESE BOUNDARY FACES ARE BOUNDARY EDGES
        self.edges =  tmesh.GetEdgesQuad()


    def GetInteriorFacesHex(self):
        """Computes interior faces of a hexahedral mesh

            returns:        

                interior_faces          ndarray of interior faces
                face_flags              1D array of face flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_faces,np.ndarray):
            self.GetFacesHex()
        if not isinstance(self.faces,np.ndarray):
            self.GetBoundaryFacesHex()

        face_flags = in2d(self.all_faces.astype(self.faces.dtype),self.faces,consider_sort=True)
        face_flags[face_flags==True] = 1
        face_flags[face_flags==False] = 0
        interior_faces = self.all_faces[face_flags==False,:]

        return interior_faces, face_flags


    def GetInteriorEdgesHex(self):
        """Computes interior faces of a hexahedral mesh

            returns:        

                interior_edges          ndarray of interior edges
                edge_flags              1D array of edge flags: 0 for interior and 1 for boundary

        """

        if not isinstance(self.all_edges,np.ndarray):
            self.GetEdgesHex()
        if not isinstance(self.edges,np.ndarray):
            self.GetBoundaryEdgesHex()

        edge_flags = in2d(self.all_edges.astype(self.edges.dtype),self.edges,consider_sort=True)
        edge_flags[edge_flags==True] = 1
        edge_flags[edge_flags==False] = 0
        interior_edges = self.all_edges[edge_flags==False,:]

        return interior_edges, edge_flags



    def GetHighOrderMesh(self,p=1,**kwargs):
        """Given a linear tri, tet, quad or hex mesh compute high order mesh based on it.
        This is a static method linked to the HigherOrderMeshing module"""

        if self.degree is None:
            self.InferPolynomialDegree()

        C = p-1
        if 'C' in kwargs.keys():
            if kwargs['C'] != p - 1:
                raise ValueError("Did not understand the specified interpolation degree of the mesh")
            del kwargs['C']

        # DO NOT COMPUTE IF ALREADY COMPUTED FOR THE SAME ORDER
        if self.degree == p:
            return

        # SITUATIONS WHEN ANOTHER HIGH ORDER MESH IS REQUIRED, WITH ONE HIGH
        # ORDER MESH ALREADY AVAILABLE
        if self.degree != 1 and self.degree - 1 != C:
            if self.element_type == "tri":
                self.elements = self.elements[:,:3]
                nmax = self.elements.max() + 1
                self.points = self.points[:nmax,:]
                if self.edges is not None:
                    self.edges = self.edges[:,:2]
                if self.all_edges is not None:
                    self.all_edges = self.all_edges[:,:2]
            elif self.element_type == "tet":
                self.elements = self.elements[:,:4]
                nmax = self.elements.max() + 1
                self.points = self.points[:nmax,:]
                if self.edges is not None:
                    self.edges = self.edges[:,:2]
                if self.faces is not None:
                    self.faces = self.faces[:,:3]
                if self.all_edges is not None:
                    self.all_edges = self.all_edges[:,:2]
                if self.all_faces is not None:
                    self.all_faces = self.all_faces[:,:3]
            elif self.element_type == "quad":
                self.elements = self.elements[:,:4]
                nmax = self.elements.max() + 1
                self.points = self.points[:nmax,:]
                if self.edges is not None:
                    self.edges = self.edges[:,:2]
                if self.all_edges is not None:
                    self.all_edges = self.all_edges[:,:2]
            elif self.element_type == "hex":
                self.elements = self.elements[:,:8]
                nmax = self.elements.max() + 1
                self.points = self.points[:nmax,:]
                if self.edges is not None:
                    self.edges = self.edges[:,:2]
                if self.faces is not None:
                    self.faces = self.faces[:,:4]
                if self.all_edges is not None:
                    self.all_edges = self.all_edges[:,:2]
                if self.all_faces is not None:
                    self.all_faces = self.all_faces[:,:4]
            else:
                raise NotImplementedError("Creating high order mesh based on another high order mesh for",
                    self.element_type, "elements is not implemented yet")



        print 'Generating p = '+str(C+1)+' mesh based on the linear mesh...'
        t_mesh = time()
        # BUILD A NEW MESH BASED ON THE LINEAR MESH
        if self.element_type == 'tri':
            # BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TRIANGLES  
            # nmesh = HighOrderMeshTri(C,self,**kwargs)
            # nmesh = HighOrderMeshTri_UNSTABLE(C,self,**kwargs)
            nmesh = HighOrderMeshTri_SEMISTABLE(C,self,**kwargs)

        elif self.element_type == 'tet':
            # BUILD A NEW MESH USING THE FEKETE NODAL POINTS FOR TETRAHEDRALS
            # nmesh = HighOrderMeshTet(C,self,**kwargs)
            # nmesh = HighOrderMeshTet_UNSTABLE(C,self,**kwargs)
            nmesh = HighOrderMeshTet_SEMISTABLE(C,self,**kwargs)

        elif self.element_type == 'quad':
            # BUILD A NEW MESH USING THE GAUSS-LOBATTO NODAL POINTS FOR QUADS
            nmesh = HighOrderMeshQuad(C,self,**kwargs)

        elif self.element_type == 'hex':
            # BUILD A NEW MESH USING THE GAUSS-LOBATTO NODAL POINTS FOR HEXES
            nmesh = HighOrderMeshHex(C,self,**kwargs)

        self.points = nmesh.points
        self.elements = nmesh.elements.astype(np.uint64)
        self.edges = nmesh.edges.astype(np.uint64)
        if isinstance(self.faces,np.ndarray):
            self.faces = nmesh.faces.astype(np.uint64)
        self.nelem = nmesh.nelem
        self.element_type = nmesh.info
        self.degree = C+1

        self.ChangeType()
        
        print 'Finished generating the high order mesh. Time taken', time()-t_mesh,'sec'


    def EdgeLengths(self,which_edges='boundary'):
        """Computes length of edges, for 2D and 3D meshes

        which_edges:            [str] 'boundary' for boundary edges only 
                                and 'all' for all edges
        """

        assert self.points is not None
        assert self.element_type is not None


        lengths = None
        if which_edges == 'boundary':
            if self.edges is None:
                self.GetBoundaryEdges()

            edge_coords = self.points[self.edges[:,:2],:]
            lengths = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)

        elif which_edges == 'all':
            if self.all_edges is None:
                self.GetEdges()

            edge_coords = self.points[self.all_edges[:,:2],:]
            lengths = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)

        return lengths  


    def Areas(self, with_sign=False, gpoints=None):
        """Find areas of all 2D elements [tris, quads]. 
            For 3D elements returns surface areas of faces 

            input:
                with_sign:              [str] compute with/without sign
                gpoints:                [ndarray] given coordinates to use instead of
                                        self.points

            returns:                    1D array of nelem x 1 containing areas

        """

        assert self.elements is not None
        assert self.element_type is not None
        if gpoints is None:
            assert self.points is not None
            gpoints = self.points

        if self.element_type == "tri":
            points = np.ones((gpoints.shape[0],3),dtype=np.float64)
            points[:,:2] = gpoints
            # FIND AREAS OF ALL THE ELEMENTS
            area = 0.5*np.linalg.det(points[self.elements[:,:3],:])

        elif self.element_type == "quad":
            # NODE ORDERING IS IRRELEVANT, AS IT IS THESE AREAS
            # WHICH DETERMINE NODE ORDERING 
            # AREA OF QUAD ABCD = AREA OF ABC + AREA OF ACD 
            points = np.ones((gpoints.shape[0],3),dtype=np.float64)
            points[:,:2] = gpoints
            # FIND AREAS ABC
            area0 = np.linalg.det(points[self.elements[:,:3],:])
            # FIND AREAS ACD
            area1 = np.linalg.det(points[self.elements[:,[1,2,3]],:])
            # FIND AREAS OF ALL THE ELEMENTS
            area = 0.5*(area0+area1)

        elif self.element_type == "tet":
            # GET ALL THE FACES
            faces = self.GetFacesTet()

            points = np.ones((gpoints.shape[0],3),dtype=np.float64)
            points[:,:2]=gpoints[:,:2]
            area0 = np.linalg.det(points[faces[:,:3],:])

            points[:,:2]=gpoints[:,[2,0]]
            area1 = np.linalg.det(points[faces[:,:3],:])

            points[:,:2]=gpoints[:,[1,2]]
            area2 = np.linalg.det(points[faces[:,:3],:])

            area = 0.5*np.linalg.norm(area0+area1+area2)

        elif self.element_type == "hex":

            from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex
            from Florence.Tensor import unique2d
            C = self.InferPolynomialDegree() - 1

            area = 0
            node_arranger = NodeArrangementHex(C)[0]
            for i in range(node_arranger.shape[0]):
                # print node_arranger[i,:]
                # AREA OF FACES
                points = np.ones((gpoints.shape[0],3),dtype=np.float64)
                if i==0 or i==1:
                    points[:,:2] = gpoints[:,:2]
                elif i==2 or i==3:
                    points[:,:2] = gpoints[:,[0,2]]
                elif i==4 or i==5:
                    points[:,:2] = gpoints[:,1:]
                # FIND AREAS ABC
                area0 = np.linalg.det(points[self.elements[:,node_arranger[i,:3]],:])
                # FIND AREAS ACD
                area1 = np.linalg.det(points[self.elements[:,node_arranger[i,1:]],:])
                # FIND AREAS OF ALL THE ELEMENTS
                area += 0.5*np.linalg.norm(area0+area1)

            # print area
            raise ValueError('Hex areas implementation requires further checks')

        else:
            raise NotImplementedError("Computing areas for", self.element_type, "elements not implemented yet")

        if with_sign is False:
            if self.element_type == "tri" or self.element_type == "quad":
                area = np.abs(area)
            elif self.element_type == "tet":
                raise NotImplementedError('Numbering order of tetrahedral faces could not be determined')

        return area


    def Volumes(self, with_sign=False, gpoints=None):
        """Find Volumes of all 3D elements [tets, hexes]

            input:
                with_sign:              [str] compute with/without sign
                gpoints:                [ndarray] given coordinates to use instead of
                                        self.points

            returns:                    1D array of nelem x 1 containing volumes

        """

        assert self.elements is not None
        assert self.element_type is not None

        if self.points.shape[1] == 2:
            raise ValueError("2D mesh does not volume")
        if gpoints is None:
            assert self.points is not None
            gpoints = self.points

        if self.element_type == "tet":

            a = gpoints[self.elements[:,0],:]
            b = gpoints[self.elements[:,1],:]
            c = gpoints[self.elements[:,2],:]
            d = gpoints[self.elements[:,3],:]

            det_array = np.dstack((a-d,b-d,c-d))
            # FIND VOLUME OF ALL THE ELEMENTS
            volume = 1./6.*np.linalg.det(det_array)

        elif self.element_type == "hex":

            # Refer: https://en.wikipedia.org/wiki/Parallelepiped
            
            a = gpoints[self.elements[:,0],:]
            b = gpoints[self.elements[:,1],:]
            c = gpoints[self.elements[:,3],:]
            d = gpoints[self.elements[:,4],:]

            det_array = np.dstack((b-a,c-a,d-a))
            # FIND VOLUME OF ALL THE ELEMENTS
            volume = np.linalg.det(det_array)

        else:
            raise NotImplementedError("Computing volumes for", self.element_type, "elements not implemented yet")

        if with_sign is False:
            volume = np.abs(volume)

        return volume


    def AspectRatios(self,algorithm='edge_based'):
        """Compute aspect ratio of the mesh element-by-element.
            For 2D meshes aspect ratio is aspect ratio is defined as 
            the ratio of maximum edge length to minimum edge length.
            For 3D meshes aspect ratio can be either length or area based. 

            input:
                algorithm:                  [str] 'edge_based' or 'face_based'
            returns:
                aspect_ratio:               [1D array] of size (self.nelem) containing aspect ratio of elements
        """ 

        assert self.points is not None
        assert self.element_type is not None

        aspect_ratio = None
        if algorithm == 'edge_based':
            if self.element_type == "tri":
                edge_coords = self.points[self.elements[:,:3],:]
                AB = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)
                AC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,0,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,1,:],axis=1)

                minimum = np.minimum(np.minimum(AB,AC),BC)
                maximum = np.maximum(np.maximum(AB,AC),BC)

                aspect_ratio = 1.0*maximum/minimum

            elif self.element_type == "quad":
                edge_coords = self.points[self.elements[:,:4],:]
                AB = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,1,:],axis=1)
                CD = np.linalg.norm(edge_coords[:,3,:] - edge_coords[:,2,:],axis=1)
                DA = np.linalg.norm(edge_coords[:,0,:] - edge_coords[:,3,:],axis=1)
  
                minimum = np.minimum(np.minimum(np.minimum(AB,BC),CD),DA)
                maximum = np.maximum(np.maximum(np.maximum(AB,BC),CD),DA)

                aspect_ratio = 1.0*maximum/minimum

            elif self.element_type == "tet":
                edge_coords = self.points[self.elements[:,:4],:]
                AB = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)
                AC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,0,:],axis=1)
                AD = np.linalg.norm(edge_coords[:,3,:] - edge_coords[:,0,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,1,:],axis=1)
                BD = np.linalg.norm(edge_coords[:,3,:] - edge_coords[:,1,:],axis=1)
                CD = np.linalg.norm(edge_coords[:,3,:] - edge_coords[:,2,:],axis=1)

                minimum = np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(AB,AC),AD),BC),BD),CD)
                maximum = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(AB,AC),AD),BC),BD),CD)

                aspect_ratio = 1.0*maximum/minimum

            elif self.element_type == "hex":
                edge_coords = self.points[self.elements[:,:8],:]
                AB = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,2,:] - edge_coords[:,1,:],axis=1)
                CD = np.linalg.norm(edge_coords[:,3,:] - edge_coords[:,2,:],axis=1)
                DA = np.linalg.norm(edge_coords[:,0,:] - edge_coords[:,3,:],axis=1)

                minimum0 = np.minimum(np.minimum(np.minimum(AB,BC),CD),DA)
                maximum0 = np.maximum(np.maximum(np.maximum(AB,BC),CD),DA)

                AB = np.linalg.norm(edge_coords[:,5,:] - edge_coords[:,4,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,6,:] - edge_coords[:,5,:],axis=1)
                CD = np.linalg.norm(edge_coords[:,7,:] - edge_coords[:,6,:],axis=1)
                DA = np.linalg.norm(edge_coords[:,4,:] - edge_coords[:,7,:],axis=1)

                minimum1 = np.minimum(np.minimum(np.minimum(AB,BC),CD),DA)
                maximum1 = np.maximum(np.maximum(np.maximum(AB,BC),CD),DA)

                AB = np.linalg.norm(edge_coords[:,4,:] - edge_coords[:,0,:],axis=1)
                BC = np.linalg.norm(edge_coords[:,5,:] - edge_coords[:,1,:],axis=1)
                CD = np.linalg.norm(edge_coords[:,6,:] - edge_coords[:,2,:],axis=1)
                DA = np.linalg.norm(edge_coords[:,7,:] - edge_coords[:,3,:],axis=1)

                minimum2 = np.minimum(np.minimum(np.minimum(AB,BC),CD),DA)
                maximum2 = np.maximum(np.maximum(np.maximum(AB,BC),CD),DA)

                minimum = np.minimum(minimum0,np.minimum(minimum1,minimum2))
                maximum = np.maximum(maximum0,np.maximum(maximum1,maximum2))

                aspect_ratio = 1.0*maximum/minimum

        elif algorithm == 'face_based':
            raise NotImplementedError("Face/area based aspect ratio is not implemented yet")

        return aspect_ratio


    def Median(self, geometric=True):
        """Computes median of the elements tri, tet, quad, hex based on the interpolation function

            input:
                geometric           [Bool] geometrically computes median with relying on FEM bases
            retruns:
                median:             [ndarray] of median of elements
                bases_at_median:    [1D array] of (p=1) bases at median            
        """

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        median = None


        if geometric == True:
            median = np.sum(self.points[self.elements,:],axis=1)/self.elements.shape[1]
            return median

        if self.element_type == "tri":
            from Florence.FunctionSpace import Tri
            from Florence.QuadratureRules import FeketePointsTri

            eps = FeketePointsTri(2)
            middle_point_isoparametric = eps[6,:] 
            if not np.isclose(sum(middle_point_isoparametric),-0.6666666):
                raise ValueError("Median of triangle does not match [-0.3333,-0.3333]. "
                    "Did you change your nodal spacing or interpolation functions?")

            hpBases = Tri.hpNodal.hpBases
            bases_for_middle_point = hpBases(0,middle_point_isoparametric[0],
                middle_point_isoparametric[1])[0]
            median = np.einsum('ijk,j',self.points[self.elements[:,:3],:],bases_for_middle_point) 

        elif self.element_type == "tet":
            from Florence.FunctionSpace import Tet
            from Florence.QuadratureRules import FeketePointsTet

            middle_point_isoparametric = FeketePointsTet(3)[21]
            if not np.isclose(sum(middle_point_isoparametric),-1.5):
                raise ValueError("Median of tetrahedral does not match [-0.5,-0.5,-0.5]. "
                    "Did you change your nodal spacing or interpolation functions?")

            # C = self.InferPolynomialDegree() - 1
            hpBases = Tet.hpNodal.hpBases
            bases_for_middle_point = hpBases(0,middle_point_isoparametric[0],
                middle_point_isoparametric[1],middle_point_isoparametric[2])[0]

            median = np.einsum('ijk,j',self.points[self.elements[:,:4],:],bases_for_middle_point) 

        else:
            raise NotImplementedError('Median for {} elements not implemented yet'.format(self.element_type))

        return median, bases_for_middle_point


    def LargestSegment(self, smallest_element=True, nsamples=50, 
        plot_segment=False, plot_element=False, figure=None, save=False, filename=None):
        """Finds the largest segment that can fit in an element. For curvilinear elements
            this measure can be used as (h) for h-refinement studies

            input:
                smallest_element                [bool] if the largest segment size is to be computed in the
                                                smallest element (i.e. element with the smallest area in 2D or
                                                smallest volume in 3D). Default is True. If False, then the largest
                                                segment in the largest element will be computed. 
                nsample:                        [int] number of sample points along the curved
                                                edges of the elements. The maximum distance between
                                                all combinations of these points is the largest
                                                segment
                plot_segment:                   [bool] plots segment on tope of [curved/straight] mesh
                plot_element:                   [bool] plots the straight/curved element to which the segment
                                                belongs
                figure:                         [an instance of matplotlib/mayavi.mlab figure for 2D/3D]
                save:                           [bool] wether to save the figure or not
                filename:                       [str] file name for the figure to be save
            
            returns:
                largest_segment_length          [float] maximum segment length that could be fit within either the 
        """

        self.__do_memebers_exist__()
        if self.element_type == "hex" or self.element_type == "tet":
            quantity = self.Volumes()
        elif self.element_type == "quad" or self.element_type == "tri":
            quantity = self.Areas()


        if smallest_element:
            omesh = self.GetLocalisedMesh(quantity.argmin())
        else:
            omesh = self.GetLocalisedMesh(quantity.argmax())

        try:
            from Florence.PostProcessing import PostProcess
        except:
            raise ImportError('This function requires florence PostProcessing module')
            return

        if save:
            if filename is None:
                raise ValueError("No file name provided. I am going to write one the current directory")
                filename = PWD(__file__) + "/output.png"

        if self.element_type == "tri":
            tmesh = PostProcess.TessellateTris(omesh,np.zeros_like(omesh.points),
                plot_edges=True, interpolation_degree=nsamples)
        elif self.element_type == "quad":
            tmesh = PostProcess.TessellateQuads(omesh,np.zeros_like(omesh.points),
                plot_edges=True, interpolation_degree=nsamples)
        elif self.element_type == "tet":
            tmesh = PostProcess.TessellateTets(omesh,np.zeros_like(omesh.points),
                plot_edges=True, interpolation_degree=nsamples)
        elif self.element_type == "hex":
            tmesh = PostProcess.TessellateHexes(omesh,np.zeros_like(omesh.points),
                plot_edges=True, interpolation_degree=nsamples)  

        ndim = omesh.InferSpatialDimension()
        nnode = tmesh.points.shape[0]
        largest_segment_lengths = []
        nodes = np.array((1,ndim))
        for i in range(nnode):
            tiled_points = np.tile(tmesh.points[i,:][:,None],nnode).T
            segment_lengths = np.linalg.norm(tmesh.points - tiled_points, axis=1)
            largest_segment_lengths.append(segment_lengths.max())
            nodes = np.vstack((nodes, np.array([i,segment_lengths.argmax()])[None,:]))

        largest_segment_lengths = np.array(largest_segment_lengths)
        nodes = nodes[1:,:]
        largest_segment_length = largest_segment_lengths.max()
        corresponding_nodes = nodes[largest_segment_lengths.argmax(),:]


        if plot_segment:
            
            segment_coords = tmesh.points[corresponding_nodes,:]

            if ndim==2:
                import matplotlib.pyplot as plt
                if figure == None:
                    figure = plt.figure()

                if plot_element:
                    if omesh.element_type == "tri":
                        PostProcess.CurvilinearPlotTri(omesh,
                            np.zeros_like(omesh.points),plot_points=True, 
                            figure=figure, interpolation_degree=nsamples, show_plot=False)
                    elif omesh.element_type == "quad":
                        PostProcess.CurvilinearPlotQuad(omesh,
                            np.zeros_like(omesh.points),plot_points=True, 
                            figure=figure, interpolation_degree=nsamples, show_plot=False)

                tmesh.SimplePlot(figure=figure,show_plot=False)
                # plt.plot(tmesh.x_edges,tmesh.y_edges,'-o',color="#FF6347")
                # plt.plot(segment_coords[:,0],segment_coords[:,1],'-o',color="#E34234", linewidth=3)

                # fl = "/home/roman/Dropbox/2016_Linearised_Electromechanics_Paper/figures/hp_Benchmark/"
                # plt.savefig(fl+"ElementSizeTri.eps", bbox_inches="tight",dpi=300)
                # plt.savefig(fl+"ElementSizeTri_Segment.eps", bbox_inches="tight",dpi=300)
                # plt.savefig(fl+"ElementSizeTri_Tessellation.eps", bbox_inches="tight",pad_inches=0,dpi=300)
                # plt.savefig(fl+"ElementSizeQuad.eps", bbox_inches="tight",dpi=300)
                # plt.savefig(fl+"ElementSizeQuad_Tessellation.eps", bbox_inches="tight",pad_inches=0,dpi=300)
                # plt.savefig(fl+"ElementSizeQuad_Segment.eps", bbox_inches="tight",dpi=300)

                if save:
                    plt.savefig(filename,bbox_inches="tight",dpi=300)

                # plt.show()

            elif ndim==3:

                import os
                os.environ['ETS_TOOLKIT'] = 'qt4'
                from mayavi import mlab
    
                if figure is None:
                    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

                if plot_element:
                    if omesh.element_type == "tet":
                        PostProcess.CurvilinearPlotTet(omesh,
                            np.zeros_like(omesh.points),plot_points=True, point_radius=0.13,
                            figure=figure, interpolation_degree=nsamples, show_plot=False)
                    elif omesh.element_type == "hex":
                        PostProcess.CurvilinearPlotHex(omesh,
                            np.zeros_like(omesh.points),plot_points=True, 
                            figure=figure, interpolation_degree=nsamples, show_plot=False)

                tmesh.GetEdges()
                edge_coords = tmesh.points[np.unique(tmesh.all_edges),:]
                mlab.triangular_mesh(tmesh.points[:,0],tmesh.points[:,1],tmesh.points[:,2],tmesh.elements, representation='wireframe', color=(0,0,0))
                # # mlab.points3d(edge_coords[:,0],edge_coords[:,1],edge_coords[:,2],color=(1., 99/255., 71./255), scale_factor=0.03)
                # # mlab.plot3d(segment_coords[:,0],segment_coords[:,1],segment_coords[:,2], color=(227./255, 66./255, 52./255)) 
                mlab.points3d(edge_coords[:,0],edge_coords[:,1],edge_coords[:,2],color=(1., 99/255., 71./255), scale_factor=0.17)
                mlab.plot3d(segment_coords[:,0],segment_coords[:,1],segment_coords[:,2], 
                    color=(227./255, 66./255, 52./255), line_width=10., representation="wireframe")
                
                if save:
                    mlab.savefig(filename,dpi=300)

                mlab.show()



        return largest_segment_length  
  


    def CheckNodeNumbering(self,change_order_to='retain', verbose=True):
        """Checks for node numbering order of the imported mesh. Mesh can be tri or tet

        input:

            change_order_to:            [str] {'clockwise','anti-clockwise','retain'} changes the order to clockwise, 
                                        anti-clockwise or retains the numbering order - default is 'retain' 

        output: 

            original_order:             [str] {'clockwise','anti-clockwise','retain'} returns the original numbering order"""


        assert self.elements is not None
        assert self.points is not None

        # CHECK IF IT IS LINEAR MESH
        # HIGH ORDER CURVED ELEMENTS HAVE AREAS WHICH CAN BE COMPUTED THROUGH BASES FUNCTIONS
        quantity = np.array([])
        if self.element_type == "tri":
            assert self.elements.shape[1]==3
            quantity = self.Areas(with_sign=True)
        elif self.element_type == "quad":
            assert self.elements.shape[1]==4
            quantity = self.Areas(with_sign=True)
        elif self.element_type == "tet":
            assert self.elements.shape[1]==4
            quantity = self.Volumes(with_sign=True)
        elif self.element_type == "hex":
            assert self.elements.shape[1]==8
            quantity = self.Volumes(with_sign=True)

        original_order = ''

        # CHECK NUMBERING
        if (quantity > 0).all():
            original_order = 'anti-clockwise'
            if change_order_to == 'clockwise':
                self.elements = np.fliplr(self.elements)
        elif (quantity < 0).all():
            original_order = 'clockwise'
            if change_order_to == 'anti-clockwise':
                self.elements = np.fliplr(self.elements)
        else:
            original_order = 'mixed'
            if change_order_to == 'clockwise':
                self.elements[quantity>0,:] = np.fliplr(self.elements[quantity>0,:])
            elif change_order_to == 'anti-clockwise':
                self.elements[quantity<0,:] = np.fliplr(self.elements[quantity<0,:])


        if original_order == 'anti-clockwise':
            print u'\u2713'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering'
        else:
            print u'\u2717'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering'

        return original_order




    def GetElementsEdgeNumberingTri(self):
        """Finds edges of elements and their flags saying which edge they are [0,1,2].
            At most a triangle can have all its three edges on the boundary.

        output: 

            edge_elements:              [1D array] array containing elements which have edges
                                        on the boundary

                                        Note that this method sets the self.edge_to_element to edge_elements,
                                        so the return value is not strictly necessary
        """

        if isinstance(self.edge_to_element,np.ndarray):
            if self.edge_to_element.shape[0] > 1:
                return self.edge_to_element

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY 
        if self.all_edges is None:
            self.GetEdgesTri()


        all_edges = np.concatenate((self.elements[:,:2],self.elements[:,[1,2]],
            self.elements[:,[2,0]]),axis=0).astype(np.int64)

        all_edges, idx = unique2d(all_edges,consider_sort=True,order=False, return_index=True)
        edge_elements = np.zeros((all_edges.shape[0],2),dtype=np.int64)

        edge_elements[:,0] = idx % self.elements.shape[0]
        edge_elements[:,1] = idx // self.elements.shape[0]

        self.edge_to_element = edge_elements
        return self.edge_to_element


    def GetElementsWithBoundaryEdgesTri(self):
        """Finds elements which have edges on the boundary.
            At most a triangle can have all its three edges on the boundary.

        output: 

            edge_elements:              [2D array] array containing elements which have edge
                                        on the boundary [cloumn 0] and a flag stating which edges they are [column 1]

        """

        if isinstance(self.boundary_edge_to_element,np.ndarray):
            if self.boundary_edge_to_element.shape[1] > 1 and self.boundary_edge_to_element.shape[0] > 1:
                return self.boundary_edge_to_element

        assert self.edges is not None or self.elements is not None

        edge_elements = np.zeros((self.edges.shape[0],2),dtype=np.int64)
        
        # FIND WHICH FACE NODES ARE IN WHICH ELEMENT
        for i in range(self.edges.shape[0]):
            x = []
            for j in range(2):
                x.append(np.where(self.elements[:,:3]==self.edges[i,j])[0])

            # FIND WHICH ELEMENTS CONTAIN ALL FACE NODES - FOR INTERIOR ELEMENTS
            # THEIR CAN BE MORE THAN ONE ELEMENT CONTAINING ALL FACE NODES
            z = x[0]
            for k in range(1,len(x)):
                z = np.intersect1d(x[k],z)

            # CHOOSE ONLY ONE OF THESE ELEMENTS
            edge_elements[i,0] = z[0]
            # WHICH COLUMNS IN THAT ELEMENT ARE THE FACE NODES LOCATED 
            cols = np.array([np.where(self.elements[z[0],:]==self.edges[i,0])[0], 
                            np.where(self.elements[z[0],:]==self.edges[i,1])[0]
                            ])

            cols = np.sort(cols.flatten())

            if cols[0] == 0 and cols[1] == 1:
                edge_elements[i,1] = 0
            elif cols[0] == 1 and cols[1] == 2:
                edge_elements[i,1] = 1
            elif cols[0] == 0 and cols[1] == 2:
                edge_elements[i,1] = 2

        self.boundary_edge_to_element = edge_elements
        return edge_elements


    def GetElementsWithBoundaryFacesTet(self):
        """Finds elements which have faces on the boundary.
            At most a tetrahedral can have all its four faces on the boundary.

        output: 

            boundary_face_to_element:   [2D array] array containing elements which have face
                                        on the boundary [cloumn 0] and a flag stating which faces they are [column 1]

        """

        assert self.faces is not None or self.elements is not None

        # THIS METHOD ALWAYS RETURNS THE FACE TO ELEMENT ARRAY, AND DOES NOT CHECK
        # IF THIS HAS BEEN COMPUTED BEFORE, THE REASON BEING THAT THE FACES CAN COME 
        # EXTERNALLY WHOSE ARRANGEMENT WOULD NOT CORRESPOND TO THE ONE USED INTERNALLY
        # HENCE THIS MAPPING BECOMES NECESSARY
        
        all_faces = np.concatenate((self.elements[:,:3],self.elements[:,[0,1,3]],
            self.elements[:,[0,2,3]],self.elements[:,[1,2,3]]),axis=0).astype(self.faces.dtype)

        all_faces_in_faces = in2d(all_faces,self.faces[:,:3],consider_sort=True)
        all_faces_in_faces = np.where(all_faces_in_faces==True)[0]

        boundary_face_to_element = np.zeros((all_faces_in_faces.shape[0],2),dtype=np.int64)
        boundary_face_to_element[:,0] = all_faces_in_faces % self.elements.shape[0]
        boundary_face_to_element[:,1] = all_faces_in_faces // self.elements.shape[0]

        # SO FAR WE HAVE COMPUTED THE ELEMENTS THAT CONTAIN FACES, HOWEVER 
        # NOTE THAT WE STILL HAVE NOT COMPUTED A MAPPING BETWEEN ELEMENTS AND 
        # FACES. WE ONLY KNOW WHICH ELEMENTS CONTAIN FACES FROM in2d.
        # WE NEED TO FIND THIS MAPPING NOW

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTet

        C = self.InferPolynomialDegree() - 1
        node_arranger = NodeArrangementTet(C)[0]

        # WE NEED TO DO THIS DUMMY RECONSTRUCTION OF FACES BASED ON ELEMENTS
        faces = self.elements[boundary_face_to_element[:,0][:,None],
            node_arranger[boundary_face_to_element[:,1],:]].astype(self.faces.dtype)

        # CHECK FOR THIS CONDITION AS ARRANGEMENT IS NO LONGER MAINTAINED
        assert np.sum(faces[:,:3].astype(np.int64) - self.faces[:,:3].astype(np.int64)) == 0

        # NOW GET THE ROW MAPPING BETWEEN OLD FACES AND NEW FACES 
        from Florence.Tensor import shuffle_along_axis
        row_mapper = shuffle_along_axis(faces[:,:3],self.faces[:,:3],consider_sort=True)

        # UPDATE THE MAP
        boundary_face_to_element[:,:] = boundary_face_to_element[row_mapper,:]
        self.boundary_face_to_element = boundary_face_to_element
        
        return self.boundary_face_to_element


    def GetElementsFaceNumberingTet(self):
        """Finds which faces belong to which elements and which faces of the elements 
            they are e.g. 0, 1, 2 or 3.  

            output: 

                face_elements:              [2D array] nfaces x 2 array containing elements which have face
                                            on the boundary with their flags

                                            Note that this method also sets the self.face_to_element to face_elements,
                                            so the return value is not strictly necessary   
        """

        if isinstance(self.face_to_element,np.ndarray):
            if self.face_to_element.shape[0] > 1:
                return self.face_to_element

        assert self.elements is not None

        # GET ALL FACES FROM ELEMENT CONNECTIVITY
        if self.all_faces is None:
            self.GetFacesTet()


        all_faces = np.concatenate((self.elements[:,:3],self.elements[:,[0,1,3]],
            self.elements[:,[0,2,3]],self.elements[:,[1,2,3]]),axis=0).astype(np.int64)

        _,idx = unique2d(all_faces,consider_sort=True,order=False, return_index=True)
        face_elements = np.zeros((self.all_faces.shape[0],2),dtype=np.int64)

        face_elements[:,0] = idx % self.elements.shape[0]
        face_elements[:,1] = idx // self.elements.shape[0]

        self.face_to_element = face_elements
        return self.face_to_element


    def ArrangeFacesTet(self):
        """Arranges all the faces of tetrahedral elements 
            with triangular type node ordering """

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTet

        if self.all_faces is None:
            self.all_faces = self.GetFacesTet()
        if self.face_to_element is None:
            self.GetElementsFaceNumberingTet()

        # DETERMINE DEGREE
        p = self.InferPolynomialDegree()

        node_arranger = NodeArrangementTet(p-1)[0]
        # for i in range(self.face_to_element.shape[0]):
            # self.all_faces = self.elements[self.face_to_element[i,0],node_arranger[self.face_to_element[i,1],:]]

        self.all_faces = self.elements[self.face_to_element[:,0][:,None],node_arranger[self.face_to_element[:,1],:]]


    def GetElementsEdgeNumberingQuad(self):
        """Finds edges of elements and their flags saying which edge they are [0,1,2,3].
            At most a quad can have all its four edges on the boundary.

        output: 

            edge_elements:              [1D array] array containing elements which have edges
                                        on the boundary

                                        Note that this method sets the self.edge_to_element to edge_elements,
                                        so the return value is not strictly necessary
        """

        if isinstance(self.edge_to_element,np.ndarray):
            if self.edge_to_element.shape[0] > 1:
                return self.edge_to_element

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY 
        if self.all_edges is None:
            self.GetEdgesQuad()


        p = self.InferPolynomialDegree()
        
        # FIND WHICH FACE NODES ARE IN WHICH ELEMENT
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        node_arranger = NodeArrangementQuad(p-1)[0]

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        all_edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],self.elements[:,node_arranger[3,:]]),axis=0).astype(np.int64)

        all_edges, idx = unique2d(all_edges,consider_sort=True,order=False, return_index=True)
        edge_elements = np.zeros((all_edges.shape[0],2),dtype=np.int64)
        # edge_elements = np.zeros((self.edges.shape[0],2),dtype=np.int64)

        edge_elements[:,0] = idx % self.elements.shape[0]
        edge_elements[:,1] = idx // self.elements.shape[0]

        self.edge_to_element = edge_elements
        return self.edge_to_element


    def GetElementsWithBoundaryEdgesQuad(self):
        """Finds elements which have edges on the boundary.
            At most a quad can have all its four edges on the boundary.

        output: 

            boundary_edge_to_element:   [2D array] array containing elements which have face
                                        on the boundary [cloumn 0] and a flag stating which edges they are [column 1]

        """

        if isinstance(self.boundary_edge_to_element,np.ndarray):
            if self.boundary_edge_to_element.shape[1] > 1 and self.boundary_edge_to_element.shape[0] > 1:
                return self.boundary_edge_to_element

        assert self.edges is not None or self.elements is not None

        p = self.InferPolynomialDegree()
        
        # FIND WHICH FACE NODES ARE IN WHICH ELEMENT
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad
        node_arranger = NodeArrangementQuad(p-1)[0]

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        all_edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],self.elements[:,node_arranger[3,:]]),axis=0).astype(np.int64)

        # GET UNIQUE ROWS 
        uniques, idx, inv = unique2d(all_edges,consider_sort=True,order=False,return_index=True,return_inverse=True)

        # ROWS THAT APPEAR ONLY ONCE CORRESPOND TO BOUNDARY EDGES
        freqs_inv = itemfreq(inv)
        edges_ext_flags = freqs_inv[freqs_inv[:,1]==1,0]
        # NOT ARRANGED
        edges = uniques[edges_ext_flags,:] 

        # DETERMINE WHICH FACE OF THE ELEMENT THEY ARE
        boundary_edge_to_element = np.zeros((edges_ext_flags.shape[0],2),dtype=np.int64)

        # FURTHER RE-ARRANGEMENT / ARANGE THE NODES BASED ON THE ORDER THEY APPEAR
        # IN ELEMENT CONNECTIVITY
        # THIS STEP IS NOT NECESSARY INDEED - ITS JUST FOR RE-ARANGMENT OF EDGES
        all_edges_in_edges = in2d(all_edges,self.edges,consider_sort=True)
        all_edges_in_edges = np.where(all_edges_in_edges==True)[0]

        boundary_edge_to_element[:,0] = all_edges_in_edges % self.elements.shape[0]
        boundary_edge_to_element[:,1] = all_edges_in_edges // self.elements.shape[0]

        # ARRANGE FOR ANY ORDER OF BASES/ELEMENTS AND ASSIGN DATA MEMBERS
        self.boundary_edge_to_element = boundary_edge_to_element

        return self.boundary_edge_to_element


    def GetElementsWithBoundaryFacesHex(self):
        """Finds elements which have faces on the boundary.
            At most a hexahedral can have all its 8 faces on the boundary.

        output: 

            boundary_face_to_element:   [2D array] array containing elements which have face
                                        on the boundary [cloumn 0] and a flag stating which faces they are [column 1]

        """

        assert self.faces is not None or self.elements is not None

        if self.boundary_face_to_element is not None:
            return self.boundary_face_to_element

        # THIS METHOD ALWAYS RETURNS THE FACE TO ELEMENT ARRAY, AND DOES NOT CHECK
        # IF THIS HAS BEEN COMPUTED BEFORE, THE REASON BEING THAT THE FACES CAN COME 
        # EXTERNALLY WHOSE ARRANGEMENT WOULD NOT CORRESPOND TO THE ONE USED INTERNALLY
        # HENCE THIS MAPPING BECOMES NECESSARY

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex

        C = self.InferPolynomialDegree() - 1
        node_arranger = NodeArrangementHex(C)[0]
        
        all_faces = np.concatenate((np.concatenate((
                np.concatenate((np.concatenate((np.concatenate((self.elements[:,node_arranger[0,:]],
                self.elements[:,node_arranger[1,:]]),axis=0),self.elements[:,node_arranger[2,:]]),axis=0),
                self.elements[:,node_arranger[3,:]]),axis=0),self.elements[:,node_arranger[4,:]]),axis=0),
                self.elements[:,node_arranger[5,:]]),axis=0).astype(self.faces.dtype)

        all_faces_in_faces = in2d(all_faces,self.faces[:,:4],consider_sort=True)
        all_faces_in_faces = np.where(all_faces_in_faces==True)[0]

        boundary_face_to_element = np.zeros((all_faces_in_faces.shape[0],2),dtype=np.int64)
        boundary_face_to_element[:,0] = all_faces_in_faces % self.elements.shape[0]
        boundary_face_to_element[:,1] = all_faces_in_faces // self.elements.shape[0]


        # SO FAR WE HAVE COMPUTED THE ELEMENTS THAT CONTAIN FACES, HOWEVER 
        # NOTE THAT WE STILL HAVE NOT COMPUTED A MAPPING BETWEEN ELEMENTS AND 
        # FACES. WE ONLY KNOW WHICH ELEMENTS CONTAIN FACES FROM in2d.
        # WE NEED TO FIND THIS MAPPING NOW

        # WE NEED TO DO THIS DUMMY RECONSTRUCTION OF FACES BASED ON ELEMENTS
        faces = self.elements[boundary_face_to_element[:,0][:,None],
            node_arranger[boundary_face_to_element[:,1],:]].astype(self.faces.dtype)

        # CHECK FOR THIS CONDITION AS ARRANGEMENT IS NO LONGER MAINTAINED
        assert np.sum(faces[:,:4].astype(np.int64) - self.faces[:,:4].astype(np.int64)) == 0

        # NOW GET THE ROW MAPPING BETWEEN OLD FACES AND NEW FACES 
        from Florence.Tensor import shuffle_along_axis
        row_mapper = shuffle_along_axis(faces[:,:4],self.faces[:,:4],consider_sort=True)

        # UPDATE THE MAP
        boundary_face_to_element[:,:] = boundary_face_to_element[row_mapper,:]
        self.boundary_face_to_element = boundary_face_to_element
        
        return self.boundary_face_to_element


    def GetElementsFaceNumberingHex(self):
        """Finds which faces belong to which elements and which faces of the elements 
            they are e.g. 0, 1, 2 or 3.  

            output: 

                face_elements:              [2D array] nfaces x 2 array containing elements which have face
                                            on the boundary with their flags

                                            Note that this method also sets the self.face_to_element to face_elements,
                                            so the return value is not strictly necessary   
        """

        if isinstance(self.face_to_element,np.ndarray):
            if self.face_to_element.shape[0] > 1:
                return self.face_to_element

        assert self.elements is not None

        # GET ALL FACES FROM ELEMENT CONNECTIVITY
        if self.all_faces is None:
            self.GetFacesHex()

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex

        C = self.InferPolynomialDegree() - 1
        node_arranger = NodeArrangementHex(C)[0]
        
        all_faces = np.concatenate((np.concatenate((
                np.concatenate((np.concatenate((np.concatenate((self.elements[:,node_arranger[0,:]],
                self.elements[:,node_arranger[1,:]]),axis=0),self.elements[:,node_arranger[2,:]]),axis=0),
                self.elements[:,node_arranger[3,:]]),axis=0),self.elements[:,node_arranger[4,:]]),axis=0),
                self.elements[:,node_arranger[5,:]]),axis=0).astype(self.all_faces.dtype)

        _,idx = unique2d(all_faces,consider_sort=True,order=False, return_index=True)
        face_elements = np.zeros((self.all_faces.shape[0],2),dtype=np.int64)

        face_elements[:,0] = idx % self.elements.shape[0]
        face_elements[:,1] = idx // self.elements.shape[0]

        self.face_to_element = face_elements
        return self.face_to_element



    def ArrangeFacesHex(self):
        """Arranges all the faces of hexahedral elements 
            with quadrilateral type node ordering """

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex

        if self.all_faces is None:
            self.all_faces = self.GetFacesHex()
        if self.face_to_element is None:
            self.GetElementsFaceNumberingHex()

        # DETERMINE DEGREE
        p = self.InferPolynomialDegree()

        node_arranger = NodeArrangementHex(p-1)[0]

        self.all_faces = self.elements[self.face_to_element[:,0][:,None],node_arranger[self.face_to_element[:,1],:]]



    def Reader(self, filename=None, element_type="tri", reader_type="Salome", reader_type_format=None, 
        reader_type_version=None, order=0, **kwargs):
        """Convenience mesh reader method to dispatch call to subsequent apporpriate methods"""

        self.filename = filename
        self.reader_type = reader_type
        self.reader_type_format = reader_type_format
        self.reader_type_version = reader_type_version

        if self.reader_type is 'Salome': 
            self.Read(filename,element_type,order)
        elif reader_type is 'GID':
            self.ReadGIDMesh(filename,element_type,order)
        elif self.reader_type is 'ReadSeparate':
            # READ MESH FROM SEPARATE FILES FOR CONNECTIVITY AND COORDINATES
            from Florence.Utils import insensitive
            # return insensitive(kwargs.keys())
            for key in kwargs.keys():
                inkey = insensitive(key)
                if "connectivity" in inkey and "delimiter" not in inkey:
                    connectivity_file = kwargs.get(key)
                if "coordinate" in insensitive(key) and "delimiter" not in inkey:
                    coordinates_file = kwargs.get(key)

            self.ReadSeparate(connectivity_file,coordinates_file,element_type,
                delimiter_connectivity=',',delimiter_coordinates=',')
            # self.ReadSeparate(MainData.MeshInfo.ConnectivityFile,MainData.MeshInfo.CoordinatesFile,MainData.MeshInfo.MeshType,
            #   edges_file=MainData.MeshInfo.EdgesFile,delimiter_connectivity=',',delimiter_coordinates=',')
        elif self.reader_type is 'ReadHighOrderMesh':
            self.ReadHighOrderMesh(MainData.MeshInfo.FileName.split(".")[0],MainData.C,MainData.MeshInfo.MeshType)
        elif self.reader_type is 'ReadHDF5':
            self.ReadHDF5(filename)
        elif self.reader_type is 'HollowCircle':
            # self.HollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=4,ncirc=12)
            # self.HollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=7,ncirc=7) # isotropic
            self.HollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=False,nrad=7,ncirc=7)
        elif self.reader_type is 'Sphere':
            # mesh.Sphere()
            self.Sphere(points=10)





    def Read(self,*args,**kwargs):
        """Default mesh reader for binary and text files used for reading Salome meshes mainly.
        The method checks if edges/faces are provided by the mesh generator and if not computes them"""

        mesh = ReadMesh(*args,**kwargs)

        self.points = mesh.points
        self.elements = mesh.elements.astype(np.int64)
        if isinstance(mesh.edges,np.ndarray):
            self.edges = mesh.edges.astype(np.int64)
        if isinstance(mesh.faces,np.ndarray):
            self.faces = mesh.faces.astype(np.int64)
        self.nelem = np.int64(mesh.nelem)
        self.element_type = mesh.info 

        # RETRIEVE FACE/EDGE ATTRIBUTE
        if self.element_type == 'tri' and self.edges is None:
            # COMPUTE EDGES
            self.GetBoundaryEdgesTri()

        # DO NOT RELY ON SALOME FACE GENERATER
        if self.element_type == 'tet':
            self.edges = None
            self.faces = None
            # COMPUTE FACES & EDGES
            self.GetBoundaryFacesTet()
            self.GetBoundaryEdgesTet()


    def ReadSeparate(self,connectivity_file,coordinates_file,mesh_type, edges_file = None, faces_file = None,
        delimiter_connectivity=' ',delimiter_coordinates=' ', delimiter_edges=' ', delimiter_faces=' ',
        ignore_cols_connectivity=None,ignore_cols_coordinates=None,ignore_cols_edges=None,
        ignore_cols_faces=None,index_style='c'):
        """Read meshes when the element connectivity and nodal coordinates are written in separate files

        input:

            connectivity_file:              [str] filename containing element connectivity
            coordinates_file:               [str] filename containing nodal coordinates
            mesh_type:                      [str] type of mesh tri/tet/quad/hex 
            edges_file:                     [str] filename containing edges of the mesh (if not given gets computed)
            faces_file:                     [str] filename containing faces of the mesh (if not given gets computed)
            delimiter_connectivity:         [str] delimiter for connectivity_file - default is white space/tab
            delimiter_coordinates:          [str] delimiter for coordinates_file - default is white space/tab
            delimiter_edges:                [str] delimiter for edges_file - default is white space/tab
            delimiter_faces:                [str] delimiter for faces_file - default is white space/tab
            ignore_cols_connectivity:       [int] no of columns to be ignored (from the start) in the connectivity_file 
            ignore_cols_coordinates:        [int] no of columns to be ignored (from the start) in the coordinates_file
            ignore_cols_edges:              [int] no of columns to be ignored (from the start) in the connectivity_file 
            ignore_cols_faces:              [int] no of columns to be ignored (from the start) in the coordinates_file
            index_style:                    [str] either 'c' C-based (zero based) indexing or 'f' fortran-based
                                                  (one based) indexing for elements connectivity - default is 'c'

            """

        index = 0
        if index_style == 'c':
            index = 1

        from time import time; t1=time()    
        self.elements = np.loadtxt(connectivity_file,dtype=np.int64,delimiter=delimiter_connectivity) - index 
        # self.elements = np.fromfile(connectivity_file,dtype=np.int64,count=-1) - index
        self.points = np.loadtxt(coordinates_file,dtype=np.float64,delimiter=delimiter_coordinates)


        if ignore_cols_connectivity != None:
            self.elements = self.elements[ignore_cols_connectivity:,:]
        if ignore_cols_coordinates != None:
            self.points = self.points[ignore_cols_coordinates:,:]

        if (mesh_type == 'tri' or mesh_type == 'quad') and self.points.shape[1]>2:
            self.points = self.points[:,:2]

        self.element_type = mesh_type
        self.nelem = self.elements.shape[0]
        # self.edges = None 
        if edges_file is None:
            if mesh_type == "tri":
                self.GetBoundaryEdgesTri()
            elif mesh_type == "tet":
                self.GetBoundaryEdgesTet()
        else:
            self.edges = np.loadtxt(edges_file,dtype=np.int64,delimiter=delimiter_edges) - index
            if ignore_cols_edges !=None:
                self.edges = self.edges[ignore_cols_edges:,:] 

        if faces_file is None:
            if mesh_type == "tet":
                self.GetBoundaryFacesTet()
        else:
            self.faces = np.loadtxt(faces_file,dtype=np.int64,delimiter=delimiter_edges) - index
            if ignore_cols_faces !=None:
                self.faces = self.faces[ignore_cols_faces:,:]




    def ReadHighOrderMesh(self,filename,C,element_type="tri"):
        """Convenience function for reading high order mesh"""

        if element_type=="tri":

            self.elements = np.loadtxt(filename+"_elements_P"+str(C+1)+".dat")
            self.edges = np.loadtxt(filename+"_edges_P"+str(C+1)+".dat")
            self.points = np.loadtxt(filename+"_points_P"+str(C+1)+".dat")
            self.nelem = self.elements.shape[0]
            self.element_type = element_type

            self.elements = self.elements.astype(np.uint64)
            self.edges = self.edges.astype(np.uint64)

        elif element_type=="tet":
            pass




    def ReadGIDMesh(self,filename,mesh_type,polynomial_order = 0):
        """Read GID meshes"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        self.element_type = mesh_type
        ndim, self.nelem, nnode, nboundary = np.fromfile(filename,dtype=np.int64,count=4,sep=' ')
    
        if ndim==2 and mesh_type=="tri":
            content = np.fromfile(filename,dtype=np.float64,count=4+3*nnode+4*self.nelem,sep=' ')
            self.points = content[4:4+3*nnode].reshape(nnode,3)[:,1:]
            self.elements = content[4+3*nnode:4+3*nnode+4*self.nelem].reshape(self.nelem,4)[:,1:].astype(np.int64)
            self.elements -= 1 

            self.GetBoundaryEdgesTri()

        if ndim==3 and mesh_type=="tet":
            content = np.fromfile(filename,dtype=np.float64,count=4+4*nnode+5*self.nelem+9*nboundary,sep=' ')
            self.points = content[4:4+4*nnode].reshape(nnode,4)[:,1:]
            self.elements = content[4+4*nnode:4+4*nnode+5*self.nelem].reshape(self.nelem,5)[:,1:].astype(np.int64)
            self.elements -= 1

            face_flags = content[4*nnode+5*self.nelem+4:].reshape(nboundary,9)[:,1:].astype(np.int64)
            self.faces = np.ascontiguousarray(face_flags[:,1:4] - 1)
            self.face_to_surface = np.ascontiguousarray(face_flags[:,7] - 1)
            # self.boundary_face_to_element = np.ascontiguousarray(face_flags[:,0])

            # self.GetBoundaryFacesTet()
            self.GetBoundaryEdgesTet()



    def ReadGmsh(self,filename):
        """Read gmsh (.msh) file"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        try:
            fid = open(filename, "r")
        except IOError:
            print "File '%s' not found." % (filename)
            sys.exit()

        self.filename = filename

        rem_nnode, rem_nelem, rem_faces = long(1e09), long(1e09), long(1e09)
        face_counter = 0
        for line_counter, line in enumerate(open(filename)):
            item = line.rstrip()
            plist = item.split()
            if plist[0] == "Dimension":
                self.ndim = plist[1]
            elif plist[0] == "Vertices":
                rem_nnode = line_counter+1
                continue
            elif plist[0] == "Triangles":
                rem_faces = line_counter+1
                continue
            elif plist[0] == "Tetrahedra":
                rem_nelem = line_counter+1
                continue
            if rem_nnode == line_counter:
                self.nnode = int(plist[0])
            if rem_faces == line_counter:
                face_counter = int(plist[0])
            if rem_nelem == line_counter:
                self.nelem = int(plist[0])
                break

        # Re-read
        points, elements = [],[]
        for line_counter, line in enumerate(open(filename)):
            item = line.rstrip()
            plist = item.split()
            if line_counter > rem_nnode and line_counter < self.nnode+rem_nnode+1:
                points.append([float(i) for i in plist[:3]])
            if line_counter > rem_nelem and line_counter < self.nelem+rem_nelem+1:
                elements.append([long(i) for i in plist[:4]])

        self.points = np.array(points,copy=True)
        self.elements = np.array(elements,copy=True) - 1

        # print self.ndim, self.nnode, self.nelem, rem_nnode, rem_nelem, rem_faces
        self.element_type = "tet"
        self.GetBoundaryFacesTet()
        self.GetBoundaryEdgesTet()

        return 

        # OTHER VARIANTS OF GMSH
        from gmsh import Mesh as msh
        self.meshname = mshfile
        mesh = msh()
        mesh.read_msh(filename)
        self.points = mesh.Verts
        # print dir(mesh)
        self.elements = mesh.Elmts 
        print self.elements


    def ReadHDF5(self,filename):
        """Read mesh from MATLAB HDF5 file format"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        DictOutput = loadmat(filename)

        # GENERIC READER - READS EVERYTHING FROM HDF5 AND ASSIGNS IT TO MESH OBJECT
        for key, value in DictOutput.items():
            if isinstance(DictOutput[key],np.ndarray):
                if "elements" in key or "edge" in key or "face" in key:
                    setattr(self, key, np.ascontiguousarray(value).astype(np.uint64))
                else:
                    setattr(self, key, np.ascontiguousarray(value))
            else:
                setattr(self, key, value)

        if isinstance(self.element_type,np.ndarray):
            self.element_type = str(self.element_type[0])



    def SimplePlot(self, to_plot='faces', 
        color=None, save=False, filename=None, figure=None, show_plot=True):
        """Simple mesh plot

            to_plot:        [str] only for 3D. 'faces' to plot only boundary faces
                            or 'all_faces' to plot all faces
            """

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        if color is None:
            color=(197/255.,241/255.,197/255.)

        if save:
            if filename is None:
                warn('File name not given. I am going to write one in the current directory')
                filename = PWD(__file__) + "/output.png"
            else:
                if filename.split(".")[-1] == filename:
                    filename += ".png"

        if self.element_type == "tri" or self.element_type == "quad":
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            if figure is None:
                figure = plt.figure()

        elif self.element_type == "tet" or self.element_type == "hex":
            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab 

            if to_plot == 'all_faces':
                faces = self.all_faces
                if self.all_faces is None:
                    self.GetFaces()
            else:
                faces = self.faces
                if self.faces is None:
                    self.GetBoundaryFaces()

            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))


        if self.element_type == "tri":

            plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3],color='k')
            plt.axis("equal")
            plt.axis('off')
            if show_plot:
                plt.show()

        elif self.element_type == "tet":

            # from mpl_toolkits.mplot3d import Axes3D
            # import matplotlib.pyplot as plt 
            # figure = plt.figure()

            # # FOR PLOTTING ELEMENTS
            # for elem in range(self.elements.shape[0]):
            #   coords = self.points[self.elements[elem,:],:]
            #   plt.gca(projection='3d')
            #   plt.plot(coords[:,0],coords[:,1],coords[:,2],'-bo')

            # # FOR PLOTTING ONLY BOUNDARY FACES - MATPLOTLIB SOLUTION
            # if self.faces.shape[1] == 3:
            #     for face in range(self.faces.shape[0]):
            #         coords = self.points[self.faces[face,:3],:]
            #         plt.gca(projection='3d')
            #         plt.plot(coords[:,0],coords[:,1],coords[:,2],'-ko')
            # else:
            #     for face in range(self.faces.shape[0]):
            #         coords = self.points[self.faces[face,:3],:]
            #         coords_all = self.points[self.faces[face,:],:]
            #         plt.gca(projection='3d')
            #         plt.plot(coords[:,0],coords[:,1],coords[:,2],'-k')
            #         plt.plot(coords_all[:,0],coords_all[:,1],coords_all[:,2],'ko')

            # plt.axis("equal")
            # plt.show()
            # return


            # MAYAVI.MLAB SOLUTION
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],
                self.points[:,2],self.faces[:,:3],color=color)
            radius = 1e-00
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2], faces[:,:3],
                line_width=radius,tube_radius=radius,color=(0,0,0),
                representation='wireframe')

            # svpoints = self.points[np.unique(self.faces),:]
            # mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=0.005)

            # mlab.view(azimuth=135, elevation=45, distance=7, focalpoint=None,
            #     roll=0, reset_roll=True, figure=None)

            if show_plot:
                mlab.show()

        elif self.element_type=="quad":

            C = self.InferPolynomialDegree() - 1
            from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad

            edge_elements = self.GetElementsEdgeNumberingQuad()
            reference_edges = NodeArrangementQuad(C)[0]
            reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
            reference_edges = np.delete(reference_edges,1,1)

            self.GetEdgesQuad()
            x_edges = np.zeros((C+2,self.all_edges.shape[0]))
            y_edges = np.zeros((C+2,self.all_edges.shape[0]))

            BasesOneD = np.eye(2,2)
            for iedge in range(self.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = self.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                x_edges[:,iedge], y_edges[:,iedge] = self.points[edge,:].T

            plt.plot(x_edges,y_edges,'-k')

            plt.axis('equal')
            plt.axis('off')
            if show_plot:
                plt.show()


        elif self.element_type == "hex":
            from Florence.PostProcessing import PostProcess
            tmesh = PostProcess.TessellateHexes(self,np.zeros_like(self.points),plot_points=True,
                interpolation_degree=0)

            Xplot = tmesh.points
            Tplot = tmesh.elements
            # color=(197/255.,241/255.,197/255.)
            point_line_width = .002

            trimesh_h = mlab.triangular_mesh(Xplot[:,0], Xplot[:,1], Xplot[:,2], Tplot,
                    line_width=point_line_width,color=color)

            src = mlab.pipeline.scalar_scatter(tmesh.x_edges.T.copy().flatten(), 
                tmesh.y_edges.T.copy().flatten(), tmesh.z_edges.T.copy().flatten())
            src.mlab_source.dataset.lines = tmesh.connections
            lines = mlab.pipeline.stripper(src)
            h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=3)

            # mlab.view(azimuth=135, elevation=45, distance=7, focalpoint=None,
                # roll=0, reset_roll=True, figure=None)

            if show_plot:
                mlab.show()

        else:
            raise NotImplementedError("SimplePlot for "+self.element_type+" not implemented yet")


        if save:
            if self.InferSpatialDimension() == 2:
                plt.savefig(filename,format="png",dpi=300)
            else:
                mlab.savefig(filename,dpi=300)



    def PlotMeshNumbering(self):
        """Plots element and node numbers on top of the triangular mesh"""

        import matplotlib.pyplot as plt
        import matplotlib as mpl

        if self.element_type == "tri":

            fig = plt.figure()
            plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3])
            plt.tricontourf(self.points[:,0], self.points[:,1], self.elements[:,:3], np.ones(self.points.shape[0]), 100,alpha=0.3)

            for i in range(0,self.elements.shape[0]):
                coord = self.points[self.elements[i,:],:]
                x_avg = np.sum(coord[:,0])/self.elements.shape[1]
                y_avg = np.sum(coord[:,1])/self.elements.shape[1]
                plt.text(x_avg,y_avg,str(i),backgroundcolor='#F88379',ha='center')

            for i in range(0,self.points.shape[0]):
                plt.text(self.points[i,0],self.points[i,1],str(i),backgroundcolor='#0087BD',ha='center')

            # plt.axis('equal')
            # plt.show(block=False)
            plt.show()

        elif self.element_type == "quad":

            fig = plt.figure()
            point_radius = 3.
 
            C = self.InferPolynomialDegree() - 1
            from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad

            edge_elements = self.GetElementsEdgeNumberingQuad()
            reference_edges = NodeArrangementQuad(C)[0]
            reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
            reference_edges = np.delete(reference_edges,1,1)

            self.GetEdgesQuad()
            x_edges = np.zeros((C+2,self.all_edges.shape[0]))
            y_edges = np.zeros((C+2,self.all_edges.shape[0]))

            BasesOneD = np.eye(2,2)
            for iedge in range(self.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = self.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                x_edges[:,iedge], y_edges[:,iedge] = self.points[edge,:].T

                
            plt.plot(x_edges,y_edges,'-k')
            
            for i in range(self.elements.shape[0]):
                coord = self.points[self.elements[i,:],:]
                x_avg = np.sum(coord[:,0])/self.elements.shape[1]
                y_avg = np.sum(coord[:,1])/self.elements.shape[1]
                plt.text(x_avg,y_avg,str(i),backgroundcolor='#F88379',ha='center')

            for i in range(0,self.points.shape[0]):
                plt.text(self.points[i,0],self.points[i,1],str(i),backgroundcolor='#0087BD',ha='center')

            plt.show()

        elif self.element_type == "tet":

            import matplotlib as mpl
            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab

            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
            view = mlab.view()
            figure.scene.disable_render = True

            color = mpl.colors.hex2color('#F88379')

            linewidth = 3.
            # trimesh_h = mlab.triangular_mesh(self.points[:,0], 
                # self.points[:,1], self.points[:,2], self.faces[:,:3],
                # line_width=linewidth,tube_radius=linewidth,color=(0,0.6,0.4),
                # representation='surface') # representation='surface'
            trimesh_h = mlab.triangular_mesh(self.points[:,0], 
                self.points[:,1], self.points[:,2], self.faces[:,:3],
                line_width=linewidth,tube_radius=linewidth,color=(0,0.6,0.4),
                representation='wireframe') # representation='surface'

            # CHANGE LIGHTING OPTION
            trimesh_h.actor.property.interpolation = 'phong'
            trimesh_h.actor.property.specular = 0.1
            trimesh_h.actor.property.specular_power = 5

            # ELEMENT NUMBERING
            # for i in range(0,self.elements.shape[0]):
            #     coord = self.points[self.elements[i,:],:]
            #     x_avg = np.sum(coord[:,0])/self.elements.shape[1]
            #     y_avg = np.sum(coord[:,1])/self.elements.shape[1]
            #     z_avg = np.sum(coord[:,2])/self.elements.shape[1]
                
            #     # mlab.text3d(x_avg,y_avg,z_avg,str(i),color=color)
            #     mlab.text3d(x_avg,y_avg,z_avg,str(i),color=(0,0,0.),scale=2)

            # POINT NUMBERING
            # for i in range(self.points.shape[0]):
                # text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=2)
                # if i==0:
                #     text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=500)
                # else:
                #     text_obj.position = self.points[i,:]
                #     text_obj.text = str(i)
                #     text_obj.scale = [2,2,2]

                # if self.points[i,2] == 0:
                    # text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=0.5)

            # for i in range(self.faces.shape[0]):
            #     if i==3:
            #         for j in self.faces[i,:]:
            #             text_obj = mlab.text3d(self.points[j,0],self.points[j,1],self.points[j,2],str(j),color=(0,0,0.),scale=10.5)

            for i in range(self.faces.shape[0]):
                for j in range(self.faces.shape[1]):
                    if self.points[self.faces[i,j],2] < 30:
                        # print i, self.faces[i,:]
                        text_obj = mlab.text3d(self.points[self.faces[i,j],0],
                            self.points[self.faces[i,j],1],self.points[self.faces[i,j],2],str(self.faces[i,j]),color=(0,0,0.),scale=0.05)



            figure.scene.disable_render = False

            # mlab.view(*view)
            mlab.show()



    def WriteVTK(self, filename=None, result=None):
        """Write mesh/results to vtu"""

        assert self.elements is not None
        assert self.points is not None

        elements = np.copy(self.elements)

        cellflag = None
        if self.element_type =='tri':
            cellflag = 5
            if self.elements.shape[1]==6:
                cellflag = 22
        elif self.element_type =='quad':
            cellflag = 9
            if self.elements.shape[1]==8:
                cellflag = 23
        if self.element_type =='tet':
            cellflag = 10
            if self.elements.shape[1]==10:
                cellflag = 24
                # CHANGE NUMBERING ORDER FOR PARAVIEW
                para_arange = [0,4,1,6,2,5,7,8,9,3]
                elements = elements[:,para_arange]    
        elif self.element_type == 'hex':
            cellflag = 12
            if self.elements.shape[1] == 20:
                cellflag = 25

        if filename is None:
            warn('File name not specified. I am going to write one in the current directory')
            filename = PWD(__file__) + "/output.vtu"

        if result is None:
            write_vtu(Verts=self.points, Cells={cellflag:elements},fname=filename)
        else:
            if isinstance(result, np.ndarray):
                if result.ndim > 1:
                    if result.shape[1] == 1:
                        result = result.flatten()

                if result.ndim > 1:
                    if result.shape[0] == self.nelem:
                        write_vtu(Verts=self.points, Cells={cellflag:elements},cvdata=result,fname=filename)
                    elif result.shape[0] == self.points.shape[0]:
                        write_vtu(Verts=self.points, Cells={cellflag:elements},pvdata=result,fname=filename)
                else:
                    if result.shape[0] == self.nelem:
                        write_vtu(Verts=self.points, Cells={cellflag:elements},cdata=result,fname=filename)
                    elif result.shape[0] == self.points.shape[0]:
                        write_vtu(Verts=self.points, Cells={cellflag:elements},pdata=result,fname=filename)



    def WriteHDF5(self, filename=None, external_fields=None):
        """Write to MATLAB's HDF5 format

            external_fields:        [dict or tuple] of fields to save together with the mesh
                                    for instance desired results. If a tuple is given keys in
                                    dictionary will be named results_0, results_1 and so on"""

        Dict = self.__dict__

        if external_fields is not None:
            if isinstance(external_fields,dict):
                Dict.update(external_fields)
            elif isinstance(external_fields,tuple):            
                for counter, fields in enumerate(external_fields):
                    Dict['results_'+str(counter)] = fields
            else:
                raise AssertionError("Fields should be eirther tuple or a dict")

        for key in Dict.keys():
            if Dict[str(key)] is None:
                del Dict[str(key)]

        if filename is None:
            pwd = os.path.dirname(os.path.realpath(__file__))
            filename = pwd+'/output.mat'

        savemat(filename, Dict, do_compression=True)



    @staticmethod
    def MeshPyTri(points,facets,*args,**kwargs):
        """MeshPy backend for generating linear triangular mesh"""
        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        return triangle.build(info,*args,**kwargs)


    def Rectangle(self,lower_left_point=(0,0), upper_right_point=(2,1), 
        nx=5, ny=5, element_type="tri"):
        """Creates a qud/tri mesh of on rectangle"""

        if element_type != "tri" and element_type != "quad":
            raise ValueError("Element type should either be tri or quad")

        if self.elements is not None and self.points is not None:
            self.__reset__()

        if (lower_left_point[0] > upper_right_point[0]) or \
            (lower_left_point[1] > upper_right_point[1]):
            raise ValueError("Incorrect coordinate for lower left and upper right vertices")


        from scipy.spatial import Delaunay

        x=np.linspace(lower_left_point[0],upper_right_point[0],nx+1)
        y=np.linspace(lower_left_point[1],upper_right_point[1],ny+1)

        X,Y = np.meshgrid(x,y)
        coordinates = np.dstack((X.ravel(),Y.ravel()))[0,:,:]

        if element_type == "tri":
            tri_func = Delaunay(coordinates)
            self.element_type = "tri"
            self.elements = tri_func.simplices
            self.nelem = self.elements.shape[0] 
            self.points = tri_func.points
            self.GetBoundaryEdgesTri()

        elif element_type == "quad":

            self.nelem = int(nx*ny)
            elements = np.zeros((self.nelem,4),dtype=np.int64)

            dum_0 = np.arange((nx+1)*ny)
            dum_1 = np.array([(nx+1)*i+nx for i in range(ny)])
            col0 = np.delete(dum_0,dum_1)
            elements[:,0] = col0
            elements[:,1] = col0 + 1
            elements[:,2] = col0 +  nx + 2
            elements[:,3] = col0 +  nx + 1

            self.nnode = int((nx+1)*(ny+1))
            self.element_type = "quad"
            self.elements = elements
            self.points = coordinates
            self.GetBoundaryEdgesQuad()
            self.GetEdgesQuad()


    def Square(self, lower_left_point=(0,0), side_length=1, nx=5, ny=5, n=None, element_type="tri"):
        """Creates a quad/tri mesh on a square

            input:
                lower_left_point            [tuple] of lower left vertex of the square
                side_length:                [int] length of side 
                nx,ny:                      [int] number of discretisation in each direction
                n:                          [int] number of discretisation in all directions
                                            i.e. nx=ny=n. Overrides nx,ny
            """

        if n != None:
            nx,ny = n,n

        upper_right_point = (side_length+lower_left_point[0],side_length+lower_left_point[1])
        self.Rectangle(lower_left_point=lower_left_point,
            upper_right_point=upper_right_point,nx=nx,ny=ny,element_type=element_type)



    def Triangle(self, c1=(0.,0.), c2=(0.,1.), c3=(1.,0.), npoints=10, element_type="tri", equally_spaced=True):
        """Creates a tri/quad mesh on a triangular region, given coordinates of the three
            nodes of the triangle

            input:
                npoints:                    [int] number of discritsation
        """

        if not isinstance(c1,tuple) or not isinstance(c2,tuple) or not isinstance(c3,tuple):
            raise ValueError("The coordinates c1, c2 and c3 should be given in tuples of two elements each (x,y)")

        npoints = int(npoints)


        npoints = npoints - 1
        if npoints < 0:
            npoints = 0 

        c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
        opoints = np.vstack((c1,c2,c3))
        oelements = np.array([[0,1,2]])

        if element_type=="tri":
            mesh = self.TriangularProjection(points=opoints, npoints=npoints, equally_spaced=equally_spaced)
            self.__update__(mesh)


        elif element_type == "quad":

            # SPLIT THE TRIANGLE INTO 3 QUADS
            omesh = Mesh()
            omesh.element_type="tri"
            omesh.elements = oelements
            omesh.nelem = omesh.elements.shape[0]
            omesh.points = opoints
            omesh.GetBoundaryEdges()

            sys.stdout = open(os.devnull, "w")
            omesh.ConvertTrisToQuads()
            sys.stdout = sys.__stdout__ 


            npoints = int(npoints/2) + 1
            mesh = self.QuadrilateralProjection(points=omesh.points[omesh.elements[0,:],:],
                npoints=npoints, equally_spaced=equally_spaced)
            for i in range(1,omesh.nelem):
                mesh += self.QuadrilateralProjection(points=omesh.points[omesh.elements[i,:],:], 
                    npoints=npoints, equally_spaced=equally_spaced)

            self.__update__(mesh)




    def Arc(self, center=(0.,0.), radius=1., nrad=16, ncirc=40, 
        start_angle=0., end_angle=np.pi/2., element_type="tri", refinement=False, refinement_level=2):
        """Creates a structured quad/tri mesh on an arc

            input:
                start_angle/end_angle:      [float] starting and ending angles in radians. Angle
                                            is measured anti-clockwise. Default start angle is 
                                            positive x-axis
                refinement_level:           [int] number of elements that each element has to be 
                                            splitted to 

        """

        # CHECK FOR ANGLE
        PI = u"\u03C0".encode('utf-8').strip()
        EPS = np.finfo(np.float64).eps
        if np.abs(start_angle) + EPS > 2.*np.pi:
            raise ValueError("The starting angle should be either in range [-2{},0] or [0,2{}]".format(PI,PI))
        if np.abs(end_angle) + EPS > 2.*np.pi:
            raise ValueError("The end angle should be either in range [-2{},0] or [0,2{}]".format(PI,PI))


        if np.sign(start_angle) == np.sign(end_angle):
            total_angle = np.abs(end_angle - start_angle)
            if np.isclose(total_angle,0.) or total_angle > 2.*np.pi:
                self.Circle(center=center, radius=radius, nrad=nrad, ncirc=ncirc, element_type=element_type)
                return 

        if not isinstance(center,tuple):
            raise ValueError("The center of the arc should be given in a tuple with two elements (x,y)")

        self.__reset__()

        if refinement:
            ndivider = refinement_level
        else:
            ndivider = 1

        ncirc = int(ncirc/ndivider)
        nrad = int(nrad/ndivider)

        if ncirc % 2 != 0 or ncirc < 2:
            ncirc = (ncirc // 2)*2 + 2 

        radii = radius

        radius = np.linspace(0,radii,nrad+1)[1:] 
        t = np.linspace(start_angle,end_angle,ncirc+1) 
        x = radius[0]*np.cos(t)[::-1]
        y = radius[0]*np.sin(t)[::-1]

        points = np.zeros((ncirc+2,2),dtype=np.float64)
        points[0,:] = [0.,0.]
        points[1:,:] = np.array([x,y]).T

        self.elements = np.zeros((ncirc // 2,4),dtype=np.int64)
        aranger = np.arange(ncirc // 2)
        self.elements[:,1] = 2*aranger + 1
        self.elements[:,2] = 2*aranger + 2
        self.elements[:,3] = 2*aranger + 3

        for i in range(1,nrad):
            t = np.linspace(start_angle,end_angle,ncirc+1)
            x = radius[i]*np.cos(t)[::-1] 
            y = radius[i]*np.sin(t)[::-1]
            points = np.vstack((points,np.array([x,y]).T))

        points[:,0] += center[0]
        points[:,1] += center[1]

        elements = np.zeros((ncirc,4),dtype=np.int64) 
        for i in range(1,nrad):
            aranger = np.arange(1+ncirc*(i-1),ncirc*i+1)
            elements[:,0] = aranger + i - 1
            elements[:,1] = aranger + i + ncirc
            elements[:,2] = aranger + i + ncirc + 1
            elements[:,3] = aranger + i

            self.elements = np.concatenate((self.elements,elements),axis=0)


        makezero(points)
        self.points = points
        self.elements[:ncirc // 2,:] = self.elements[:ncirc // 2, [1,2,3,0]]

        self.element_type = "quad"
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdges()

        if refinement:
            mesh = self.QuadrilateralProjection(points=self.points[self.elements[0,:],:], npoints=ndivider)
            for i in range(1,self.nelem):
                mesh += self.QuadrilateralProjection(points=self.points[self.elements[i,:],:], npoints=ndivider)
            self.__update__(mesh)
        
        if element_type == "tri":
            sys.stdout = open(os.devnull, "w")
            self.ConvertQuadsToTris()
            sys.stdout = sys.__stdout__            



    def Circle(self, center=(0.,0.), radius=1., nrad=16, ncirc=40, 
        element_type="tri", refinement=False, refinement_level=2):
        """Creates a structured quad/tri mesh on circle 

        """

        if not isinstance(center,tuple):
            raise ValueError("The center of the circle should be given in a tuple with two elements (x,y)")

        self.__reset__()

        if refinement:
            ndivider = refinement_level
            if nrad==1: nrad=2
        else:
            ndivider = 1

        ncirc = int(ncirc/ndivider)
        nrad = int(nrad/ndivider)
        

        if ncirc % 8 != 0 or ncirc < 8:
            ncirc = (ncirc // 8)*8 + 8

        radii = radius

        radius = np.linspace(0,radii,nrad+1)[1:] 
        t = np.linspace(0,2*np.pi,ncirc+1) 
        x = radius[0]*np.sin(t)[::-1][:-1] 
        y = radius[0]*np.cos(t)[::-1][:-1] 

        points = np.zeros((ncirc+1,2),dtype=np.float64)
        points[0,:] = [0.,0.]
        points[1:,:] = np.array([x,y]).T


        self.elements = np.zeros((ncirc // 2,4),dtype=np.int64)
        aranger = np.arange(ncirc // 2)
        self.elements[:,1] = 2*aranger + 1
        self.elements[:,2] = 2*aranger + 2
        self.elements[:,3] = 2*aranger + 3
        self.elements[-1,-1] = 1

        for i in range(1,nrad):
            t = np.linspace(0,2*np.pi,ncirc+1); 
            x = radius[i]*np.sin(t)[::-1][:-1]; 
            y = radius[i]*np.cos(t)[::-1][:-1];
            points = np.vstack((points,np.array([x,y]).T))

        points[:,0] += center[0]
        points[:,1] += center[1]

        elements = np.zeros((ncirc,4),dtype=np.int64) 
        for i in range(1,nrad):
            aranger = np.arange(1+ncirc*(i-1),ncirc*i+1)
            elements[:,0] = aranger
            elements[:,1] = aranger + ncirc
            elements[:,2] = np.append((aranger + 1 + ncirc)[:-1],i*ncirc+1)
            elements[:,3] = np.append((aranger + 1)[:-1],1+(i-1)*ncirc)

            self.elements = np.concatenate((self.elements,elements),axis=0)

        makezero(points)
        self.points = points
        self.elements[:ncirc // 2,:] = self.elements[:ncirc // 2, [1,2,3,0]]

        self.element_type = "quad"
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdges()

        if refinement:
            mesh = self.QuadrilateralProjection(points=self.points[self.elements[0,:],:], npoints=ndivider)
            for i in range(1,self.nelem):
                mesh += self.QuadrilateralProjection(points=self.points[self.elements[i,:],:], npoints=ndivider)
            self.__update__(mesh)

            # SECOND LEVEL OF REFINEMENT IF NEEDED
            # mesh = self.QuadrilateralProjection(points=self.points[self.elements[0,:],:], npoints=2)
            # for i in range(1,self.nelem):
            #     mesh += self.QuadrilateralProjection(points=self.points[self.elements[i,:],:], npoints=2)
            # self.__update__(mesh)
        
        if element_type == "tri":
            sys.stdout = open(os.devnull, "w")
            self.ConvertQuadsToTris()
            sys.stdout = sys.__stdout__


    def HollowArc(self, center=(0.,0.), inner_radius=1., outer_radius=2., nrad=16, ncirc=40, 
        start_angle=0., end_angle=np.pi/2., element_type="tri", refinement=False, refinement_level=2):
        """Creates a structured quad/tri mesh on a hollow arc (i.e. two arc bounded by straight lines)

            input:
                start_angle/end_angle:      [float] starting and ending angles in radians. Angle
                                            is measured anti-clockwise. Default start angle is 
                                            positive x-axis
                refinement_level:           [int] number of elements that each element has to be 
                                            splitted to 

        """

        # CHECK FOR ANGLE
        PI = u"\u03C0".encode('utf-8').strip()
        EPS = np.finfo(np.float64).eps
        if np.abs(start_angle) + EPS > 2.*np.pi:
            raise ValueError("The starting angle should be either in range [-2{},0] or [0,2{}]".format(PI,PI))
        if np.abs(end_angle) + EPS > 2.*np.pi:
            raise ValueError("The end angle should be either in range [-2{},0] or [0,2{}]".format(PI,PI))


        if np.sign(start_angle) == np.sign(end_angle):
            total_angle = np.abs(end_angle - start_angle)
            if np.isclose(total_angle,0.) or total_angle > 2.*np.pi:
                self.Circle(center=center, radius=radius, nrad=nrad, ncirc=ncirc, element_type=element_type)
                return 

        if not isinstance(center,tuple):
            raise ValueError("The center of the arc should be given in a tuple with two elements (x,y)")

        self.__reset__()

        if refinement:
            ndivider = refinement_level
        else:
            ndivider = 1

        ncirc = int(ncirc/ndivider)
        nrad = int(nrad/ndivider) + 2

        if ncirc % 2 != 0 or ncirc < 2:
            ncirc = (ncirc // 2)*2 + 2 


        radius = np.linspace(inner_radius,outer_radius,nrad)
        points = np.zeros((1,2),dtype=np.float64)
        for i in range(nrad):
            t = np.linspace(start_angle,end_angle,ncirc+1)
            x = radius[i]*np.cos(t)[::-1] 
            y = radius[i]*np.sin(t)[::-1]
            points = np.vstack((points,np.array([x,y]).T))
        points = points[ncirc+2:,:]

        points[:,0] += center[0]
        points[:,1] += center[1]
        makezero(points)
        self.points = points

        self.elements = np.zeros((1,4),dtype=np.int64)
        elements = np.zeros((ncirc,4),dtype=np.int64) 
        for i in range(nrad-2):
            aranger = np.arange(ncirc*i,ncirc*(i+1))
            elements[:,0] = aranger + i
            elements[:,1] = aranger + i + ncirc + 1
            elements[:,2] = aranger + i + ncirc + 2
            elements[:,3] = aranger + i + 1

            self.elements = np.concatenate((self.elements,elements),axis=0)
        self.elements = self.elements[1:,:]


        self.element_type = "quad"
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdges()

        if refinement:
            mesh = self.QuadrilateralProjection(points=self.points[self.elements[0,:],:], npoints=ndivider)
            for i in range(1,self.nelem):
                mesh += self.QuadrilateralProjection(points=self.points[self.elements[i,:],:], npoints=ndivider)
            self.__update__(mesh)


        if element_type == "tri":
            sys.stdout = open(os.devnull, "w")
            self.ConvertQuadsToTris()
            sys.stdout = sys.__stdout__ 


    def HollowCircle(self,center=(0,0),inner_radius=1.0,outer_radius=2.,element_type='tri',isotropic=True,nrad=5,ncirc=10):
        """Generates isotropic and anisotropic tri and quad meshes on a hollow circle.

        input:

            center:             [tuple] containing the (x,y) coordinates of the center of the circle
            inner_radius:       [double] radius of inner circle 
            outer_radius:       [double] radius of outer circle
            element_type:       [str] tri for triangular mesh and quad for quadrilateral mesh 
            isotropic:          [boolean] option for isotropy or anisotropy of the mesh 
            nrad:               [int] number of disrectisation in the radial direction
            ncirc:              [int] number of disrectisation in the circumferential direction

        output:                 [Mesh] an instance of the Mesh class  
        """

        # FOR SAFETY, RESET THE CLASS
        self.__reset__()

        if np.allclose(inner_radius,0):
            raise ValueError('inner_radius cannot be zero')

        t = np.linspace(0,2*np.pi,ncirc+1)
        if isotropic is True:
            radii = np.linspace(inner_radius,outer_radius,nrad+1)
        else:
            base = 3
            radii = np.zeros(nrad+1,dtype=np.float64)
            mm = np.linspace(np.power(inner_radius,1./base),np.power(outer_radius,1./base),nrad+1)
            for i in range(0,nrad+1):
                radii[i] = mm[i]**base


            # base = 3
            # mm = np.linspace(np.power(inner_radius,1./base),np.power(2.,1./base),nrad+1)
            # mm = np.append(mm,np.linspace(2,outer_radius,nrad+1)) 
            # radii = np.zeros(mm.shape[0],dtype=np.float64)
            # for i in range(0,mm.shape[0]):
            #   radii[i] = mm[i]**base


        # dd =   np.logspace(inner_radius,outer_radius,nrad+1,base=2)/2**np.linspace(inner_radius,outer_radius,nrad+1)
        # print dd*np.linspace(inner_radius,outer_radius,nrad+1)
        # print np.logspace(0,1.5,nrad+1,base=2)
        

        xy = np.zeros((radii.shape[0]*t.shape[0],2),dtype=np.float64)
        for i in range(0,radii.shape[0]):
            xy[i*t.shape[0]:(i+1)*t.shape[0],0] = radii[i]*np.cos(t)
            xy[i*t.shape[0]:(i+1)*t.shape[0],1] = radii[i]*np.sin(t)


        # REMOVE DUPLICATES GENERATED BY SIN/COS OF LINSPACE
        xy = xy[np.setdiff1d( np.arange(xy.shape[0]) , np.linspace(t.shape[0]-1,xy.shape[0]-1,radii.shape[0]).astype(int) ),:]

        connec = np.zeros((1,4),dtype=np.int64)

        for j in range(1,radii.shape[0]):   
            for i in range((j-1)*(t.shape[0]-1),j*(t.shape[0]-1)):
                if i<j*(t.shape[0]-1)-1:
                    connec = np.concatenate((connec,np.array([[i,t.shape[0]-1+i,t.shape[0]+i,i+1 ]])),axis=0)
                    # connec = connec + ((i,t.shape[0]-1+i,t.shape[0]+i,i+1),)
                else:
                    connec = np.concatenate((connec,np.array([[i,t.shape[0]-1+i,j*(t.shape[0]-1),(j-1)*(t.shape[0]-1) ]])),axis=0)
                    # connec = connec + ((i,t.shape[0]-1+i,j*(t.shape[0]-1),(j-1)*(t.shape[0]-1)),)

        connec = connec[1:,:]
        # connec = np.asarray(connec[1:])


        if element_type == 'tri':
            connec_tri = np.zeros((2*connec.shape[0],3),dtype=np.int64)
            for i in range(connec.shape[0]):
                connec_tri[2*i,:] = np.array([connec[i,0],connec[i,1],connec[i,3]])
                connec_tri[2*i+1,:] = np.array([connec[i,2],connec[i,3],connec[i,1]])

            self.elements = connec_tri
            self.nelem = self.elements.shape[0]
            self.element_type = element_type
            # OBTAIN MESH EDGES
            self.GetBoundaryEdgesTri()

        elif element_type == 'quad':
            self.elements = connec
            self.nelem = self.elements.shape[0]
            self.element_type = element_type
            self.GetBoundaryEdgesQuad()

        # ASSIGN NODAL COORDINATES
        self.points = xy
        # IF CENTER IS DIFFERENT FROM (0,0)
        self.points[:,0] += center[0]
        self.points[:,1] += center[1]
        # ASSIGN PROPERTIES
        self.nnode = self.points.shape[0]


    def Parallelepiped(self,lower_left_rear_point=(0,0,0), upper_right_front_point=(2,4,10), 
        nx=2, ny=4, nz=10, element_type="tet"):
        """Creates a tet/hex mesh on rectangular parallelepiped"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        if element_type != "tet" and element_type != "hex":
            raise ValueError("Can only generate parallelepiped mesh using tetrahedrals or hexahedrals")

        if (lower_left_rear_point[0] > upper_right_front_point[0]) or \
            (lower_left_rear_point[1] > upper_right_front_point[1]) or \
            (lower_left_rear_point[2] > upper_right_front_point[2]):
            raise ValueError("Incorrect coordinate for lower left rear and upper right front vertices")


        x=np.linspace(lower_left_rear_point[0],upper_right_front_point[0],nx+1)
        y=np.linspace(lower_left_rear_point[1],upper_right_front_point[1],ny+1)
        z=np.linspace(lower_left_rear_point[2],upper_right_front_point[2],nz+1)

        Y,X,Z = np.meshgrid(y,x,z)
        coordinates = np.dstack((X.T.flatten(),Y.T.flatten(),Z.T.flatten()))[0,:,:]

        self.element_type = "hex"
        self.nelem = int(nx*ny*nz)
        elements = np.zeros((self.nelem,8),dtype=np.int64)

        dum_0 = np.arange((nx+1)*ny)
        dum_1 = np.array([(nx+1)*i+nx for i in range(ny)])
        dum_2 = np.delete(dum_0,dum_1)
        col0 = np.array([dum_2+i*(nx+1)*(ny+1) for i in range(nz)]).flatten()

        elements[:,0] = col0
        elements[:,1] = col0 + 1
        elements[:,2] = col0 +  nx + 2
        elements[:,3] = col0 +  nx + 1
        elements[:,4] = col0 + (nx + 1) * (ny + 1)
        elements[:,5] = col0 + (nx + 1) * (ny + 1) + 1
        elements[:,6] = col0 + (nx + 1) * (ny + 1) + nx + 2
        elements[:,7] = col0 + (nx + 1) * (ny + 1) + nx + 1

        self.elements = elements
        self.points = coordinates

        self.GetBoundaryFacesHex()
        self.GetBoundaryEdgesHex()

        if element_type == "tet":
            self.ConvertHexesToTets()


    def Cube(self, lower_left_rear_point=(0.,0.,0.), side_length=1, nx=5, ny=5, nz=5, n=None, element_type="tet"):
        """Creates a quad/tri mesh on a cube

            input:
                lower_left_rear_point       [tuple] of lower left rear vertex of the cube
                side_length:                [int] length of side 
                nx,ny,nz:                   [int] number of discretisation in each direction
                n:                          [int] number of discretisation in all directions
                                            i.e. nx=ny=nz=n. Overrides nx,ny,nz 
            """

        if n != None:
            nx,ny,nz = n,n,n

        upper_right_front_point = (side_length+lower_left_rear_point[0],
            side_length+lower_left_rear_point[1],side_length+lower_left_rear_point[2])
        self.Parallelepiped(lower_left_rear_point=lower_left_rear_point,
            upper_right_front_point=upper_right_front_point,nx=nx,ny=ny,nz=nz,element_type=element_type)




    def Sphere(self,radius=1.0, npoints=10):
        """Creates a tetrahedral mesh on a sphere

        input:

            radius:         [double] radius of sphere
            points:         [int] no of disrectisation
        """

        # RESET MESH
        self.__reset__()

        from math import pi, cos, sin
        from meshpy.tet import MeshInfo, build
        from meshpy.geometry import generate_surface_of_revolution, EXT_OPEN, GeometryBuilder

        r = radius

        points = npoints
        dphi = pi/points

        def truncate(r):
            if abs(r) < 1e-10:
                return 0
            else:
                return r

        rz = [(truncate(r*sin(i*dphi)), r*cos(i*dphi)) for i in range(points+1)]

        geob = GeometryBuilder()
        geob.add_geometry(*generate_surface_of_revolution(rz,
                closure=EXT_OPEN, radial_subdiv=10))

        mesh_info = MeshInfo()
        geob.set(mesh_info)

        mesh = build(mesh_info)

        self.points = np.asarray(mesh.points)
        self.elements = np.asarray(mesh.elements)
        # self.faces = np.asarray(mesh.faces)
        # self.edges = np.asarray(self.edges)
        self.nelem = self.elements.shape[0]
        self.element_type = "tet"


        # GET EDGES & FACES - NONE ASSIGNMENT IS NECESSARY OTHERWISE IF FACES/EDGES ALREADY EXIST 
        # THEY WON'T GET UPDATED
        self.faces = None
        self.edges = None
        self.GetBoundaryFacesTet()
        self.GetBoundaryEdgesTet()

        # CHECK MESH
        points = self.points[np.unique(self.faces),:]
        if not np.isclose(np.linalg.norm(points,axis=1),radius).all():
            raise ValueError("MeshPy could not construct a valid linear mesh for sphere")



    def OneElementCylinder(self,radius=1, length=100, nz=10, element_type="hex"):
        """Creates a mesh on cylinder with one hexahedral element across the cross section"""

        if element_type == "hex":
            elements = np.arange(0,8)[:,None]
            for i in range(1,nz):
                elements = np.concatenate((elements,np.arange(4*i,4*i+8)[:,None]),axis=1)
            elements = elements.T.copy()

        theta = np.array([225,315,45,135])*np.pi/180
        
        xs = np.tile(radius*np.cos(theta),nz+1)
        ys = np.tile(radius*np.sin(theta),nz+1)

        points = np.array([xs,ys]).T.copy()
        points = np.concatenate((points,np.zeros((4*(nz+1),1))),axis=1)

        zs = np.linspace(0,length, nz+1)[1:]
        zs = np.repeat(zs,4)
        points[4:,-1] = zs

        if element_type == "hex":
            self.element_type = element_type
            self.elements = elements
            self.points = points
            self.GetBoundaryFacesHex()
            self.GetBoundaryEdgesHex()
            self.GetFacesHex()
            self.GetEdgesHex()
            self.nelem = self.elements.shape[0]
            self.nnode = self.points.shape[0]

        elif element_type == "tet":
            # USE MESHPY
            raise NotImplementedError('Not implemented yet')
        else:
            raise ValueError('element type not suppported')


    def Cylinder(self, center=(0.,0.,0.), radius=1., length=10., nrad=16, ncirc=40, nlong=50, element_type="hex"):
        """Creates a structured hexahedral mesh on cylinder. The base of cylinder is always in the (X,Y)
            plane
        """

        if element_type != "hex":
            raise NotImplementedError('Generating {} mesh of cylinder is not supported yet'.format(element_type))

        if not isinstance(center,tuple):
            raise ValueError("The center for the base of the cylinder should be given in a tuple with three elements (x,y,z)")

        self.__reset__()

        nlong = int(nlong)
        if nlong==0:
            nlong = 1

        mesh = Mesh()
        mesh.Circle(center=(center[0],center[1]), radius=radius, nrad=nrad, ncirc=ncirc, element_type="quad")

        self.Extrude(base_mesh=mesh, length=length, nlong=nlong)
        self.points += center[2]



    def ArcCylinder(self, center=(0.,0.,0.), radius=1., start_angle=0, end_angle=np.pi/2., 
        length=10., nrad=16, ncirc=40, nlong=50, element_type="hex"):
        """Creates a structured hexahedral mesh on cylinder with a base made of arc. 
            The base of cylinder is always in the (X,Y) plane
        """

        if element_type != "hex":
            raise NotImplementedError('Generating {} mesh of cylinder is not supported yet'.format(element_type))

        if not isinstance(center,tuple):
            raise ValueError("The center for the base of the cylinder should be given in a tuple with three elements (x,y,z)")

        self.__reset__()

        nlong = int(nlong)
        if nlong==0:
            nlong = 1

        mesh = Mesh()
        mesh.Arc(center=(center[0],center[1]), radius=radius, start_angle=start_angle, 
            end_angle=end_angle, nrad=nrad, ncirc=ncirc, element_type="quad")
 
        self.Extrude(base_mesh=mesh, length=length, nlong=nlong)
        self.points += center[2]



    def HollowCylinder(self,center=(0,0,0),inner_radius=1.0,outer_radius=2.,element_type='hex',isotropic=True,nrad=5,ncirc=10, nlong=20,length=10):
        """Creates a hollow cylindrical mesh. Only hexes are supported for now"""

        if element_type != "hex":
            raise NotImplementedError('Generating {} mesh of cylinder is not supported yet'.format(element_type))

        if not isinstance(center,tuple):
            raise ValueError("The center for the base of the cylinder should be given in a tuple with three elements (x,y,z)")

        self.__reset__()

        nlong = int(nlong)
        if nlong==0:
            nlong = 1

        mesh = Mesh()
        mesh.HollowCircle(center=(center[0],center[1]), inner_radius=inner_radius, 
            outer_radius=outer_radius, element_type="quad", 
            isotropic=isotropic, nrad=nrad, ncirc=ncirc)
 
        self.Extrude(base_mesh=mesh, length=length, nlong=nlong)
        self.points += center[2]



    def Extrude(self, base_mesh=None, length=10, path=None, nlong=10):
        """Extrude a 2D mesh to 3D. At the moment only quad to hex extrusion is supported

            input:
                base_mesh:                  [Mesh] an instance of class Mesh to be extruded.
                                            If base_mesh is not provided (None), then self is 
                                            taken as base_mesh
                length:                     length along extrusion
                path:                       [GeometricPath] the path along which the mesh needs
                                            to be extruded. If not None overrides length
                nlong:                      [int] number of discretisation in the extrusion
                                            direction
        """

        mesh = base_mesh
        if mesh != None:
            if not isinstance(mesh,Mesh):
                raise ValueError("Base mesh has to be instance of class Florence.Mesh")
            else:
                mesh.__do_essential_memebers_exist__()
                if mesh.element_type !="quad":
                    raise NotImplementedError("Extrusion for {} mesh not supported yet".format(mesh.element_type))
        else:
            self.__do_essential_memebers_exist__()
            if self.element_type !="quad":
                raise NotImplementedError("Extrusion for {} mesh not supported yet".format(self.element_type))

            mesh = deepcopy(self)


        if mesh.IsHighOrder:
            raise NotImplementedError("Extruding high order meshes is not supported yet")


        nlong = int(nlong)
        if nlong==0:
            nlong = 1

        nnode = (nlong+1)*mesh.points.shape[0]
        nnode_2D = mesh.points.shape[0]

        self.points = np.zeros((nnode,3),dtype=np.float64)

        if path is None:

            node_aranger = np.linspace(0,length,nlong+1)

            for i in range(nlong+1):
                self.points[nnode_2D*i:nnode_2D*(i+1),:2] = mesh.points
                self.points[nnode_2D*i:nnode_2D*(i+1), 2] = node_aranger[i]

        elif isinstance(path,GeometricPath):
            points = path.ComputeExtrusion(nlong=nlong)
            for i in range(nlong+1):
                self.points[nnode_2D*i:nnode_2D*(i+1),:2] = mesh.points + points[i,:2]
                self.points[nnode_2D*i:nnode_2D*(i+1), 2] = points[i,2]


        nelem= nlong*mesh.nelem
        nelem_2D = mesh.nelem
        element_aranger = np.arange(nlong)
        self.elements = np.zeros((nelem,8),dtype=np.int64)

        for i in range(nlong):
            self.elements[nelem_2D*i:nelem_2D*(i+1),:4] = mesh.elements + i*nnode_2D
            self.elements[nelem_2D*i:nelem_2D*(i+1),4:] = mesh.elements + (i+1)*nnode_2D


        self.element_type = "hex"
        self.nelem = nelem
        self.nnode = nnode
        self.GetBoundaryFaces()
        self.GetBoundaryEdges()

        if isinstance(path,GeometricPath):
            return points




    def RemoveElements(self,(x_min,y_min,x_max,y_max),element_removal_criterion="all",keep_boundary_only=False,
            compute_edges=True,compute_faces=True,plot_new_mesh=True):
        """Removes elements with some specified criteria

        input:              
            (x_min,y_min,x_max,y_max)       [tuple of floats] box selection. Deletes all the elements apart 
                                            from the ones within this box 
            element_removal_criterion       [str]{"all","any"} the criterion for element removal with box selection. 
                                            How many nodes of the element should be within the box in order 
                                            not to be removed. Default is "all". "any" implies at least one node 
            keep_boundary_only              [bool] delete all elements apart from the boundary ones 
            compute_edges                   [bool] if True also compute new edges
            compute_faces                   [bool] if True also compute new faces (only 3D)
            plot_new_mesh                   [bool] if True also plot the new mesh

        1. Note that this method computes a new mesh without maintaining a copy of the original 
        2. Different criteria can be mixed for instance removing all elements in the mesh apart from the ones
        in the boundary which are within a box
        """

        assert self.elements != None
        assert self.points != None
        assert self.edges != None

        new_elements = np.zeros((1,3),dtype=np.int64)

        edge_elements = np.arange(self.nelem)
        if keep_boundary_only == True:
            if self.element_type == "tri":
                edge_elements = self.GetElementsWithBoundaryEdgesTri()

        for elem in edge_elements:
            xe = self.points[self.elements[elem,:],0]
            ye = self.points[self.elements[elem,:],1]

            if element_removal_criterion == "all":
                if ( (xe > x_min).all() and (ye > y_min).all() and (xe < x_max).all() and (ye < y_max).all() ):
                    new_elements = np.vstack((new_elements,self.elements[elem,:]))

            elif element_removal_criterion == "any":
                if ( (xe > x_min).any() and (ye > y_min).any() and (xe < x_max).any() and (ye < y_max).any() ):
                    new_elements = np.vstack((new_elements,self.elements[elem,:]))

        self.elements = new_elements[1:,:]
        self.nelem = self.elements.shape[0]
        unique_elements, inv_elements =  np.unique(self.elements,return_inverse=True)
        self.points = self.points[unique_elements,:]
        # RE-ORDER ELEMENT CONNECTIVITY
        remap_elements =  np.arange(self.points.shape[0])
        self.elements = remap_elements[inv_elements].reshape(self.nelem,self.elements.shape[1])

        # RECOMPUTE EDGES 
        if compute_edges == True:
            if self.element_type == "tri":
                self.GetBoundaryEdgesTri()

        # RECOMPUTE FACES 
        if compute_faces == True:
            if self.element_type == "tet":
                # FACES WILL BE COMPUTED AUTOMATICALLY
                self.GetBoundaryEdgesTet()

        # PLOT THE NEW MESH
        if plot_new_mesh == True:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.triplot(self.points[:,0],self.points[:,1],self.elements[:,:3])
            plt.axis('equal')
            plt.show()


    def MergeWith(self, mesh):
        """ Merges self with another mesh:
            NOTE: It is the responsibility of the user to ensure that meshes are conforming
        """
        
        self.__do_essential_memebers_exist__()
        mesh.__do_essential_memebers_exist__()

        if mesh.element_type != self.element_type:
            raise NotImplementedError('Merging two diffferent meshes is not possible yet')

        if mesh.elements.shape[1] != mesh.elements.shape[1]:
            warn('Elements are of not the same order. I am going to modify both meshes to their linear variant')
            self.GetLinearMesh()
            mesh.GetLinearMesh()

        tol = 1e-10
        makezero(self.points, tol=tol)
        makezero(mesh.points, tol=tol)


        from Florence.Tensor import remove_duplicates_2D, unique2d
        points = np.concatenate((self.points,mesh.points),axis=0)
        rounded_points = np.round(points,decimals=8)

        _, idx_mpoints, inv_mpoints = unique2d(rounded_points,order=False,
            consider_sort=False,return_index=True,return_inverse=True)
        mpoints = points[idx_mpoints,:]

        elements = np.concatenate((self.elements, self.elements.max()+1+mesh.elements),axis=0)
        nelem = elements.shape[0]
        nodeperelem = elements.shape[1]
        element_type = self.element_type

        unique_elements, inv_elements = np.unique(elements,return_inverse=True)
        unique_elements = unique_elements[inv_mpoints]
        melements = unique_elements[inv_elements]
        melements = melements.reshape(nelem,nodeperelem) 


        self.__reset__()
        self.element_type = element_type
        self.elements = melements
        self.nelem = melements.shape[0]
        self.points = mpoints

        ndim = self.InferSpatialDimension()
        if ndim==3:
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()
        elif ndim==2:
            self.GetBoundaryEdges()


    def Smoothing(self, criteria={'aspect_ratio':3}):
        """Performs mesh smoothing based a given criteria.
            
            input:
                criteria                [dict] criteria can be either None, {'volume':<number>}, 
                                        {'area':<number>} or {'aspect_ratio':<number>}. The
                                        number implies that all elements above that number 
                                        should be refined. Default is {'aspect_ratio':4}

            Note that this is a simple mesh smoothing, and does not perform rigorous check, in
            particular it does not guarantee mesh conformality
        """

        self.__do_essential_memebers_exist__()

        if not isinstance(criteria,dict):
            raise ValueError("Smoothing criteria should be a dictionry")

        if len(criteria.keys()) > 1:
            raise ValueError("Smoothing criteria should be a dictionry with only one key")            

        criterion = criteria.keys()[0]
        number = criteria.values()[0]

        if "aspect_ratio" in insensitive(criterion):
            quantity = self.AspectRatios()
        elif "area" in insensitive(criterion):
            quantity = self.Areas()
        elif "volume" in insensitive(criterion):
            quantity = self.Volumes()

        non_smooth_elements_idx = np.where(quantity >= number)[0]

        if non_smooth_elements_idx.shape[0]==0:
            return 

        if self.element_type == "quad":
            refiner_func = self.QuadrilateralProjection
        elif self.element_type == "tri":
            refiner_func = self.TriangularProjection
        else:
            raise ValueError("Smoothing of {} elements not supported yet".format(self.element_type))


        mesh = refiner_func(points=self.points[self.elements[non_smooth_elements_idx[0],:],:],npoints=2)
        for i in range(1,non_smooth_elements_idx.shape[0]):
            mesh += refiner_func(points=self.points[self.elements[non_smooth_elements_idx[i],:],:],npoints=2)

        smooth_elements_idx = np.where(quantity < number)[0]
        if smooth_elements_idx.shape[0]>0:
            mesh += self.GetLocalisedMesh(smooth_elements_idx)
 
        self.__update__(mesh)




    @staticmethod
    def TriangularProjection(c1=(0,0), c2=(2,0), c3=(2,2), points=None, npoints=10, equally_spaced=True):
        """Builds an instance of Mesh on a triangular region through FE interpolation
            given four vertices of the triangular region. Alternatively you can specify 
            the vertices as numpy array of 3x2. 

            This is a static immutable function, in that it does not modify self 
        """

        if points is None or not isinstance(points,np.ndarray):
            if not isinstance(c1,tuple) or not isinstance(c2,tuple) or not isinstance(c3,tuple):
                raise ValueError("coordinates should be given in tuples of two elements (x,y)")
            else:
                c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3)
                opoints = np.vstack((c1,c2,c3))
        else:
            opoints = points

        from scipy.spatial import Delaunay
        from Florence.FunctionSpace import Tri 
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTri
        from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri
        
        if equally_spaced:
            points = EquallySpacedPointsTri(npoints)
        else:
            points = FeketePointsTri(npoints)

        BasesTri = np.zeros((3,points.shape[0]),dtype=np.float64)
        hpBases = Tri.hpNodal.hpBases
        for i in range(points.shape[0]):
            BasesTri[:,i] = hpBases(0,points[i,0],points[i,1],
                EvalOpt=1,EquallySpacedPoints=equally_spaced,Transform=1)[0]

        func = Delaunay(points,qhull_options="QJ")
        triangles = func.simplices
        nnode = func.points.shape[0]
        nelem = func.nsimplex
        nsize = int((npoints+2)*(npoints+3)/2.)

        mesh = Mesh()
        mesh.element_type="tri"
        mesh.points = np.dot(BasesTri.T, opoints)
        mesh.elements = triangles
        mesh.nelem = mesh.elements.shape[0]
        mesh.nnode = mesh.points.shape[0]
        mesh.GetBoundaryEdges()

        return mesh


    @staticmethod
    def QuadrilateralProjection(c1=(0,0), c2=(2,0), c3=(2,2), c4=(0,2), points=None, npoints=10, equally_spaced=True):
        """Builds an instance of Mesh on a quadrilateral region through FE interpolation
            given four vertices of the quadrilateral region. Alternatively you can specify 
            the vertices as numpy array of 4x2. 

            This is a static immutable function, in that it does not modify self 
        """

        if points is None or not isinstance(points,np.ndarray):
            if not isinstance(c1,tuple) or not isinstance(c2,tuple) or not isinstance(c3,tuple) or not isinstance(c4,tuple):
                raise ValueError("coordinates should be given in tuples of two elements (x,y)")
            else:
                c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3); c4 = np.array(c4)
                opoints = np.vstack((c1,c2,c3,c4))
        else:
            opoints = points

        from Florence.FunctionSpace import Quad, QuadES
        from Florence.QuadratureRules.GaussLobattoPoints import GaussLobattoPointsQuad
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuad

        npoints = int(npoints)
        if npoints ==0: npoints=1

        if equally_spaced:
            points = EquallySpacedPoints(ndim=3,C=npoints-1)
            hpBases = QuadES.Lagrange
        else:
            points = GaussLobattoPointsQuad(npoints-1)
            hpBases = Quad.LagrangeGaussLobatto

        BasesQuad = np.zeros((4,points.shape[0]),dtype=np.float64)
        for i in range(points.shape[0]):
            BasesQuad[:,i] = hpBases(0,points[i,0],points[i,1],arrange=1)[:,0]

        node_arranger = NodeArrangementQuad(npoints-1)[2]

        qmesh = Mesh()
        qmesh.Square(lower_left_point=(-1.,-1.), side_length=2,n=npoints, element_type="quad")
        quads = qmesh.elements


        nnode = qmesh.nnode
        nelem = qmesh.nelem
        nsize = int((npoints+1)**2)

        mesh = Mesh()
        mesh.points = np.dot(BasesQuad.T, opoints)

        _, inv = np.unique(quads,return_inverse=True)
        sorter = np.argsort(node_arranger)
        mesh.elements = sorter[inv].reshape(quads.shape)

        mesh.element_type="quad"
        mesh.nelem = mesh.elements.shape[0]
        mesh.nnode = mesh.points.shape[0]
        mesh.GetBoundaryEdges()

        return mesh



    @staticmethod
    def HexahedralProjection(c1=(0,0,0), c2=(2,0,0), c3=(2,2,0), c4=(0,2,0.), 
        c5=(0,1.8,3.), c6=(0.2,0,3.), c7=(2,0.2,3.), c8=(1.8,2,3.),  points=None, npoints=6, equally_spaced=True):
        """Builds an instance of Mesh on a hexahedral region through FE interpolation
            given eight vertices of the quadrilateral region. Alternatively you can specify 
            the vertices as numpy array of 8x2. 

            This is a static immutable function, in that it does not modify self 
        """

        if points is None or not isinstance(points,np.ndarray):
            if not isinstance(c1,tuple) or not isinstance(c2,tuple) or not isinstance(c3,tuple) or not isinstance(c4,tuple) or \
                not isinstance(c5,tuple) or not isinstance(c6,tuple) or not isinstance(c7,tuple) or not isinstance(c8,tuple):
                raise ValueError("coordinates should be given in tuples of two elements (x,y,z)")
            else:
                c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3); c4 = np.array(c4)
                c5 = np.array(c5); c6 = np.array(c6); c7 = np.array(c7); c8 = np.array(c8)
                opoints = np.vstack((c1,c2,c3,c4,c5,c6,c7,c8))
        else:
            opoints = points

        from Florence.FunctionSpace import Hex, HexES
        from Florence.QuadratureRules.GaussLobattoPoints import GaussLobattoPointsHex
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHex

        npoints = int(npoints)
        if npoints ==0: npoints=1

        if equally_spaced:
            points = EquallySpacedPoints(ndim=4,C=npoints-1)
            hpBases = HexES.Lagrange
        else:
            points = GaussLobattoPointsHex(npoints-1)
            hpBases = Hex.LagrangeGaussLobatto

        BasesHex = np.zeros((8,points.shape[0]),dtype=np.float64)
        for i in range(points.shape[0]):
            BasesHex[:,i] = hpBases(0,points[i,0],points[i,1],points[i,2],arrange=1)[:,0]

        node_arranger = NodeArrangementHex(npoints-1)[2]

        hmesh = Mesh()
        hmesh.Cube(lower_left_rear_point=(-1.,-1.,-1.), side_length=2, n=npoints, element_type="hex")
        hexes = hmesh.elements


        nnode = hmesh.nnode
        nelem = hmesh.nelem
        nsize = int((npoints+1)**3)

        mesh = Mesh()
        mesh.points = np.dot(BasesHex.T, opoints)

        _, inv = np.unique(hexes,return_inverse=True)
        sorter = np.argsort(node_arranger)
        mesh.elements = sorter[inv].reshape(hexes.shape)

        mesh.element_type="hex"
        mesh.nelem = mesh.elements.shape[0]
        mesh.nnode = mesh.points.shape[0]
        mesh.GetBoundaryFaces()
        mesh.GetBoundaryEdges()

        return mesh




    def ChangeType(self):
        """Change mesh data type from signed to unsigned"""
        if isinstance(self.elements,np.ndarray):
            self.elements = self.elements.astype(np.uint64)
        if isinstance(self.edges,np.ndarray):
            self.edges = self.edges.astype(np.uint64)
        if isinstance(self.faces,np.ndarray):
            self.faces = self.faces.astype(np.uint64)


    def InferPolynomialDegree(self):
        """Infer the degree of interpolation (p) based on the shape of 
            self.elements

            returns:        [int] polynomial degree
            """

        assert self.element_type is not None
        assert self.elements is not None

        if self.degree is not None:
            if isinstance(self.degree,np.ndarray):
                self.degree = np.asscalar(self.degree)
            i = self.degree
            if self.element_type == "tet" and (i+1)*(i+2)*(i+3)/6==self.elements.shape[1]:
                return self.degree
            if self.element_type == "tri" and (i+1)*(i+2)/2==self.elements.shape[1]:
                return self.degree
        

        p = 0
        if self.element_type == "tet":
            for i in range(100):
                if (i+1)*(i+2)*(i+3)/6==self.elements.shape[1]:
                    p = i
                    break

        elif self.element_type == "tri":
            for i in range(100):
                if (i+1)*(i+2)/2==self.elements.shape[1]:
                    p = i
                    break

        elif self.element_type == "hex":
            for i in range(100):
                if int((i+1)**3)==self.elements.shape[1]:
                    p = i
                    break

        elif self.element_type == "quad":
            for i in range(100):
                if int((i+1)**2)==self.elements.shape[1]:
                    p = i
                    break

        self.degree = p
        return p

    @property
    def IsHighOrder(self):
        is_high_order = False
        if self.InferPolynomialDegree() > 1:
            is_high_order = True
        return is_high_order


    def InferSpatialDimension(self):
        """Infer the spatial dimension of the mesh"""

        assert self.points is not None
        if self.points.shape[1] == 3:
            if self.element_type == "tri" or self.element_type == "quad":
                print("3D surface mesh of ", self.element_type)

        return self.points.shape[1]


    def InferNumberOfNodesPerElement(self, p=None, element_type=None):
        """Infers number of nodes per element. If p and element_type are 
            not None then returns the number of nodes required for the given
            element type with the given polynomial degree"""

        if p is not None and element_type is not None:
            if element_type=="tri":
                return int((p+1)*(p+2)/2)
            if element_type=="quad":
                return int((p+1)**2)
            if element_type=="tet":
                return int((p+1)*(p+2)*(p+3)/6)
            elif element_type=="hex":
                return int((p+1)**3)
            else:
                raise ValueError("Did not understand element type")    
                
        assert self.elements.shape[0] is not None
        return self.elements.shape[1]


    def InferElementType(self):

        if self.element_type is not None:
            return self.element_type

        assert self.elements is not None
        assert self.points is not None

        p = self.InferPolynomialDegree() + 1
        ndim = InferSpatialDimension()
        nodeperelem = self.InferNumberOfNodesPerElement()

        if ndim==3:
            if nodeperelem in [int((i+1)*(i+2)*(i+3)/6) for i in range(1,50)]:
                self.element_type = "tet"
            elif nodeperelem in [int((i+1)**3) for i in range(1,50)]:
                self.element_type = "hex"
            else:
                raise ValueError("Element type not understood")
        elif ndim==2:
            if nodeperelem in [int((i+1)*(i+2)/2) for i in range(1,100)]:
                self.element_type = "tri"
            elif nodeperelem in [int((i+1)**2) for i in range(1,100)]:
                self.element_type = "quad"
            else:
                raise ValueError("Element type not understood")
        elif ndim==1:
            element_type = "line"
        else:
            raise ValueError("Element type not understood")

        return self.element_type


    def GetLinearMesh(self):
        """Returns the linear mesh from a high order mesh. If mesh is already linear returns the same mesh.
            For safety purposes, always makes a copy"""

        assert self.elements is not None
        assert self.points is not None

        ndim = self.InferSpatialDimension()
        if ndim==2:
            assert self.edges is not None
        elif ndim==3:
            assert self.faces is not None


        if self.IsHighOrder is False:
            return self
        else:
            lmesh = Mesh()
            lmesh.element_type = self.element_type
            lmesh.degree = 1
            if self.element_type == "tri":
                lmesh.elements = np.copy(self.elements[:,:3])
                lmesh.edges = np.copy(self.edges[:,:2])
                lmesh.nnode = int(np.max(lmesh.elements)+1)
                lmesh.points = np.copy(self.points[:lmesh.nnode,:])
            elif self.element_type == "tet":
                lmesh.elements = np.copy(self.elements[:,:4])
                lmesh.faces = np.copy(self.faces[:,:3])
                lmesh.nnode = int(np.max(lmesh.elements)+1)
                lmesh.points = np.copy(self.points[:lmesh.nnode,:])
            elif self.element_type == "quad":
                lmesh.elements = np.copy(self.elements[:,:4])
                lmesh.edges = np.copy(self.edges[:,:2])
                lmesh.nnode = int(np.max(lmesh.elements)+1)
                lmesh.points = np.copy(self.points[:lmesh.nnode,:])
            elif self.element_type == "hex":
                lmesh.elements = np.copy(self.elements[:,:8])
                lmesh.faces = np.copy(self.faces[:,:2])
                lmesh.nnode = int(np.max(lmesh.elements)+1)
                lmesh.points = np.copy(self.points[:lmesh.nnode,:])
            lmesh.nelem = lmesh.elements.shape[0]

        return lmesh


    def GetLocalisedMesh(self,elements, solution=None):
        """Make a new Mesh instance from part of a big mesh

            inputs:
                elements:           [int, tuple, list, 1D array] of elements in big mesh (self)
                                    from which a small localised needs to be constructed
                solution            [1D array having the same length as big mesh points] 
                                    if a solution also needs to be mapped over the localised element
        """

        elements = np.array(elements).flatten()

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        nodeperelem = self.elements.shape[1]
        tmesh = Mesh()
        tmesh.element_type = self.element_type
        tmesh.nelem = elements.shape[0]
        unnodes, inv = np.unique(self.elements[elements,:nodeperelem], return_inverse=True)
        aranger = np.arange(tmesh.nelem*nodeperelem)
        tmesh.elements = inv[aranger].reshape(tmesh.nelem,nodeperelem)
        tmesh.points = self.points[unnodes,:]
        if tmesh.element_type == "hex" or tmesh.element_type == "tet":
            tmesh.GetBoundaryFaces()
        tmesh.GetBoundaryEdges()

        if solution is not None:
            solution = solution[unnodes,:]
            return tmesh, solution

        return tmesh


    def ConvertTrisToQuads(self):
        """Converts a tri mesh to a quad mesh through refinement/splitting. 
            This is a simpler version of the the Blossom-quad algorithm implemented in gmsh"""

        self.__do_memebers_exist__()
        assert self.element_type == "tri"
        if self.IsHighOrder:
            raise ValueError('High order triangular elements cannot be converted to low/high order quads')

        tconv = time()

        # SPLIT THE TRIANGLE INTO 3 QUADS BY CONNECTING THE 
        # MEDIAN AND MIDPOINTS OF THE TRIANGLE

        # FIND MEDIAN OF TRIANGLES
        # median = self.Median()
        median = np.sum(self.points[self.elements,:],axis=1)/self.elements.shape[1]
        # FIND EDGE MIDPOINTS OF TRIANGLES
        mid0 = np.sum(self.points[self.elements[:,:2],:],axis=1)/2.
        mid1 = np.sum(self.points[self.elements[:,[1,2]],:],axis=1)/2.
        mid2 = np.sum(self.points[self.elements[:,[2,0]],:],axis=1)/2.

        # STABLE APPROACH
        # points = np.zeros((1,2))
        # for elem in range(self.nelem):
        #     quad0 = np.concatenate((self.points[self.elements[elem,0],:][None,:],mid0[elem,:][None,:],median[elem,:][None,:],mid2[elem,:][None,:]),axis=0)
        #     quad1 = np.concatenate((self.points[self.elements[elem,1],:][None,:],mid1[elem,:][None,:],median[elem,:][None,:],mid0[elem,:][None,:]),axis=0)
        #     quad2 = np.concatenate((self.points[self.elements[elem,2],:][None,:],mid2[elem,:][None,:],median[elem,:][None,:],mid1[elem,:][None,:]),axis=0)
        #     points = np.concatenate((points,quad0,quad1,quad2))
        # points = points[1:,:]

        points = np.zeros((3*self.nelem*4,2))
        points[::3*4,:] = self.points[self.elements[:,0],:]
        points[1::3*4,:] = mid0
        points[2::3*4,:] = median
        points[3::3*4,:] = mid2

        points[4::3*4,:] = self.points[self.elements[:,1],:]
        points[5::3*4,:] = mid1
        points[6::3*4,:] = median
        points[7::3*4,:] = mid0

        points[8::3*4,:] = self.points[self.elements[:,2],:]
        points[9::3*4,:] = mid2
        points[10::3*4,:] = median
        points[11::3*4,:] = mid1


        # KEEP ZEROFY ON, OTHERWISE YOU GET STRANGE BEHVAIOUR
        Decimals = 10
        rounded_points = points.copy()
        makezero(rounded_points) 
        rounded_repoints = np.round(rounded_points,decimals=Decimals)
        points, idx_points, inv_points = unique2d(rounded_points,order=False,
            consider_sort=False,return_index=True,return_inverse=True)

        elements = np.arange(points.shape[0])[inv_points].reshape(3*self.nelem,4)

        self.__reset__()

        self.element_type = "quad"
        self.elements = elements
        self.points = points
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdgesQuad()

        print "Triangular to quadrilateral mesh conversion took", time() - tconv, "seconds" 


    def ConvertTetsToHexes(self):
        """Converts a tet mesh to a hex mesh through refinement/splitting
        """

        self.__do_memebers_exist__()
        assert self.element_type == "tet"
        if self.IsHighOrder:
            raise ValueError('High order tetrahedral elements cannot be converted to low/high order hexahedrals')

        tconv = time()

        # SPLIT THE TET INTO 4 QUADS

        # FIND MEDIAN OF TETS
        # median = self.Median()
        median = np.sum(self.points[self.elements,:],axis=1)/self.elements.shape[1]
        # FIND EDGE MIDPOINTS OF TETS
        mid01 = np.sum(self.points[self.elements[:,[0,1]],:],axis=1)/2.
        mid02 = np.sum(self.points[self.elements[:,[2,0]],:],axis=1)/2.
        mid03 = np.sum(self.points[self.elements[:,[0,3]],:],axis=1)/2.
        mid12 = np.sum(self.points[self.elements[:,[1,2]],:],axis=1)/2.
        mid13 = np.sum(self.points[self.elements[:,[1,3]],:],axis=1)/2.
        mid23 = np.sum(self.points[self.elements[:,[2,3]],:],axis=1)/2.
        # FIND MEDIAN OF FACES
        med012 = np.sum(self.points[self.elements[:,[0,1,2]],:],axis=1)/3.
        med013 = np.sum(self.points[self.elements[:,[0,1,3]],:],axis=1)/3.
        med023 = np.sum(self.points[self.elements[:,[0,2,3]],:],axis=1)/3.
        med123 = np.sum(self.points[self.elements[:,[1,2,3]],:],axis=1)/3.

        # # STABLE APPROACH
        # points = np.zeros((1,3))
        # for elem in range(self.nelem):
        #     hex0 = np.concatenate((self.points[self.elements[elem,0],:][None,:], mid01[elem,:][None,:], 
        #         med012[elem,:][None,:], mid02[elem,:][None,:], 
        #         mid03[elem,:][None,:], med013[elem,:][None,:], 
        #         median[elem,:][None,:], med023[elem,:][None,:]),axis=0)

        #     hex1 = np.concatenate((self.points[self.elements[elem,1],:][None,:], mid13[elem,:][None,:], 
        #         med123[elem,:][None,:], mid12[elem,:][None,:], 
        #         mid01[elem,:][None,:], med013[elem,:][None,:],
        #         median[elem,:][None,:], med012[elem,:][None,:]),axis=0)

        #     hex2 = np.concatenate((self.points[self.elements[elem,3],:][None,:], mid23[elem,:][None,:], 
        #         med123[elem,:][None,:], mid13[elem,:][None,:], 
        #         mid03[elem,:][None,:], med023[elem,:][None,:],
        #         median[elem,:][None,:], med013[elem,:][None,:]),axis=0)

        #     hex3 = np.concatenate((self.points[self.elements[elem,2],:][None,:], mid02[elem,:][None,:], 
        #         med012[elem,:][None,:], mid12[elem,:][None,:], 
        #         mid23[elem,:][None,:], med023[elem,:][None,:], 
        #         median[elem,:][None,:],med123[elem,:][None,:]),axis=0)

        #     points = np.concatenate((points,hex0,hex1,hex2,hex3))
        # points = points[1:,:]

        points = np.zeros((4*self.nelem*8,3))
        points[0::4*8,:] = self.points[self.elements[:,0],:]
        points[1::32,:] = mid01
        points[2::32,:] = med012
        points[3::32,:] = mid02
        points[4::32,:] = mid03
        points[5::32,:] = med013
        points[6::32,:] = median
        points[7::32,:] = med023

        points[8::32,:] = self.points[self.elements[:,1],:]
        points[9::32,:] = mid13
        points[10::32,:] = med123
        points[11::32,:] = mid12
        points[12::32,:] = mid01
        points[13::32,:] = med013
        points[14::32,:] = median
        points[15::32,:] = med012

        points[16::32,:] = self.points[self.elements[:,3],:]
        points[17::32,:] = mid23
        points[18::32,:] = med123
        points[19::32,:] = mid13
        points[20::32,:] = mid03
        points[21::32,:] = med023
        points[22::32,:] = median
        points[23::32,:] = med013

        points[24::32,:] = self.points[self.elements[:,2],:]
        points[25::32,:] = mid02
        points[26::32,:] = med012
        points[27::32,:] = mid12
        points[28::32,:] = mid23
        points[29::32,:] = med023
        points[30::32,:] = median
        points[31::32,:] = med123

        # KEEP ZEROFY ON, OTHERWISE YOU GET STRANGE BEHVAIOUR
        Decimals = 10
        rounded_points = points.copy()
        makezero(rounded_points) 
        rounded_repoints = np.round(rounded_points,decimals=Decimals)
        points, inv_points = unique2d(rounded_points,order=False,
            consider_sort=False,return_inverse=True)


        elements = np.arange(points.shape[0])[inv_points].reshape(4*self.nelem,8)


        self.__reset__()

        self.element_type = "hex"
        self.elements = elements
        self.points = points
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryFacesHex()
        self.GetBoundaryEdgesHex()

        print "Tetrahedral to hexahedral mesh conversion took", time() - tconv, "seconds"





    def ConvertQuadsToTris(self):
        """Converts a quad mesh to a tri mesh through refinement/splitting

            NOTE: If only linear elements are required conversion of quads to tris
            can be done using Delauney triangularation. The following implementation
            takes care of high order elements as well
        """

        self.__do_memebers_exist__()
        assert self.element_type == "quad"

        tconv = time()

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementQuadToTri

        C = self.InferPolynomialDegree() - 1
        node_arranger = NodeArrangementQuadToTri(C)

        tris = np.concatenate((self.elements[:,node_arranger[0,:]], 
            self.elements[:,node_arranger[1,:]]),axis=0).astype(self.elements.dtype)
        
        points = self.points
        edges = self.edges

        self.__reset__()

        self.element_type = "tri"
        self.elements = tris.copy('c')
        self.points = points
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.edges = edges

        print "Quadrilateral to triangular mesh conversion took", time() - tconv, "seconds"


    def ConvertHexesToTets(self):
        """Converts a hex mesh to a tet mesh through refinement/splitting

            A hexahedron can be split into 5 or 6 tetrahedrons and there are
            many possible configuration without the edges/faces intersecting each
            other. This method splits a hex into 6 tets. 

            Note that in principle, this splitting produces non-conformal meshes
        """

        self.__do_memebers_exist__()
        assert self.element_type == "hex"

        tconv = time()

        from Florence.QuadratureRules.NodeArrangement import NodeArrangementHexToTet

        C = self.InferPolynomialDegree() - 1
        node_arranger = NodeArrangementHexToTet(C)

        tets = np.concatenate((self.elements[:,node_arranger[0,:]],
            self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],
            self.elements[:,node_arranger[3,:]],
            self.elements[:,node_arranger[4,:]],
            self.elements[:,node_arranger[5,:]]),axis=0).astype(self.elements.dtype)

       
        points = self.points
        edges = self.edges
        all_edges = self.all_edges

        self.__reset__()

        self.element_type = "tet"
        self.elements = tets.copy('c')
        self.points = points.copy('c')
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryFacesTet()
        if edges is not None:
            self.edges = edges.copy('c')
        if all_edges is not None:
            self.all_edges = all_edges

        print "Hexahedral to tetrahedral mesh conversion took", time() - tconv, "seconds"


    @staticmethod
    def BoundaryEdgesfromPhysicalParametrisation(points, facets, mesh_points, mesh_edges):
        """Given a 2D planar mesh (mesh_points,mesh_edges) and the parametrisation of the physical geometry (points and facets)
            finds boundary edges

        input:

            points:         nx2 numpy array of points given to meshpy/distmesh/etc API for the actual geometry
            facets:         mx2 numpy array connectivity of points given to meshpy/distmesh/etc API for the actual geometry
            mesh_points:    px2 numpy array of points taken from the mesh generator
            mesh_edges:     qx2 numpy array of edges taken from the mesh generator

        returns:

            arr_1:          nx2 numpy array of boundary edges

        Note that this method could be useful for finding arbitrary edges not necessarily lying on the boundary"""

        # COMPUTE SLOPE OF THE GEOMETRY EDGE
        geo_edges = np.array(facets); geo_points = np.array(points)
        # ALLOCATE
        mesh_boundary_edges = np.zeros((1,2),dtype=int)
        # LOOP OVER GEOMETRY EDGES
        for i in range(geo_edges.shape[0]):
            # GET COORDINATES OF BOTH NODES AT THE EDGE
            geo_edge_coord = geo_points[geo_edges[i]]
            geo_x1 = geo_edge_coord[0,0];       geo_y1 = geo_edge_coord[0,1]
            geo_x2 = geo_edge_coord[1,0];       geo_y2 = geo_edge_coord[1,1]

            # COMPUTE SLOPE OF THIS LINE
            geo_angle = np.arctan((geo_y2-geo_y1)/(geo_x2-geo_x1)) #*180/np.pi

            # NOW FOR EACH OF THESE GEOMETRY LINES LOOP OVER ALL MESH EDGES
            for j in range(0,mesh_edges.shape[0]):
                mesh_edge_coord = mesh_points[mesh_edges[j]]
                mesh_x1 = mesh_edge_coord[0,0];         mesh_y1 = mesh_edge_coord[0,1]
                mesh_x2 = mesh_edge_coord[1,0];         mesh_y2 = mesh_edge_coord[1,1]

                # FIND SLOPE OF THIS LINE 
                mesh_angle = np.arctan((mesh_y2-mesh_y1)/(mesh_x2-mesh_x1))

                # CHECK IF GEOMETRY AND MESH EDGES ARE PARALLEL
                if np.allclose(geo_angle,mesh_angle,atol=1e-12):
                    # IF SO THEN FIND THE NORMAL DISTANCE BETWEEN THEM
                    P1 = np.array([geo_x1,geo_y1,0]);               P2 = np.array([geo_x2,geo_y2,0])        # 1st line's coordinates
                    P3 = np.array([mesh_x1,mesh_y1,0]);             P4 = np.array([mesh_x2,mesh_y2,0])      # 2nd line's coordinates

                    dist = NormalDistance(P1,P2,P3,P4)
                    # IF NORMAL DISTANCE IS ZEROS THEN MESH EDGE IS ON THE BOUNDARY
                    if np.allclose(dist,0,atol=1e-14):
                        mesh_boundary_edges = np.append(mesh_boundary_edges,mesh_edges[j].reshape(1,2),axis=0)

        # return np.delete(mesh_boundary_edges,0,0)
        return mesh_boundary_edges[1:,:]


    def __add__(self, other):
        """Add self with other. Hybrid meshes not supported"""
        self.MergeWith(other)
        return self

    def __iadd__(self, other):
        """Add self with other. Hybrid meshes not supported"""
        self.MergeWith(other)
        return self


    def __do_memebers_exist__(self):
        """Check if fundamental members exist"""
        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None
        assert self.edges is not None
        ndim = self.InferSpatialDimension()
        if ndim==3:
            assert self.faces is not None


    def __do_essential_memebers_exist__(self):
        """Check if essential members exist"""
        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None


    def __update__(self,other):
        self.__dict__.update(other.__dict__)


    def __reset__(self):
        """Class resetter. Resets all elements of the class
        """

        for i in self.__dict__.keys():
            self.__dict__[i] = None