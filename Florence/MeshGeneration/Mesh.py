from __future__ import print_function, division
import os, sys, warnings, platform
from time import time
import numpy as np
if "PyPy" not in platform.python_implementation():
    from scipy.io import loadmat, savemat
from Florence.Tensor import makezero, itemfreq, unique2d, in2d
from Florence.Utils import insensitive
from .vtk_writer import write_vtu
try:
    import meshpy.triangle as triangle
    has_meshpy = True
except ImportError:
    has_meshpy = False
from .HigherOrderMeshing import *
from .NodeArrangement import *
from .GeometricPath import *
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
    8. Writing meshes to unstructured vtk file format (.vtu) in xml and binary formats,
        including high order elements
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
        self.corners = None
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
        # TYPE OF BOUNDARY FACES/EDGES
        self.boundary_element_type = None

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
            return bounds
        elif self.points.shape[1] == 1:
            bounds = np.array([[np.min(self.points[:,0])],
                        [np.max(self.points[:,0])]])
            makezero(bounds)
            return bounds
        else:
            raise ValueError("Invalid dimension for mesh coordinates")


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

        if not isinstance(p,int):
            raise ValueError("p must be an integer")
        else:
            if p < 1:
                raise ValueError("Value of p={} is not acceptable. Provide p>=1.".format(p))

        if self.degree is None:
            self.InferPolynomialDegree()

        C = p-1
        if 'C' in kwargs.keys():
            if kwargs['C'] != p - 1:
                raise ValueError("Did not understand the specified interpolation degree of the mesh")
            del kwargs['C']

        # DO NOT COMPUTE IF ALREADY COMPUTED FOR THE SAME ORDER
        if self.degree == None:
            self.degree = self.InferPolynomialDegree()
        if self.degree == p:
            return

        # SITUATIONS WHEN ANOTHER HIGH ORDER MESH IS REQUIRED, WITH ONE HIGH
        # ORDER MESH ALREADY AVAILABLE
        if self.degree != 1 and self.degree - 1 != C:
            dum = self.GetLinearMesh(remap=True)
            self.__dict__.update(dum.__dict__)

        print('Generating p = '+str(C+1)+' mesh based on the linear mesh...')
        t_mesh = time()
        # BUILD A NEW MESH BASED ON THE LINEAR MESH
        if self.element_type == 'line':
            nmesh = HighOrderMeshLine(C,self,**kwargs)
        if self.element_type == 'tri':
            if self.edges is None:
                self.GetBoundaryEdgesTri()
            # nmesh = HighOrderMeshTri(C,self,**kwargs)
            nmesh = HighOrderMeshTri_SEMISTABLE(C,self,**kwargs)
        elif self.element_type == 'tet':
            # nmesh = HighOrderMeshTet(C,self,**kwargs)
            nmesh = HighOrderMeshTet_SEMISTABLE(C,self,**kwargs)
        elif self.element_type == 'quad':
            if self.edges is None:
                self.GetBoundaryEdgesTri()
            nmesh = HighOrderMeshQuad(C,self,**kwargs)
        elif self.element_type == 'hex':
            nmesh = HighOrderMeshHex(C,self,**kwargs)

        self.points = nmesh.points
        self.elements = nmesh.elements.astype(np.uint64)
        if isinstance(self.corners,np.ndarray):
            # NOT NECESSARY BUT GENERIC
            self.corners = nmesh.corners.astype(np.uint64)
        if isinstance(self.edges,np.ndarray):
            self.edges = nmesh.edges.astype(np.uint64)
        if isinstance(self.faces,np.ndarray):
            if isinstance(nmesh.faces,np.ndarray):
                self.faces = nmesh.faces.astype(np.uint64)
        self.nelem = nmesh.nelem
        self.nnode = self.points.shape[0]
        self.element_type = nmesh.info
        self.degree = C+1

        self.ChangeType()

        print('Finished generating the high order mesh. Time taken', time()-t_mesh,'sec')


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


    def Lengths(self,):
        """Computes length of all types of elements
        """

        self.__do_essential_memebers_exist__()

        if self.element_type == "line":
            coords = self.points[self.elements[:,:2],:]
            lengths = np.linalg.norm(coords[:,1,:] - coords[:,0,:],axis=1)
        else:
            self.GetEdges()
            coord = self.all_edges
            coords = self.points[self.elements[:,:2],:]
            lengths = np.linalg.norm(coords[:,1,:] - coords[:,0,:],axis=1)

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
            area1 = np.linalg.det(points[self.elements[:,[0,2,3]],:])
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
            raise ValueError("2D mesh does not have volume")
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


    def Sizes(self):
        """Computes the size of elements for all element types.
            This is a generic method that for 1D=lengths, for 2D=areas and for 3D=volumes.
            It works for planar and curved elements
        """

        self.__do_essential_memebers_exist__()

        try:
            from Florence import DisplacementFormulation
        except ImportError:
            raise ValueError("This functionality requires Florence's support")

        if self.element_type != "line":
            # FOR LINE ELEMENTS THIS APPROACH DOES NOT WORK AS JACOBIAN IS NOT WELL DEFINED
            formulation = DisplacementFormulation(self)
            sizes = np.zeros(self.nelem)
            for elem in range(self.nelem):
                LagrangeElemCoords = self.points[self.elements[elem,:],:]
                sizes[elem] = formulation.GetVolume(formulation.function_spaces[0],
                    LagrangeElemCoords, LagrangeElemCoords, False, elem=elem)
            return sizes

        else:
            warn("Sizes of line elements could be incorrect if the mesh is curvilinear")
            return self.Lengths()


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

            elif self.element_type == "line":
                raise ValueError("Line elments do no have aspect ratio")

        elif algorithm == 'face_based':
            raise NotImplementedError("Face/area based aspect ratio is not implemented yet")

        return aspect_ratio


    def FaceNormals(self):
        """Computes outward unit normals on faces.
            This is a generic method for all element types apart from lines. If the mesh is in 2D plane
            then the unit outward normals will point in Z direction. If the mesh is quad or tri type but
            in 3D plane, this will still compute the correct unit outward normals. outwardness can only
            be guaranteed for volume meshes.
            This method is different from the method self.Normals() as the latter can compute normals
            for 1D/2D elements in-plane
        """

        self.__do_memebers_exist__()

        points = np.copy(self.points)
        if points.shape[1] < 3:
            dum = np.zeros((points.shape[0],3))
            dum[:,:points.shape[1]] = points
            points = dum

        if self.element_type == "tet" or self.element_type == "hex":
            faces = self.faces
        elif self.element_type == "tri" or self.element_type == "quad":
            faces = self.elements
        else:
            raise ValueError("Cannot compute face normals on {}".format(self.element_type))


        face_coords = self.points[faces[:,:3],:]

        p1p0 = face_coords[:,1,:] - face_coords[:,0,:]
        p2p0 = face_coords[:,2,:] - face_coords[:,0,:]

        normals = np.cross(p1p0,p2p0)
        norm_normals = np.linalg.norm(normals,axis=1)
        normals[:,0] /= norm_normals
        normals[:,1] /= norm_normals
        normals[:,2] /= norm_normals

        # CHECK IF THE NORMAL IS OUTWARD - FOR LINES DIRECTIONALITY DOES NOT MATTER
        if self.element_type == "tet" or self.element_type == "hex":
            self.GetElementsWithBoundaryFaces()
            meds = self.Medians()
            face_element_meds = meds[self.boundary_face_to_element[:,0],:]
            p1pm = face_coords[:,1,:] - face_element_meds
            # IF THE DOT PROUCT OF NORMALS AND EDGE-MED NODE VECTOR IS NEGATIVE THEN FLIP
            _check = np.einsum("ij,ij->i",normals,p1pm)
            normals[np.less(_check,0.)] = -normals[np.less(_check,0.)]

        return normals



    def Normals(self, show_plot=False):
        """Computes unit outward normals to the boundary for all element types.
            Unity and outwardness are guaranteed
        """

        self.__do_memebers_exist__()
        ndim = self.InferSpatialDimension()

        if self.element_type == "tet" or self.element_type == "hex":
            normals = self.FaceNormals()
        elif self.element_type == "tri" or self.element_type == "quad" or self.element_type == "line":
            if self.points.shape[1] == 3:
                normals = self.FaceNormals()
            else:
                if self.element_type == "tri" or self.element_type == "quad":
                    edges = self.edges
                elif self.element_type == "line":
                    edges = self.elements

                edge_coords = self.points[edges[:,:2],:]
                p1p0 = edge_coords[:,1,:] - edge_coords[:,0,:]

                normals = np.zeros_like(p1p0)
                normals[:,0] = -p1p0[:,1]
                normals[:,1] =  p1p0[:,0]
                norm_normals = np.linalg.norm(normals,axis=1)
                normals[:,0] /= norm_normals
                normals[:,1] /= norm_normals

                # CHECK IF THE NORMAL IS OUTWARD - FOR LINES DIRECTIONALITY DOES NOT MATTER
                if self.element_type == "tri" or self.element_type == "quad":
                    self.GetElementsWithBoundaryEdges()
                    meds = self.Medians()
                    edge_element_meds = meds[self.boundary_edge_to_element[:,0],:]
                    p1pm = edge_coords[:,1,:] - edge_element_meds
                    # IF THE DOT PROUCT OF NORMALS AND EDGE-MED NODE VECTOR IS NEGATIVE THEN FLIP
                    _check = np.einsum("ij,ij->i",normals,p1pm)
                    normals[np.less(_check,0.)] = -normals[np.less(_check,0.)]


        if show_plot:

            if ndim == 2:
                mid_edge_coords = 0.5*(edge_coords[:,1,:] + edge_coords[:,0,:])

                import matplotlib.pyplot as plt
                figure = plt.figure()

                self.SimplePlot(figure=figure, show_plot=False)

                q = plt.quiver(mid_edge_coords[:,0], mid_edge_coords[:,1],
                    normals[:,0], normals[:,1],
                    color='Teal', headlength=5, width=0.004)

                plt.axis('equal')
                plt.axis('off')
                plt.tight_layout()
                plt.show()


            elif ndim == 3:
                mid_face_coords = np.sum(self.points[self.faces,:3],axis=1)/self.faces.shape[1]

                import os
                os.environ['ETS_TOOLKIT'] = 'qt4'
                from mayavi import mlab

                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

                self.SimplePlot(figure=figure, show_plot=False)

                mlab.quiver3d(mid_face_coords[:,0], mid_face_coords[:,1], mid_face_coords[:,2],
                    normals[:,0], normals[:,1], normals[:,2],
                    color=(0.,128./255,128./255),line_width=2)

                mlab.show()

        return normals



    def Medians(self, geometric=True):
        """Computes median of the elements tri, tet, quad, hex based on the interpolation function

            input:
                geometric           [Bool] geometrically computes median without relying on FEM bases
            retruns:
                median:             [ndarray] of median of elements
                bases_at_median:    [1D array] of (p=1) bases at median
        """

        self.__do_essential_memebers_exist__()

        median = None
        if geometric == True:
            median = np.sum(self.points[self.elements,:],axis=1)/self.elements.shape[1]
            return median

        else:
            try:
                from Florence.FunctionSpace import Tri, Tet
                from Florence.QuadratureRules import FeketePointsTri, FeketePointsTet
            except ImportError:
                raise ImportError("This functionality requires florence's support")

            if self.element_type == "tri":

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


    def FindElementContainingPoint(self, point, algorithm="fem", find_parametric_coordinate=True,
        scaling_factor=5., tolerance=1.0e-7, maxiter=20, use_simple_bases=False, return_on_geometric_finds=False):
        """Find which element does a point lie in using specificed algorithm.
            The FEM isoparametric coordinate of the point is returned as well.
            If the isoparametric coordinate of the point is not required, issue find_parametric_coordinate=False

            input:
                point:                  [tuple] XYZ of enquiry point
                algorithm:              [str] either 'fem' or 'geometric'. The 'fem' algorithm uses k-d tree
                                        search to get the right bounding box around as few elements as possible.
                                        The size of the box can be specified by the user through the keyword scaling_factor.
                                        The geometric algorithm is a lot more stable and converges much quicker.
                                        The geomtric algorithm first identifies the right element using volume check,
                                        then tries all possible combination of initial guesses to get the FEM
                                        isoparametric point. Trying all possible combination with FEM can be potentially
                                        more costly since bounding box size can be large.

                return_on_geometric_finds:
                                        [bool] if geometric algorithm is chosen and this option is on, then it returns
                                        the indices of elements as soon as the volume check and no further checks are
                                        done. This is useful for situations when searching for points that are meant to
                                        be in the interior of the elements rather than at the boundaries or nodes
                                        otherwise the number of elements returned by geometric algorithm is going to be
                                        more than one
            return:
                element_index           [int/1D array of ints] element(s) containing the point.
                                        If the point is shared between many elements a 1D array is returned
                iso_parametric_point    [1D array] the parametric coordinate of the point within the element.
                                        return only if find_parametric_coordinate=True
        """

        self.__do_essential_memebers_exist__()
        C = self.InferPolynomialDegree() - 1
        if C > 0:
            warn("Note that finding a point within higher order curved mesh is not supported yet")
        if C > 0 and algorithm == "geometric":
            warn("High order meshes are not supported using geometric algorithim. I am going to operate on linear mesh")
            if use_simple_bases:
                raise ValueError("Simple bases for high order elements are not available")
                return

        ndim = self.InferSpatialDimension()
        assert len(point) == ndim

        from Florence.FunctionSpace import PointInversionIsoparametricFEM
        candidate_element, candidate_piso = None, None

        if self.element_type == "tet" and algorithm == "fem":
            algorithm = "geometric"

        if algorithm == "fem":
            scaling_factor = float(scaling_factor)
            max_h = self.EdgeLengths().max()
            # FOR CURVED ELEMENTS
            # max_h = self.LargestSegment().max()
            # GET A BOUNDING BOX AROUND THE POINT, n TIMES LARGER THAN MAXIMUM h, WHERE n is the SCALING FACTOR
            if ndim==3:
                bounding_box = (point[0]-scaling_factor*max_h,
                                point[1]-scaling_factor*max_h,
                                point[2]-scaling_factor*max_h,
                                point[0]+scaling_factor*max_h,
                                point[1]+scaling_factor*max_h,
                                point[2]+scaling_factor*max_h)
            elif ndim==2:
                bounding_box = (point[0]-scaling_factor*max_h,
                                point[1]-scaling_factor*max_h,
                                point[0]+scaling_factor*max_h,
                                point[1]+scaling_factor*max_h)
            # SELECT ELEMENTS ONLY WITHIN THE BOUNDING BOX
            mesh = deepcopy(self)
            idx_kept_element = self.RemoveElements(bounding_box)[1]

            if ndim==3:
                for i in range(self.nelem):
                    coord = self.points[self.elements[i,:],:]
                    p_iso, converged = PointInversionIsoparametricFEM(self.element_type, C, coord, point,
                        tolerance=tolerance, maxiter=maxiter, verbose=True)

                    if converged:
                        # if p_iso[0] >= -1. and p_iso[0] <=1. and \
                        #     p_iso[1] >= -1. and p_iso[1] <=1. and \
                        #         p_iso[2] >= -1. and p_iso[2] <=1. :

                        if  (p_iso[0] > -1. or np.isclose(p_iso[0],-1.,rtol=tolerance)) and \
                            (p_iso[0] < 1.  or np.isclose(p_iso[0], 1.,rtol=tolerance)) and \
                            (p_iso[1] > -1. or np.isclose(p_iso[1],-1.,rtol=tolerance)) and \
                            (p_iso[1] < 1.  or np.isclose(p_iso[1],-1.,rtol=tolerance)) and \
                            (p_iso[2] > -1. or np.isclose(p_iso[2],-1.,rtol=tolerance)) and \
                            (p_iso[2] < 1.  or np.isclose(p_iso[2], 1.,rtol=tolerance)) :
                            candidate_element, candidate_piso = i, p_iso
                            break
            elif ndim==2:
                for i in range(self.nelem):
                    coord = self.points[self.elements[i,:],:]
                    p_iso, converged = PointInversionIsoparametricFEM(self.element_type, C, coord, point,
                        tolerance=tolerance, maxiter=maxiter, verbose=True)
                    # if p_iso[0] >= -1. and p_iso[0] <=1. and \
                    #     p_iso[1] >= -1. and p_iso[1] <=1.:
                    #     candidate_element, candidate_piso = i, p_iso
                    #     break
                    if  (p_iso[0] > -1. or np.isclose(p_iso[0],-1.,rtol=tolerance)) and \
                        (p_iso[0] < 1.  or np.isclose(p_iso[0], 1.,rtol=tolerance)) and \
                        (p_iso[1] > -1. or np.isclose(p_iso[1],-1.,rtol=tolerance)) and \
                        (p_iso[1] < 1.  or np.isclose(p_iso[1],-1.,rtol=tolerance)) :
                        candidate_element, candidate_piso = i, p_iso
                        break

            # if candidate_element is None:
                # raise RuntimeError("Could not find element containing the point")

            self.__update__(mesh)
            # print(candidate_element)
            if candidate_element is not None:
                candidate_element = idx_kept_element[candidate_element]

            if find_parametric_coordinate:
                return candidate_element, candidate_piso
            else:
                return candidate_element

        else:
            if self.element_type == "tet":

                from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet
                initial_guesses = FeketePointsTet(C)

                def GetVolTet(a0,b0,c0,d0):
                    det_array = np.dstack((a0-d0,b0-d0,c0-d0))
                    # FIND VOLUME OF ALL THE ELEMENTS
                    volume = 1./6.*np.abs(np.linalg.det(det_array))
                    return volume

                a = self.points[self.elements[:,0],:]
                b = self.points[self.elements[:,1],:]
                c = self.points[self.elements[:,2],:]
                d = self.points[self.elements[:,3],:]
                o = np.tile(point,self.nelem).reshape(self.nelem,a.shape[1])

                # TOTAL VOLUME
                vol = self.Volumes()
                # PARTS' VOLUMES
                vol0 = GetVolTet(a,b,c,o)
                vol1 = GetVolTet(a,b,o,d)
                vol2 = GetVolTet(a,o,c,d)
                vol3 = GetVolTet(o,b,c,d)

                criterion_check = vol0+vol1+vol2+vol3-vol
                elems = np.isclose(criterion_check,0.,rtol=tolerance)
                elems_idx = np.where(elems==True)[0]

            elif self.element_type == "quad":

                from Florence.QuadratureRules.GaussLobattoPoints import GaussLobattoPointsQuad
                initial_guesses = GaussLobattoPointsQuad(C)

                def GetAreaQuad(a0,b0,c0,d0):
                    # AREA OF QUAD ABCD = AREA OF ABC + AREA OF ACD
                    a00 = np.ones((a0.shape[0],3),dtype=np.float64); a00[:,:2] = a0
                    b00 = np.ones((b0.shape[0],3),dtype=np.float64); b00[:,:2] = b0
                    c00 = np.ones((c0.shape[0],3),dtype=np.float64); c00[:,:2] = c0
                    d00 = np.ones((d0.shape[0],3),dtype=np.float64); d00[:,:2] = d0

                    # FIND AREAS ABC
                    area0 = np.abs(np.linalg.det(np.dstack((a00,b00,c00))))
                    # FIND AREAS ACD
                    area1 = np.abs(np.linalg.det(np.dstack((a00,c00,d00))))
                    # FIND AREAS OF ALL THE ELEMENTS
                    area = 0.5*(area0+area1)

                    return area

                a = self.points[self.elements[:,0],:]
                b = self.points[self.elements[:,1],:]
                c = self.points[self.elements[:,2],:]
                d = self.points[self.elements[:,3],:]
                o = np.tile(point,self.nelem).reshape(self.nelem,a.shape[1])

                # TOTAL VOLUME
                vol = self.Areas()
                # PARTS' VOLUMES - DONT CHANGE THE ORDERING OF SPECIALLY vol1
                vol0 = GetAreaQuad(o,c,b,a)
                vol1 = GetAreaQuad(o,a,d,c)

                criterion_check = vol0+vol1-vol
                elems = np.isclose(criterion_check,0.,rtol=tolerance)
                elems_idx = np.where(elems==True)[0]

            else:
                raise NotImplementedError("Not implemented yet")

            if return_on_geometric_finds:
                return elems_idx

            for i in range(len(elems_idx)):
                coord = self.points[self.elements[elems_idx[i],:],:]
                # TRY ALL POSSIBLE INITIAL GUESSES - THIS IS CHEAP AS THE SEARCH SPACE CONTAINS ONLY A
                # FEW ELEMENTS
                for guess in initial_guesses:
                    p_iso, converged = PointInversionIsoparametricFEM(self.element_type, C, coord, point,
                        tolerance=tolerance, maxiter=maxiter, verbose=True,
                        use_simple_bases=use_simple_bases, initial_guess=guess)
                    if converged:
                        break

                if converged:
                    candidate_element, candidate_piso = elems_idx[i], p_iso
                    break

            if find_parametric_coordinate:
                return candidate_element, candidate_piso
            else:
                return candidate_element



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
                mlab.triangular_mesh(tmesh.points[:,0],tmesh.points[:,1],tmesh.points[:,2],
                    tmesh.elements, representation='wireframe', color=(0,0,0))
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
            print(u'\u2713'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering')
        else:
            print(u'\u2717'.encode('utf8')+' : ','Imported mesh has',original_order,'node ordering')

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
            At most an element can have all its three edges on the boundary.

        output:

            edge_elements:              [2D array] array containing elements which have edge
                                        on the boundary [cloumn 0] and a flag stating which edges they are [column 1]

        """

        if isinstance(self.boundary_edge_to_element,np.ndarray):
            if self.boundary_edge_to_element.shape[1] > 1 and self.boundary_edge_to_element.shape[0] > 1:
                return self.boundary_edge_to_element

        # DO NOT COMPUTE EDGES AND RAISE BECAUSE OF CYCLIC DEPENDENCIES
        assert self.elements is not None
        assert self.edges is not None

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
            At most a tetrahedral element can have all its four faces on the boundary.

        output:

            boundary_face_to_element:   [2D array] array containing elements which have face
                                        on the boundary [column 0] and a flag stating which faces they are [column 1]

        """

        # DO NOT COMPUTE FACES AND RAISE BECAUSE OF CYCLIC DEPENDENCIES
        assert self.elements is not None
        assert self.faces is not None

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

        # DO NOT COMPUTE EDGES AND RAISE BECAUSE OF CYCLIC DEPENDENCIES
        assert self.elements is not None
        assert self.edges is not None

        p = self.InferPolynomialDegree()

        # FIND WHICH FACE NODES ARE IN WHICH ELEMENT
        node_arranger = NodeArrangementQuad(p-1)[0]

        # GET ALL EDGES FROM THE ELEMENT CONNECTIVITY
        all_edges = np.concatenate((self.elements[:,node_arranger[0,:]],self.elements[:,node_arranger[1,:]],
            self.elements[:,node_arranger[2,:]],self.elements[:,node_arranger[3,:]]),axis=0).astype(self.edges.dtype)

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
                                        on the boundary [column 0] and a flag stating which faces they are [column 1]

        """

        # DO NOT COMPUTE FACES AND RAISE BECAUSE OF CYCLIC DEPENDENCIES
        assert self.elements is not None
        assert self.faces is not None

        if self.boundary_face_to_element is not None:
            return self.boundary_face_to_element

        # THIS METHOD ALWAYS RETURNS THE FACE TO ELEMENT ARRAY, AND DOES NOT CHECK
        # IF THIS HAS BEEN COMPUTED BEFORE, THE REASON BEING THAT THE FACES CAN COME
        # EXTERNALLY WHOSE ARRANGEMENT WOULD NOT CORRESPOND TO THE ONE USED INTERNALLY
        # HENCE THIS MAPPING BECOMES NECESSARY

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

        if self.all_faces is None:
            self.all_faces = self.GetFacesHex()
        if self.face_to_element is None:
            self.GetElementsFaceNumberingHex()

        # DETERMINE DEGREE
        p = self.InferPolynomialDegree()

        node_arranger = NodeArrangementHex(p-1)[0]

        self.all_faces = self.elements[self.face_to_element[:,0][:,None],node_arranger[self.face_to_element[:,1],:]]



    def GetNodeCommonality(self):
        """Finds the elements sharing a node.
            The return values are linked lists [list of numpy of arrays].
            Each numpy array within the list gives the elements that contain a given node.
            As a result the size of the linked list is nnode

            outputs:
                els:                        [list of numpy arrays] element numbers containing nodes
                pos:                        [list of numpy arrays] elemental positions of the nodes
                res_flat:                   [list of numpy arrays] position of nodes in the
                                            flattened element connectivity.
        """

        self.__do_essential_memebers_exist__()

        elements = self.elements.flatten()
        idx_sort = np.argsort(elements)
        sorted_elements = elements[idx_sort]
        # vals, idx_start, count = np.unique(sorted_elements, return_counts=True, return_index=True)
        vals, idx_start = np.unique(sorted_elements, return_index=True)

        # Sets of indices
        flat_pos = np.split(idx_sort, idx_start[1:])
        els = np.split(idx_sort // int(self.elements.shape[1]), idx_start[1:])
        pos = np.split(idx_sort %  int(self.elements.shape[1]), idx_start[1:])

        # In case one wants to return only the duplicates i.e. filter keeping only items occurring more than once
        # vals = vals[count > 1]
        # res = filter(lambda x: x.size > 1, res)

        return els, pos, flat_pos



    def Read(self, filename=None, element_type="tri", reader_type=None, reader_type_format=None,
        reader_type_version=None, order=0, read_surface_info=False, **kwargs):
        """Convenience mesh reader method to dispatch call to subsequent apporpriate methods"""

        if not isinstance(filename,str):
            raise ValueError("filename must be a string")
            return
        if reader_type is not None:
            if not isinstance(filename,str):
                raise ValueError("filename must be a string")
                return

        if reader_type is None:
            if filename.split('.')[-1] == "msh":
                reader_type = "gmsh"
            elif filename.split('.')[-1] == "obj":
                reader_type = "obj"
            elif filename.split('.')[-1] == "fro":
                reader_type = "fro"
            elif filename.split('.')[-1] == "dat":
                for key in kwargs.keys():
                    inkey = insensitive(key)
                    if "connectivity" in inkey and "delimiter" not in inkey:
                        reader_type = "read_separate"
                        break
            if reader_type is None:
                raise ValueError("Mesh file format was not undertood. Please specify it using reader_type keyword")


        self.filename = filename
        self.reader_type = reader_type
        self.reader_type_format = reader_type_format
        self.reader_type_version = reader_type_version

        if self.reader_type is 'salome':
            self.ReadSalome(filename, element_type=element_type, read_surface_info=read_surface_info)
        elif reader_type is 'GID':
            self.ReadGIDMesh(filename, element_type, order)
        elif self.reader_type is 'gmsh':
            self.ReadGmsh(filename, element_type=element_type, read_surface_info=read_surface_info)
        elif self.reader_type is 'obj':
            self.ReadOBJ(filename, element_type=element_type, read_surface_info=read_surface_info)
        elif self.reader_type is 'fro':
            self.ReadFRO(filename, element_type)
        elif self.reader_type is 'read_separate':
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
        elif self.reader_type is 'ReadHDF5':
            self.ReadHDF5(filename)

        self.nnode = self.points.shape[0]
        # MAKE SURE MESH DATA IS CONTIGUOUS
        self.points   = np.ascontiguousarray(self.points)
        self.elements = np.ascontiguousarray(self.elements)
        return


    def ReadSalome(self, filename, element_type="tri", read_surface_info=False):
        """Salome .dat format mesh reader"""

        if element_type == "line":
            el = "102"
            bel = ""
        elif element_type == "tri":
            el = "203"
            bel = "102"
        elif element_type == "quad":
            el = "204"
            bel = "102"
        elif element_type == "tet":
            el = "304"
            bel = "203"
        elif element_type == "hex":
            el = "308"
            bel = "204"

        if read_surface_info is True and element_type == "line":
            warn("No surface info for lines. I am going to ignore this")
            read_surface_info = False


        with open(filename,'r') as f:
            lines = f.readlines()

        info = lines[0].rstrip().split()

        self.nnode = int(info[0])
        all_nelem  = int(info[1])

        nodes = lines[1:self.nnode+1]

        points = []
        for line in nodes:
            points.append([float(i) for i in line.rstrip().split()[1:4]])
        self.points = np.array(points,copy=True)
        self.nnode = self.points.shape[0]

        edges, faces, elements = [], [], []
        for counter in range(self.nnode+1,len(lines)):
            line = lines[counter].rstrip().split()
            if read_surface_info:
                if bel == line[1]:
                    faces.append([int(i) for i in line[2:]])
            if el == line[1]:
                elements.append([int(i) for i in line[2:]])

        self.element_type = element_type
        self.elements = np.array(elements,dtype=np.int64,copy=True) - 1
        self.nelem = self.elements.shape[0]
        if self.nelem == 0:
            raise ValueError("file does not contain {} elements".format(element_type))

        ndim = self.InferSpatialDimension()
        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()



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



    def ReadGmsh(self, filename, element_type, read_surface_info=False):
        """Read gmsh (.msh) file"""

        try:
            fid = open(filename, "r")
        except IOError:
            print("File '%s' not found." % (filename))
            sys.exit()

        if self.elements is not None and self.points is not None:
            self.__reset__()

        self.filename = filename

        bel = -1
        if element_type == "line":
            el = 1
        elif element_type == "tri":
            el = 2
            bel = 2
        elif element_type == "quad":
            el = 3
            bel = 3
        elif element_type == "tet":
            el = 4
            bel = 2
        elif element_type == "hex":
            el = 5
            bel = 3
        else:
            raise ValueError("Element type not understood")

        # NEW FAST READER
        var = 0 # for old gmsh versions - needs checks
        rem_nnode, rem_nelem, rem_faces = int(1e09), int(1e09), int(1e09)
        face_counter = 0
        for line_counter, line in enumerate(open(filename)):
            item = line.rstrip()
            plist = item.split()
            if plist[0] == "Dimension":
                self.ndim = plist[1]
            elif plist[0] == "Vertices":
                rem_nnode = line_counter+1
                continue
            elif plist[0] == "$Nodes":
                rem_nnode = line_counter+1
                continue
            elif plist[0] == "Triangles":
                rem_faces = line_counter+1
                continue
            elif plist[0] == "Tetrahedra":
                rem_nelem = line_counter+1
                continue
            elif plist[0] == "$Elements":
                rem_nelem = line_counter+1
                var = 1
                continue
            if rem_nnode == line_counter:
                self.nnode = int(plist[0])
            if rem_faces == line_counter:
                face_counter = int(plist[0])
            if rem_nelem == line_counter:
                self.nelem = int(plist[0])
                break

        # Re-read
        points, elements, faces, face_to_surface = [],[], [], []
        for line_counter, line in enumerate(open(filename)):
            item = line.rstrip()
            plist = item.split()
            if var == 0:
                if line_counter > rem_nnode and line_counter < self.nnode+rem_nnode+1:
                    points.append([float(i) for i in plist[:3]])
                if line_counter > rem_nelem and line_counter < self.nelem+rem_nelem+1:
                    elements.append([int(i) for i in plist[:4]])
            elif var == 1:
                if line_counter > rem_nnode and line_counter < self.nnode+rem_nnode+1:
                    points.append([float(i) for i in plist[1:]])
                if line_counter > rem_nelem and line_counter < self.nelem+rem_nelem+1:
                    if int(plist[1]) == el:
                        elements.append([int(i) for i in plist[5:]])

                    # WRITE SURFACE INFO - CERTAINLY ONLY IF ELEMENT TYPE IS QUADS/TRIS
                    if read_surface_info:
                        if int(plist[1]) == bel:
                            faces.append([int(i) for i in plist[5:]])
                            face_to_surface.append(int(plist[4]))


        self.points = np.array(points,copy=True)
        self.elements = np.array(elements,copy=True) - 1
        # CORRECT
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        if self.nelem == 0:
            raise ValueError("msh file does not contain {} elements".format(element_type))

        if read_surface_info:
            self.faces = np.array(faces,copy=True) - 1
            self.face_to_surface = np.array(face_to_surface, dtype=np.int64, copy=True).flatten()
            self.face_to_surface -= 1
            # CHECK IF FILLED
            if isinstance(self.face_to_surface,list):
                if not self.face_to_surface:
                    self.face_to_surface = None
            elif isinstance(self.face_to_surface,np.ndarray):
                if self.face_to_surface.shape[0]==0:
                    self.face_to_surface = None

        if self.points.shape[1] == 3:
            if np.allclose(self.points[:,2],0.):
                self.points = np.ascontiguousarray(self.points[:,:2])

        self.element_type = element_type
        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()

        return


    def ReadOBJ(self, filename, element_type="tri"):

        try:
            fid = open(filename, "r")
        except IOError:
            print("File '%s' not found." % (filename))
            sys.exit()

        if self.elements is not None and self.points is not None:
            self.__reset__()

        self.filename = filename


        bel = -1
        if element_type == "line":
            el = 2
        elif element_type == "tri":
            el = 3
            bel = 2
        elif element_type == "quad":
            el = 4
            bel = 2
        elif element_type == "tet":
            el = 4
            bel = 3
        elif element_type == "hex":
            el = 8
            bel = 4
        else:
            raise ValueError("Element type not understood")


        # Read
        points, elements, faces = [],[], []
        vertex_normal, vertex_texture = [], []
        for line_counter, line in enumerate(open(filename)):
            item = line.rstrip()
            plist = item.split()

            if plist[0] == 'v':
                points.append([float(i) for i in plist[1:4]])
            if plist[0] == 'f':
                for i in range(1,el+1):
                    if "/" in plist[i]:
                        plist[i] = plist[i].split("//")[0]
                elements.append([int(i) for i in plist[1:el+1]])
            if plist[0] == 'vn':
                vertex_normal.append([float(i) for i in plist[1:4]])


        self.points = np.array(points,copy=True)
        self.elements = np.array(elements,copy=True) - 1
        if not vertex_normal:
            self.vertex_normal = np.array(vertex_normal,copy=True)

        # CORRECT
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        if self.nelem == 0:
            raise ValueError("obj file does not contain {} elements".format(element_type))

        if self.points.shape[1] == 3:
            if np.allclose(self.points[:,2],0.):
                self.points = np.ascontiguousarray(self.points[:,:2])

        self.element_type = element_type
        ndim = self.InferSpatialDimension()
        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()


    def ReadFRO(self, filename, element_type):
        """Read fro mesh"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        if element_type == "tri":
            el = 5
        else:
            raise NotImplementedError("Reading FRO files for {} elements not yet implemented".format(element_type))

        content = np.fromfile(filename, dtype=np.float64, sep=" ")
        nelem = int(content[0])
        nnode = int(content[1])
        nsurface = int(content[3])

        points = content[8:8+4*nnode].reshape(nnode,4)[:,1:]
        elements = content[8+4*nnode::].reshape(nelem,el)[:,1:-1].astype(np.int64) - 1
        face_to_surface = content[8+4*nnode::].reshape(nelem,el)[:,-1].astype(np.int64) - 1

        self.nelem = nelem
        self.nnode = nnode
        self.elements = np.ascontiguousarray(elements)
        self.element_type = element_type
        self.points = np.ascontiguousarray(points)

        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()

        self.face_to_surface = np.ascontiguousarray(face_to_surface)

        return

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
        if isinstance(self.nelem,np.ndarray):
            self.nelem = int(self.nelem[0])

        for key in self.__dict__.keys():
            if isinstance(self.__dict__[str(key)],np.ndarray):
                if self.__dict__[str(key)].size == 1:
                    self.__dict__[str(key)] = np.asscalar(self.__dict__[str(key)])


    def ReadDCM(self, filename, element_type="quad", ndim=2):
        """ EZ4U mesh reader
        """

        if element_type != "quad":
            raise NotImplementedError("DCM/EZ4U reader for {} elements not yet implemented".format(element_type))

        self.__reset__()
        self.element_type = element_type

        content = np.fromfile(filename, dtype=np.float64, sep=" ")
        self.nnode = int(content[0])
        self.nelem = int(content[1])
        if ndim==2:
            self.points = content[3:self.nnode*4+3].reshape(self.nnode,4)[:,[1,2]]
        else:
            self.points = content[3:self.nnode*4+3].reshape(self.nnode,4)[:,1:]
        self.elements = content[self.nnode*4+3:].astype(np.int64).reshape(self.nelem,11)[:,7:] - 1

        if self.points.shape[1] == 3:
            if np.allclose(self.points[:,2],0.):
                self.points = np.ascontiguousarray(self.points[:,:2])

        self.GetEdgesQuad()
        self.GetBoundaryEdgesQuad()



    def SimplePlot(self, to_plot='faces',
        color=None, plot_points=False, plot_edges=True, point_radius=0.1,
        save=False, filename=None, figure=None, show_plot=True,
        show_axis=False, grid="off"):
        """Simple mesh plot

            to_plot:        [str] only for 3D. 'faces' to plot only boundary faces
                            or 'all_faces' to plot all faces
            grid:           [str] None, "on" or "off"
            """

        self.__do_essential_memebers_exist__()

        # REDIRECT FOR 3D SURFACE MESHES
        if self.element_type == "tri" or self.element_type == "quad":
            if self.points.ndim == 2 and self.points.shape[1] == 3:
                mesh = self.CreateDummy3DMeshfrom2DMesh()
                mesh.SimplePlot(to_plot=to_plot, color=color, plot_points=plot_points,
                    plot_edges=plot_edges, point_radius=point_radius,
                    save=save, filename=filename, figure=figure, show_plot=show_plot,
                    show_axis=show_axis, grid=grid)
                return

        if color is None:
            color=(197/255.,241/255.,197/255.)
        if grid is None:
            grid = "off"

        if save:
            if filename is None:
                warn('File name not given. I am going to write one in the current directory')
                filename = PWD(__file__) + "/output.png"
            else:
                if filename.split(".")[-1] == filename:
                    filename += ".png"

        import matplotlib as mpl
        if self.element_type == "tri" or self.element_type == "quad":
            import matplotlib.pyplot as plt
            if figure is None:
                figure = plt.figure()

        elif self.element_type == "tet" or self.element_type == "hex":
            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab

            if to_plot == 'all_faces':
                if self.all_faces is None:
                    self.GetFaces()
                faces = self.all_faces
            else:
                if self.faces is None:
                    self.GetBoundaryFaces()
                faces = self.faces
            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

            if color is not None:
                if isinstance(color,tuple):
                    if len(color) != 3:
                        raise ValueError("Color should be given in a rgb/RGB tuple format with 3 values i.e. (x,y,z)")
                    if color[0] > 1.0 or color[1] > 1.0 or color[2] > 1.0:
                        color = (color[0]/255.,color[1]/255.,color[2]/255.)
                elif isinstance(color,str):
                    color = mpl.colors.hex2color(color)


        if self.element_type == "tri":

            plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3],color='k')
            plt.axis("equal")
            if not show_axis:
                plt.axis('off')
            if grid == "on":
                plt.grid("on")
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
                self.points[:,2],faces[:,:3],color=color)
            radius = 1e-00
            if plot_edges:
                mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2], faces[:,:3],
                    line_width=radius,tube_radius=radius,color=(0,0,0),
                    representation='wireframe')

            if plot_points:
                mlab.points3d(self.points[:,0],self.points[:,1],self.points[:,2]
                    ,color=(0,0,0),mode='sphere',scale_factor=point_radius)

            # svpoints = self.points[np.unique(self.faces),:]
            # mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=0.005)

            # mlab.view(azimuth=135, elevation=45, distance=7, focalpoint=None,
            #     roll=0, reset_roll=True, figure=None)

            if show_plot:
                mlab.show()

        elif self.element_type=="quad":

            C = self.InferPolynomialDegree() - 1
            pdim = self.points.shape[1]

            edge_elements = self.GetElementsEdgeNumberingQuad()
            reference_edges = NodeArrangementQuad(C)[0]
            reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
            reference_edges = np.delete(reference_edges,1,1)

            self.GetEdgesQuad()
            x_edges = np.zeros((C+2,self.all_edges.shape[0]))
            y_edges = np.zeros((C+2,self.all_edges.shape[0]))
            z_edges = np.zeros((C+2,self.all_edges.shape[0]))

            BasesOneD = np.eye(2,2)
            for iedge in range(self.all_edges.shape[0]):
                ielem = edge_elements[iedge,0]
                edge = self.elements[ielem,reference_edges[edge_elements[iedge,1],:]]
                if pdim == 2:
                    x_edges[:,iedge], y_edges[:,iedge] = self.points[edge,:].T
                elif pdim == 3:
                    x_edges[:,iedge], y_edges[:,iedge], z_edges[:,iedge] = self.points[edge,:].T

            plt.plot(x_edges,y_edges,'-k')

            plt.axis('equal')
            if not show_axis:
                plt.axis('off')
            if grid == "on":
                plt.grid("on")
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

            if plot_edges:
                src = mlab.pipeline.scalar_scatter(tmesh.x_edges.T.copy().flatten(),
                    tmesh.y_edges.T.copy().flatten(), tmesh.z_edges.T.copy().flatten())
                src.mlab_source.dataset.lines = tmesh.connections
                lines = mlab.pipeline.stripper(src)
                h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=3)

            # mlab.view(azimuth=135, elevation=45, distance=7, focalpoint=None,
                # roll=0, reset_roll=True, figure=None)

            if show_plot:
                mlab.show()

        elif self.element_type == "line":

            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab

            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

            if self.points.ndim == 1:
                self.points = self.points[:,None]

            points = np.zeros((self.points.shape[0],3))
            if self.points.shape[1] == 1:
                points[:,0] = np.copy(self.points[:,0])
            if self.points.shape[1] == 2:
                points[:,:2] = np.copy(self.points)
            elif self.points.shape[1] == 3:
                points = np.copy(self.points)

            if plot_edges:
                src = mlab.pipeline.scalar_scatter(points[:,0],points[:,1],points[:,2])
                src.mlab_source.dataset.lines = self.elements[:,:2]
                lines = mlab.pipeline.stripper(src)
                h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

            if plot_points:
                h_points = mlab.points3d(points[:,0],points[:,1],points[:,2],color=(0,0,0),mode='sphere',scale_factor=point_radius)

            if show_plot:
                mlab.show()

        else:
            raise NotImplementedError("SimplePlot for {} not implemented yet".format(self.element_type))


        if save:
            ndim = self.InferSpatialDimension()
            if ndim == 2:
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

        elif self.element_type == "tet" or self.element_type == "hex":

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
            for i in range(self.faces.shape[0]):
                for j in range(self.faces.shape[1]):
                    if self.points[self.faces[i,j],2] < 30:
                        text_obj = mlab.text3d(self.points[self.faces[i,j],0],
                            self.points[self.faces[i,j],1],self.points[self.faces[i,j],2],str(self.faces[i,j]),color=(0,0,0.),scale=0.05)


            figure.scene.disable_render = False

            # mlab.view(*view)
            mlab.show()



    def WriteVTK(self, filename=None, result=None, fmt="binary", interpolation_degree=10, ProjectionFlags=None):
        """Write mesh/results to vtu

            inputs:
                fmt:                    [str] VTK writer format either "binary" or "xml".
                                        "xml" files do not support big vtk/vtu files
                                        typically greater than 2GB whereas "binary" files can.  Also "xml" writer is
                                        in-built whereas "binary" writer depends on evtk/pyevtk module
                interpolation_degree:   [int] used only for writing high order curved meshes
        """

        self.__do_essential_memebers_exist__()

        if fmt is "xml":
            pass
        elif fmt is "binary":
            try:
                from pyevtk.hl import pointsToVTK, linesToVTK, gridToVTK, unstructuredGridToVTK
                from pyevtk.vtk import VtkVertex, VtkLine, VtkTriangle, VtkQuad, VtkTetra, VtkPyramid, VtkHexahedron
            except ImportError:
                raise ImportError("Could not import evtk. Install it using 'pip install pyevtk'")
        else:
            raise ValueError("Writer format not understood")

        elements = np.copy(self.elements)

        cellflag = None
        if self.element_type =='tri':
            cellflag = 5
            offset = 3
            if self.elements.shape[1]==6:
                cellflag = 22
                offset = 6
        elif self.element_type =='quad':
            cellflag = 9
            offset = 4
            if self.elements.shape[1]==8:
                cellflag = 23
                offset = 8
        if self.element_type =='tet':
            cellflag = 10
            offset = 4
            if self.elements.shape[1]==10:
                cellflag = 24
                offset = 10
                # CHANGE NUMBERING ORDER FOR PARAVIEW
                para_arange = [0,4,1,6,2,5,7,8,9,3]
                elements = elements[:,para_arange]
        elif self.element_type == 'hex':
            cellflag = 12
            offset = 8
            if self.elements.shape[1] == 20:
                cellflag = 25
                offset = 20
        elif self.element_type == 'line':
            cellflag = 3
            offset = 2

        if filename is None:
            warn('File name not specified. I am going to write one in the current directory')
            filename = PWD(__file__) + "/output.vtu"
        if ".vtu" in filename and fmt is "binary":
            filename  = filename.split('.')[0]
        if ".vtu" not in filename and fmt is "xml":
            filename  = filename + ".vtu"


        if self.InferPolynomialDegree() > 1:
            try:
                from Florence.PostProcessing import PostProcess
                from Florence.VariationalPrinciple import DisplacementFormulation
            except ImportError:
                raise RuntimeError("Writing high order elements to VTK is not supported yet")
            if result is not None and result.ndim > 1:
                raise NotImplementedError("Writing multliple or vector/tensor valued results to binary vtk not supported yet")
                return
            else:
                if result is None:
                    result = np.zeros_like(self.points)[:,:,None]
                if result.ndim == 1:
                    result = result.reshape(result.shape[0],1,1)
                pp = PostProcess(3,3)
                pp.SetMesh(self)
                pp.SetSolution(result)
                pp.SetFormulation(DisplacementFormulation(self,compute_post_quadrature=False))
                pp.WriteVTK(filename,quantity=0,interpolation_degree=interpolation_degree, ProjectionFlags=ProjectionFlags)
                return


        if self.InferSpatialDimension() == 2:
            points = np.zeros((self.points.shape[0],3))
            points[:,:2] = self.points
        else:
            points = self.points

        if result is None:
            if fmt is "xml":
                write_vtu(Verts=self.points, Cells={cellflag:elements},fname=filename)
            elif fmt is "binary":
                unstructuredGridToVTK(filename,
                    np.ascontiguousarray(points[:,0]),np.ascontiguousarray(points[:,1]),
                    np.ascontiguousarray(points[:,2]), np.ascontiguousarray(elements.ravel()),
                    np.arange(0,offset*self.nelem,offset)+offset, np.ones(self.nelem)*cellflag)
        else:
            if isinstance(result, np.ndarray):
                if result.ndim > 1:
                    if result.size == result.shape[0]:
                        result = result.flatten()

                if fmt is "xml":
                    if result.ndim > 1:
                        if result.shape[0] == self.nelem:
                            write_vtu(Verts=self.points, Cells={cellflag:elements},
                                cvdata={cellflag:result.ravel()},fname=filename)
                        elif result.shape[0] == self.points.shape[0]:
                            write_vtu(Verts=self.points, Cells={cellflag:elements},
                                pvdata=result.ravel(),fname=filename)
                    else:
                        if result.shape[0] == self.nelem:
                            write_vtu(Verts=self.points, Cells={cellflag:elements},cdata=result,fname=filename)
                        elif result.shape[0] == self.points.shape[0]:
                            write_vtu(Verts=self.points, Cells={cellflag:elements},pdata=result,fname=filename)
                elif fmt is "binary":
                    if result.ndim <= 1:
                        if result.shape[0] == self.nelem:
                            unstructuredGridToVTK(filename,
                                np.ascontiguousarray(points[:,0]),np.ascontiguousarray(points[:,1]),
                                np.ascontiguousarray(points[:,2]), np.ascontiguousarray(elements.ravel()),
                                np.arange(0,offset*self.nelem,offset)+offset, np.ones(self.nelem)*cellflag,
                                cellData={'result':np.ascontiguousarray(result.ravel())})
                        elif result.shape[0] == self.points.shape[0]:
                            unstructuredGridToVTK(filename,
                                np.ascontiguousarray(points[:,0]),np.ascontiguousarray(points[:,1]),
                                np.ascontiguousarray(points[:,2]), np.ascontiguousarray(elements.ravel()),
                                np.arange(0,offset*self.nelem,offset)+offset, np.ones(self.nelem)*cellflag,
                                pointData={'result':np.ascontiguousarray(result.ravel())})
                    else:
                        raise NotImplementedError("Writing multliple or vector/tensor valued results to binary vtk not supported yet")




    def WriteHDF5(self, filename=None, external_fields=None):
        """Write to MATLAB's HDF5 format

            external_fields:        [dict or tuple] of fields to save together with the mesh
                                    for instance desired results. If a tuple is given keys in
                                    dictionary will be named results_0, results_1 and so on"""

        # DO NOT WRITE IF POINTS DO NOT EXIST - THIS IS TO PREVENT ACCIDENTAL WRITING OF
        # POTENTIALLU EMPTY MESH OBJECT
        if self.points is None:
            warn("Nothing to write")
            return

        Dict = deepcopy(self.__dict__)

        if external_fields is not None:
            if isinstance(external_fields,dict):
                Dict.update(external_fields)
            elif isinstance(external_fields,tuple):
                for counter, fields in enumerate(external_fields):
                    Dict['results_'+str(counter)] = fields
            else:
                raise AssertionError("Fields should be either tuple or a dict")

        if filename is None:
            pwd = os.path.dirname(os.path.realpath(__file__))
            filename = pwd+'/output.mat'

        for key in list(Dict.keys()):
            if Dict[str(key)] is None:
                del Dict[str(key)]

        savemat(filename, Dict, do_compression=True)



    @staticmethod
    def MeshPyTri(points,facets,*args,**kwargs):
        """MeshPy backend for generating linear triangular mesh"""
        info = triangle.MeshInfo()
        info.set_points(points)
        info.set_facets(facets)

        return triangle.build(info,*args,**kwargs)


    def Line(self, left_point=0., right_point=1., n=10, p=1):
        """Creates a mesh of on a line for 1D rods/beams"""

        self.__reset__()
        assert p > 0

        if not isinstance(left_point,float):
            if not isinstance(left_point,int):
                raise ValueError("left_point must be a number")
        if not isinstance(right_point,float):
            if not isinstance(right_point,int):
                raise ValueError("right_point must be a number")
        left_point = float(left_point)
        right_point = float(right_point)

        self.element_type = "line"
        self.points = np.linspace(left_point,right_point,p*n+1)[:,None]
        self.elements = np.zeros((n,p+1),dtype=np.int64)
        for i in range(p+1):
            self.elements[:,i] = p*np.arange(0,n)+i
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]


    def Rectangle(self,lower_left_point=(0,0), upper_right_point=(2,1),
        nx=5, ny=5, element_type="tri"):
        """Creates a quad/tri mesh of a rectangle"""

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
            self.nnode = self.points.shape[0]
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
            self.nnode = self.points.shape[0]
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

        a1 = np.sign(start_angle) if np.sign(start_angle)!=0. else np.sign(end_angle)
        a2 = np.sign(end_angle) if np.sign(end_angle)!=0. else np.sign(start_angle)
        if a1 == a2:
            total_angle = np.abs(end_angle - start_angle)
            if np.isclose(total_angle,0.) or np.isclose(total_angle,2.*np.pi) or total_angle > 2.*np.pi:
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

        # radius = np.linspace(inner_radius,outer_radius,nrad)
        # points = np.zeros((1,2),dtype=np.float64)
        # for i in range(nrad):
        #     t = np.linspace(start_angle,end_angle,ncirc+1)
        #     x = radius[i]*np.cos(t)[::-1]
        #     y = radius[i]*np.sin(t)[::-1]
        #     points = np.vstack((points,np.array([x,y]).T))
        # points = points[ncirc+2:,:]

        radius = np.linspace(inner_radius,outer_radius,nrad-1)
        points = np.zeros((1,2),dtype=np.float64)
        for i in range(nrad-1):
            t = np.linspace(start_angle,end_angle,ncirc+1)
            x = radius[i]*np.cos(t)[::-1]
            y = radius[i]*np.sin(t)[::-1]
            points = np.vstack((points,np.array([x,y]).T))
        points = points[1:,:]

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

        self.points = np.ascontiguousarray(self.points)


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



    def InverseArc(self, radius=10, start_angle=0, end_angle=np.pi/2., ncirc=3, element_type="tri"):
        """Generate inverse arc (concave arc with the upper portion being meshed).
            Note that this routine does not generate CAD-conformal meshes, that is not all the points
            on the arc would be on the arc
        """


        if np.allclose(radius,0):
            raise ValueError("Arc radius cannot be zero")
        if element_type != "tri" and element_type != "quad":
            raise ValueError("Element type can only be tri or quad")

        if ncirc > 3:
            ncirc = int(ncirc/2)


        t = np.linspace(start_angle,end_angle,ncirc+1)
        x = radius*np.cos(t)[::-1]
        y = radius*np.sin(t)[::-1]

        points = np.zeros((ncirc+2,2),dtype=np.float64)
        points[:-1,:] = np.array([x,y]).T
        points[-1,:] = [radius,radius]

        cmesh = Mesh()
        cmesh.points = points
        cmesh.elements = np.zeros((ncirc,3),dtype=np.int64)

        for i in range(ncirc):
            cmesh.elements[i,0] = i
            cmesh.elements[i,1] = ncirc+1
            cmesh.elements[i,2] = i+1

        mesh = Mesh()
        for i in range(ncirc):
            c0=(cmesh.points[cmesh.elements[i,0],0],cmesh.points[cmesh.elements[i,0],1])
            c1=(cmesh.points[cmesh.elements[i,1],0],cmesh.points[cmesh.elements[i,1],1])
            c2=(cmesh.points[cmesh.elements[i,2],0],cmesh.points[cmesh.elements[i,2],1])
            mesh1 = Mesh()
            mesh1.Triangle(c0,c1,c2,  element_type="quad", npoints=ncirc)
            if i==0:
                mesh = deepcopy(mesh1)
            mesh += mesh1

        if element_type == "tri":
            sys.stdout = open(os.devnull, "w")
            mesh.ConvertQuadsToTris()
            sys.stdout = sys.__stdout__


        self.points = mesh.points
        self.elements = mesh.elements
        self.element_type = element_type
        self.nelem = mesh.elements.shape[0]
        self.nnode = mesh.points.shape[0]

        self.GetBoundaryEdges()



    def CircularArcPlate(self, side_length=15, radius=10, center=(0.,0.),
        start_angle=0., end_angle=np.pi/4., ncirc=5, nrad=2, element_type="tri"):
        """Create an arc hole out-boxed by a squared geometry
        """

        if np.allclose(radius,0):
            raise ValueError("Arc radius cannot be zero")
        if element_type != "tri" and element_type != "quad":
            raise ValueError("Element type can only be tri or quad")

        if not np.isclose(start_angle,0.) and not \
            (np.isclose(end_angle,np.pi/4.) or np.isclose(end_angle,np.pi/2.) or np.isclose(end_angle,np.pi)):
            raise ValueError("Start and end angles should be 0 and 45 degrees respectively")

        self.__reset__()


        tmp_end_angle = np.pi/4.
        t = np.linspace(start_angle,tmp_end_angle,ncirc+1)
        x = radius*np.cos(t)[::-1]
        y = radius*np.sin(t)[::-1]

        points = np.array([x,y]).T
        points = np.flipud(points)

        plate_wall_ys = np.linspace(0.,side_length,ncirc+1)
        plate_wall_xs = np.zeros(ncirc+1) + side_length
        wpoints = np.array([plate_wall_xs,plate_wall_ys]).T

        lengths = np.linalg.norm(wpoints - points, axis=1)
        xs, ys = np.zeros((ncirc+1,nrad+1)), np.zeros((ncirc+1,nrad+1))
        for j in range(ncirc+1):
            xs[j,:] = np.linspace(points[j,0],wpoints[j,0],nrad+1)
            ys[j,:] = np.linspace(points[j,1],wpoints[j,1],nrad+1)
        self.points = np.array([xs.ravel(),ys.ravel()]).T


        self.elements = np.zeros((nrad*ncirc,4),dtype=np.int64)
        node_arranger = (nrad+1)*np.arange(ncirc)
        for i in range(nrad):
            self.elements[ncirc*i:(i+1)*ncirc,0] = node_arranger + i
            self.elements[ncirc*i:(i+1)*ncirc,1] = node_arranger + i + 1
            self.elements[ncirc*i:(i+1)*ncirc,2] = node_arranger + i + nrad + 2
            self.elements[ncirc*i:(i+1)*ncirc,3] = node_arranger + i + nrad + 1

        self.element_type = "quad"
        if np.isclose(end_angle,np.pi/2.):
            # First mirror the points along 45 degree axis
            # new_points   = np.copy(self.points)
            # new_elements = np.copy(self.elements)

            # dpoints  = np.zeros((2*new_points.shape[0]-1,2))
            # dpoints[:new_points.shape[0],:] = new_points
            # dpoints[new_points.shape[0]:,0] = new_points[:-1,1][::-1]
            # dpoints[new_points.shape[0]:,1] = new_points[:-1,0][::-1]

            # self.points = dpoints
            # self.elements = np.vstack((new_elements,new_elements+new_elements.max()))

            self.elements = np.fliplr(self.elements)
            mmesh = deepcopy(self)
            mmesh.points[:,0] = self.points[:,1][::-1]
            mmesh.points[:,1] = self.points[:,0][::-1]
            mmesh.elements = np.fliplr(mmesh.elements)
            self += mmesh




        if np.isclose(end_angle,np.pi):
            # First mirror the points along 45 degree axis
            self.elements = np.fliplr(self.elements)
            mmesh = deepcopy(self)
            mmesh.points[:,0] = self.points[:,1][::-1]
            mmesh.points[:,1] = self.points[:,0][::-1]
            mmesh.elements = np.fliplr(mmesh.elements)
            self += mmesh

            # Mirror along Y axis
            nmesh = deepcopy(self)
            nmesh.points[:,0] *= -1.
            self += nmesh


        # If called for stetching purposes its best to keep center at (0,0)
        self.points[:,0] += center[0]
        self.points[:,1] += center[1]

        # self.element_type = "quad"
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdges()

        if element_type == "tri":
            sys.stdout = open(os.devnull, "w")
            self.ConvertQuadsToTris()
            sys.stdout = sys.__stdout__



    def CircularPlate(self, side_length=30, radius=10, center=(0.,0.), ncirc=5, nrad=5, element_type="tri"):
        """Create a plate with hole
        """

        self.CircularArcPlate(side_length=side_length, radius=radius, center=(0,0),
            start_angle=0., end_angle=np.pi/4., ncirc=ncirc,
            nrad=nrad, element_type=element_type)

        # First mirror the points along 45 degree axis
        # new_points   = np.copy(self.points)
        # new_elements = np.copy(self.elements)
        # self.__reset__()

        # self.element_type = element_type

        # dpoints  = np.zeros((2*new_points.shape[0]-1,2))
        # dpoints[:new_points.shape[0],:] = new_points
        # dpoints[new_points.shape[0]:,0] = new_points[:-1,1][::-1]
        # dpoints[new_points.shape[0]:,1] = new_points[:-1,0][::-1]

        # self.points = dpoints
        # self.elements = np.vstack((new_elements,new_elements+new_elements.max()))

        self.elements = np.fliplr(self.elements)
        mmesh = deepcopy(self)
        mmesh.points[:,0] = self.points[:,1][::-1]
        mmesh.points[:,1] = self.points[:,0][::-1]
        mmesh.elements = np.fliplr(mmesh.elements)
        self += mmesh

        # Mirror along Y axis
        nmesh = deepcopy(self)
        nmesh.points[:,0] *= -1.
        self += nmesh

        # Mirror along X axis
        nmesh = deepcopy(self)
        nmesh.points[:,1] *= -1.
        self += nmesh

        # This needs to be done here
        self.points[:,0] += center[0]
        self.points[:,1] += center[1]

        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]
        self.GetBoundaryEdges()



    def Parallelepiped(self,lower_left_rear_point=(0,0,0), upper_right_front_point=(2,4,10),
        nx=2, ny=4, nz=10, element_type="hex"):
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
        self.nnode = self.points.shape[0]

        self.GetBoundaryFacesHex()
        self.GetBoundaryEdgesHex()

        if element_type == "tet":
            sys.stdout = open(os.devnull, "w")
            self.ConvertHexesToTets()
            sys.stdout = sys.__stdout__


    def Cube(self, lower_left_rear_point=(0.,0.,0.), side_length=1, nx=5, ny=5, nz=5, n=None, element_type="hex"):
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



    def SphericalArc(self, center=(0.,0.,0.), inner_radius=9., outer_radius=10.,
        start_angle=0., end_angle=np.pi/2., ncirc=5, nrad=5, nthick=1, element_type="hex"):

        from Florence.Tensor import makezero, itemfreq

        inner_radius = float(inner_radius)
        outer_radius = float(outer_radius)
        if np.allclose(inner_radius,0.):
            raise ValueError('inner_radius cannot be zero')
        if inner_radius > outer_radius:
            raise ValueError('inner_radius cannot be greater than outer_radius')
        self.__reset__()

        self.Arc(radius=outer_radius, element_type="quad",start_angle=start_angle,
            end_angle=end_angle,nrad=nrad,ncirc=ncirc)
        self.Extrude(length=outer_radius-inner_radius,nlong=nthick)

        points = np.copy(self.points)
        tts = itemfreq(points[:,2])
        radius = outer_radius

        # Apply projection for all layers through the thickness
        for i in range(tts.shape[0]):
            num = tts[i,0]
            cond = np.where(np.isclose(points[:,2],num))[0]

            layer_points = self.points[cond,:]
            tmp_radius = radius - num
            layer_points[:,1] *= tmp_radius/radius
            layer_points[:,0] *= tmp_radius/radius

            z_radius = tmp_radius**2 - layer_points[:,0]**2  - layer_points[:,1]**2
            makezero(z_radius[:,None],tol=1e-10)
            z_radius[z_radius<0] = 0.
            Z = np.sqrt(z_radius)
            self.points[cond,0] = layer_points[:,0]
            self.points[cond,1] = layer_points[:,1]
            self.points[cond,2] = radius - Z

        # Change back to make center at (0,0,0)
        self.points[:,2] *= -1.
        self.points[:,2] += radius

        for i in range(3):
            self.points[:,i] += center[i]

        self.GetBoundaryFaces()
        self.GetBoundaryEdges()

        if element_type == "tet":
            sys.stdout = open(os.devnull, "w")
            self.ConvertHexesToTets()
            sys.stdout = sys.__stdout__



    def HollowSphere(self, inner_radius=9., outer_radius=10.,
        ncirc=5, nrad=5, nthick=1, element_type="hex"):

        self.SphericalArc(inner_radius=inner_radius, outer_radius=outer_radius,
            ncirc=ncirc, nrad=nrad, nthick=nthick, element_type=element_type)

        # Mirror self in X, Y & Z
        for i in range(2):
            mesh = deepcopy(self)
            mesh.points[:,i] *=-1.
            self += mesh





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



    def HollowCylinder(self,center=(0,0,0),inner_radius=1.0,outer_radius=2.,
        element_type='hex',isotropic=True,nrad=5,ncirc=10, nlong=20,length=10):
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
        if mesh is not None:
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
            self.__reset__()

        if mesh.points.ndim == 2:
            if mesh.points.shape[1] == 3:
                raise ValueError("Cannot extrude a mesh which already has 3D nodal coordinates")

        nlong = int(nlong)
        if nlong==0:
            nlong = 1

        mp = mesh.InferPolynomialDegree()
        # LAYERS ONLY HOLD FOR UNIFORM (HEXAHEDRAL TYPE) EXTRUSIONS, BUT
        # THEN TETRAHEDRAL OR OTHER TYPES OF EXTRUSIONS ARE PROBABLY NOT
        # GOING TO MAKE IT TO THIS FUNCTION - USED FOR HIGH P EXTRUSIONS ONLY
        nlayer = mp
        nnode = (nlong + 1 + nlong*(nlayer-1))*mesh.points.shape[0]
        nnode_2D = mesh.points.shape[0]

        nelem= nlong*mesh.nelem
        nelem_2D = mesh.nelem
        nsize_2d = int((mp+1)**2)
        nsize = int((mp+1)**3)
        element_aranger = np.arange(nlong)
        self.elements = np.zeros((nelem,nsize),dtype=np.int64)

        self.points = np.zeros((nnode,3),dtype=np.float64)

        if mp == 1:
            # LINEAR MESH EXTRUSION
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

            for i in range(nlong):
                self.elements[nelem_2D*i:nelem_2D*(i+1),:nsize_2d] = mesh.elements + i*nnode_2D
                self.elements[nelem_2D*i:nelem_2D*(i+1),nsize_2d:] = mesh.elements + (i+1)*nnode_2D

        else:
            # HIGH ORDER MESH EXTRUSION
            if path is None:
                node_aranger = np.linspace(0,length,nlong+1)
                counter = 0
                for i in range(nlong):
                    node_aranger_2d = np.linspace(node_aranger[i],node_aranger[i+1],nlayer+1)
                    for j in range(nlayer):
                        self.points[nnode_2D*counter:nnode_2D*(counter+1),:2] = mesh.points
                        self.points[nnode_2D*counter:nnode_2D*(counter+1), 2] = node_aranger_2d[j]
                        counter += 1
                # LAST COUNTER
                self.points[nnode_2D*counter:nnode_2D*(counter+1),:2] = mesh.points
                self.points[nnode_2D*counter:nnode_2D*(counter+1), 2] = length
            else:
                raise NotImplementedError("Extruding high order elements along specified geometric path is not implemented yet")

            node_aranger = NodeArrangementLayeredToHex(mp-1)
            dum_e = np.zeros((nelem_2D,nsize))
            counter = 0
            for i in range(nlong):
                for j in range(nlayer+1):
                    dum_e[:,j*nsize_2d:(j+1)*nsize_2d] = mesh.elements + (counter+0)*nnode_2D
                    counter += 1
                counter -= 1
                self.elements[nelem_2D*i:nelem_2D*(i+1),:] = dum_e
            self.elements = self.elements[:,node_aranger]


        self.element_type = "hex"
        self.nelem = nelem
        self.nnode = nnode
        self.degree = mp
        self.GetBoundaryFaces()
        self.GetBoundaryEdges()

        if isinstance(path,GeometricPath):
            return points


    def RemoveDuplicateNodes(self, deci=8, tol=1e-08):
        """Remove duplicate points in the mesh
        """

        self.__do_essential_memebers_exist__()

        from Florence.Tensor import remove_duplicates_2D, makezero

        makezero(self.points,tol=1e-10)

        points, idx_points, inv_points = remove_duplicates_2D(self.points, decimals=8)
        if points.shape[0] == self.points.shape[0]:
            return

        unique_elements, inv_elements = np.unique(self.elements,return_inverse=True)
        unique_elements = unique_elements[inv_points]
        elements = unique_elements[inv_elements]
        elements = elements.reshape(self.elements.shape)

        # RECOMPUTE EVERYTHING
        self.elements = np.ascontiguousarray(elements, dtype=np.int64)
        self.points = np.ascontiguousarray(points, dtype=np.float64)
        self.nnode = self.points.shape[0]

        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()


    def RemoveElements(self, xyz_min_max, element_removal_criterion="all", keep_boundary_only=False, return_removed_mesh=False,
            compute_edges=True, compute_faces=True, plot_new_mesh=False):
        """Removes elements from the mesh given some specified criteria

        input:
            (x_min,y_min,z_min,x_max,y_max,z_max)       [tuple of floats or np.ndarray] of box selection. Deletes all the elements
                                                        apart from the ones within this box, either a tuple of 4/6 floats (2D/3D)
                                                        or 2D numpy array of shape (2,ndim) where ndim=2,3
            element_removal_criterion                   [str]{"all","any"} the criterion for element removal with box selection.
                                                        How many nodes of the element should be within the box in order
                                                        not to be removed. Default is "all". "any" implies at least one node
            keep_boundary_only                          [bool] delete all elements apart from the boundary ones
            return_removed_mesh                         [bool] return the removed mesh [inverse of what is selected]
            compute_edges                               [bool] if True also compute new edges
            compute_faces                               [bool] if True also compute new faces (only 3D)
            plot_new_mesh                               [bool] if True also plot the new mesh

        return:
            nodal_map:                                  [1D array] numbering of nodes in old mesh
            idx_kept_elements:                          [1D array] indices of kept element
            removed_mesh:                               [Mesh] an instance of removed mesh, returned only if return_removed_mesh=True

        1. Note that this method computes a new mesh without maintaining a copy of the original
        2. Different criteria can be mixed for instance removing all elements in the mesh apart from the ones
        in the boundary which are within a box
        """

        self.__do_memebers_exist__()

        ndim = self.InferSpatialDimension()
        if isinstance(xyz_min_max,tuple):
            if ndim==2:
                assert len(xyz_min_max)==4
                x_min = xyz_min_max[0]
                y_min = xyz_min_max[1]
                x_max = xyz_min_max[2]
                y_max = xyz_min_max[3]
            elif ndim == 3:
                assert len(xyz_min_max)==6
                x_min = xyz_min_max[0]
                y_min = xyz_min_max[1]
                z_min = xyz_min_max[2]
                x_max = xyz_min_max[3]
                y_max = xyz_min_max[4]
                z_max = xyz_min_max[5]
        elif isinstance(xyz_min_max,np.ndarray):
            assert xyz_min_max.shape == (2,ndim)
            if ndim==2:
                x_min = xyz_min_max[0,0]
                y_min = xyz_min_max[0,1]
                x_max = xyz_min_max[1,0]
                y_max = xyz_min_max[1,1]
            elif ndim == 3:
                x_min = xyz_min_max[0,0]
                y_min = xyz_min_max[0,1]
                z_min = xyz_min_max[0,2]
                x_max = xyz_min_max[1,0]
                y_max = xyz_min_max[1,1]
                z_max = xyz_min_max[1,2]

        if x_min >= x_max:
            raise ValueError("Invalid range for mesh removal")
        if y_min >= y_max:
            raise ValueError("Invalid range for mesh removal")
        if ndim == 3:
            if z_min >= z_max:
                raise ValueError("Invalid range for mesh removal")

        all_nelems = self.nelem

        if ndim==2:
            xe = self.points[self.elements,0]
            ye = self.points[self.elements,1]

            if element_removal_criterion == "all":
                cond =  np.logical_and(np.logical_and(np.logical_and(
                            (xe > x_min).all(axis=1),(ye > y_min).all(axis=1)),
                            (xe < x_max).all(axis=1)),(ye < y_max).all(axis=1))
            elif element_removal_criterion == "any":
                cond =  np.logical_and(np.logical_and(np.logical_and(
                            (xe > x_min).any(axis=1),(ye > y_min).any(axis=1)),
                            (xe < x_max).any(axis=1)),(ye < y_max).any(axis=1))

        elif ndim==3:
            xe = self.points[self.elements,0]
            ye = self.points[self.elements,1]
            ze = self.points[self.elements,2]

            if element_removal_criterion == "all":
                cond =  np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(
                            (xe > x_min).all(axis=1),(ye > y_min).all(axis=1)),
                            (ze > z_min).all(axis=1)),(xe < x_max).all(axis=1)),
                            (ye < y_max).all(axis=1)), (ze < z_max).all(axis=1))
            elif element_removal_criterion == "any":
                cond =  np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(
                            (xe > x_min).any(axis=1),(ye > y_min).any(axis=1)),
                            (ze > z_min).any(axis=1)),(xe < x_max).any(axis=1)),
                            (ye < y_max).any(axis=1)), (ze < z_max).any(axis=1))

        boundary_elements = np.arange(self.nelem)
        if keep_boundary_only == True:
            if ndim==2:
                boundary_elements = self.GetElementsWithBoundaryEdges()
            elif ndim==3:
                boundary_elements = self.GetElementsWithBoundaryFaces()
            cond_boundary = np.zeros(all_nelems,dtype=bool)
            cond_boundary[boundary_elements[:,0]] = True
            cond = np.logical_and(cond,cond_boundary)
            new_elements = self.elements[cond,:]
        else:
            new_elements = self.elements[cond,:]

        new_elements = new_elements.astype(self.elements.dtype)
        new_points = np.copy(self.points)
        element_type = self.element_type

        if return_removed_mesh:
            omesh = deepcopy(self)

        # RESET FIRST OR MESH WILL CONTAIN INCONSISTENT DATA
        self.__reset__()
        self.element_type = element_type
        self.elements = np.copy(new_elements)
        self.nelem = self.elements.shape[0]
        nodal_map, inv_elements =  np.unique(self.elements,return_inverse=True)
        self.points = new_points[nodal_map,:]
        # RE-ORDER ELEMENT CONNECTIVITY
        remap_elements =  np.arange(self.points.shape[0])
        self.elements = remap_elements[inv_elements].reshape(self.nelem,self.elements.shape[1])

        # self.edges = None
        # self.faces = None
        # RECOMPUTE EDGES
        if compute_edges == True:
            self.GetBoundaryEdges()

        # RECOMPUTE FACES
        if compute_faces == True:
            if self.element_type == "tet" or self.element_type == "hex":
                self.GetBoundaryFaces()
            if self.edges is not None:
                self.GetBoundaryEdges()

        if return_removed_mesh:
            new_elements = omesh.elements[~cond,:]
            new_elements = new_elements.astype(omesh.elements.dtype)
            new_points = np.copy(omesh.points)
            element_type = omesh.element_type
            # RESET FIRST OR MESH WILL CONTAIN INCONSISTENT DATA
            mesh = Mesh()
            mesh.__reset__()
            mesh.element_type = element_type
            mesh.elements = np.copy(new_elements)
            mesh.nelem = mesh.elements.shape[0]
            unique_elements_inv, inv_elements =  np.unique(mesh.elements,return_inverse=True)
            mesh.points = new_points[unique_elements_inv,:]
            # RE-ORDER ELEMENT CONNECTIVITY
            remap_elements =  np.arange(mesh.points.shape[0])
            mesh.elements = remap_elements[inv_elements].reshape(mesh.nelem,mesh.elements.shape[1])

            # RECOMPUTE EDGES
            if compute_edges == True:
                mesh.GetBoundaryEdges()

            # RECOMPUTE FACES
            if compute_faces == True:
                if mesh.element_type == "tet" or mesh.element_type == "hex":
                    mesh.GetBoundaryFaces()
                mesh.GetBoundaryEdges()



        # PLOT THE NEW MESH
        if plot_new_mesh == True:
            self.SimplePlot()

        aranger = np.arange(all_nelems)
        idx_kept_elements = aranger[cond]
        if return_removed_mesh:
            return nodal_map, idx_kept_elements, mesh
        else:
            return nodal_map, idx_kept_elements


    def MergeWith(self, mesh, self_solution=None, other_solution=None):
        """ Merges self with another mesh:
            NOTE: It is the responsibility of the user to ensure that meshes are conforming
        """

        self.__do_essential_memebers_exist__()
        mesh.__do_essential_memebers_exist__()

        if mesh.element_type != self.element_type:
            raise NotImplementedError('Merging two diffferent meshes is not possible yet')

        if self.elements.shape[1] != mesh.elements.shape[1]:
            warn('Elements are of not the same order. I am going to modify both meshes to their linear variants')
            if self.InferPolynomialDegree() > 1:
                dum = self.GetLinearMesh(remap=True)
                self.__dict__.update(dum.__dict__)
            if mesh.InferPolynomialDegree() > 1:
                mesh = mesh.GetLinearMesh(remap=True)

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
        melements = melements.reshape(nelem,nodeperelem).astype(np.int64)


        self.__reset__()
        self.element_type = element_type
        self.elements = melements
        self.nelem = melements.shape[0]
        self.points = mpoints
        self.nnode = mpoints.shape[0]

        ndim = self.InferSpatialDimension()
        if self.element_type == "tet" or self.element_type == "hex":
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()
        elif self.element_type == "tri" or self.element_type == "quad":
            self.GetBoundaryEdges()

        if self_solution is not None and other_solution is not None:
            if isinstance(self_solution,np.ndarray) and isinstance(other_solution,np.ndarray):
                if self_solution.ndim == 3 and other_solution.ndim == 3:
                    solution = np.concatenate((self_solution,other_solution),axis=0)
                    solution = solution[idx_mpoints,:,:]
                elif self_solution.ndim == 2 and other_solution.ndim == 2:
                    solution = np.concatenate((self_solution,other_solution),axis=0)
                    solution = solution[idx_mpoints,:]

            return solution


    def Smooth(self, criteria={'aspect_ratio':3}):
        """Performs mesh smoothing based on a given criteria.

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


        criterion = list(criteria.keys())[0]
        number = list(criteria.values())[0]

        if "aspect_ratio" in insensitive(criterion):
            quantity = self.AspectRatios()
        elif "area" in insensitive(criterion):
            quantity = self.Areas()
        elif "volume" in insensitive(criterion):
            quantity = self.Volumes()
        else:
            quantity = self.AspectRatios()

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


    def Refine(self, level=2):
        """Refines a given mesh (self) to specified level uniformly.

            Note that uniform refinement implies two things:
            1. "ALL" elements will refinement, otherwise non-conformal elements will be created
            2. Equal refinement takes place in every direction
        """

        from scipy.spatial import Delaunay
        try:
            from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints, EquallySpacedPointsTri, EquallySpacedPointsTet
            from Florence.FunctionSpace import Line, Tri, Quad, Tet, Hex
            from Florence.FunctionSpace.OneDimensional.Line import Lagrange
            from Florence.Tensor import remove_duplicates_2D
        except ImportError:
            raise ImportError("This functionality requires florence's support")


        # WE NEED AN ACTUAL NDIM
        # ndim = self.InferSpatialDimension()
        if self.element_type == "line":
            ndim = 1
        if self.element_type == "tri" or self.element_type == "quad":
            ndim = 2
        elif self.element_type == "tet" or self.element_type == "hex":
            ndim = 3

        mesh = deepcopy(self)
        if mesh.InferPolynomialDegree() > 1:
            mesh = mesh.GetLinearMesh(remap=True)

        C = level - 1
        p = C+1
        # CActual = self.InferPolynomialDegree() - 1
        CActual = 0 # MUST BE ALWAYS ZERO
        if self.element_type == "line":
            nsize = int(C+2)
            nsize_2 = int(CActual+2)
        elif self.element_type == "tri":
            nsize = int((p+1)*(p+2)/2.)
            nsize_2 = int((CActual+2)*(CActual+3)/2.)
        elif self.element_type == "quad":
            nsize = int((C+2)**2)
            nsize_2 = int((CActual+2)**2)
        elif self.element_type == "tet":
            nsize = int((p+1)*(p+2)*(p+3)/6.)
            nsize_2 = int((CActual+2)*(CActual+3)*(CActual+4)/6.)
        elif self.element_type == "hex":
            nsize = int((C+2)**3)
            nsize_2 = int((CActual+2)**3)
        else:
            raise ValueError("Element type not undersood")

        if self.element_type == "line":
            SingleElementPoints = EquallySpacedPoints(ndim+1,C).ravel()

        elif self.element_type == "quad" or self.element_type == "hex":
            SingleElementPoints = EquallySpacedPoints(ndim+1,C)
            # RE-ARANGE NODES PROVIDED BY EquallySpacedPoints
            if ndim == 2:
                node_aranger = np.lexsort((SingleElementPoints[:,0],SingleElementPoints[:,1]))
            elif ndim == 3:
                node_aranger = np.lexsort((SingleElementPoints[:,0],SingleElementPoints[:,1],SingleElementPoints[:,2]))
            SingleElementPoints = SingleElementPoints[node_aranger,:]

        elif self.element_type == "tri":
            SingleElementPoints = EquallySpacedPointsTri(C)
            simplices = Delaunay(SingleElementPoints).simplices.copy()
            nsimplices = simplices.shape[0]

        elif self.element_type == "tet":
            SingleElementPoints = EquallySpacedPointsTet(C)
            simplices = Delaunay(SingleElementPoints).simplices.copy()
            nsimplices = simplices.shape[0]


        Bases = np.zeros((nsize_2,SingleElementPoints.shape[0]),dtype=np.float64)

        if mesh.element_type == "line":
            smesh = Mesh()
            smesh.Line(n=level)
            simplices = smesh.elements
            nsimplices = smesh.nelem

            hpBases = Line.Lagrange
            for i in range(SingleElementPoints.shape[0]):
                Bases[:,i] = hpBases(CActual,SingleElementPoints[i])[0]

        elif mesh.element_type == "tri":
            hpBases = Tri.hpNodal.hpBases
            for i in range(SingleElementPoints.shape[0]):
                Bases[:,i] = hpBases(CActual,SingleElementPoints[i,0],SingleElementPoints[i,1],
                    EvalOpt=1,equally_spaced=True,Transform=1)[0]

        elif mesh.element_type == "quad":
            smesh = Mesh()
            smesh.Rectangle(element_type="quad", nx=level, ny=level)
            simplices = smesh.elements
            nsimplices = smesh.nelem

            hpBases = Quad.LagrangeGaussLobatto
            for i in range(SingleElementPoints.shape[0]):
                Bases[:,i] = hpBases(CActual,SingleElementPoints[i,0],SingleElementPoints[i,1])[:,0]

        elif mesh.element_type == "tet":
            hpBases = Tet.hpNodal.hpBases
            for i in range(SingleElementPoints.shape[0]):
                Bases[:,i] = hpBases(CActual,SingleElementPoints[i,0],SingleElementPoints[i,1],
                    SingleElementPoints[i,2],EvalOpt=1,equally_spaced=True,Transform=1)[0]

        elif mesh.element_type == "hex":
            smesh = Mesh()
            smesh.Parallelepiped(element_type="hex", nx=level, ny=level, nz=level)
            simplices = smesh.elements
            nsimplices = smesh.nelem

            hpBases = Hex.LagrangeGaussLobatto
            for i in range(SingleElementPoints.shape[0]):
                Bases[:,i] = hpBases(CActual,SingleElementPoints[i,0],SingleElementPoints[i,1],SingleElementPoints[i,2])[:,0]


        nnode = nsize*mesh.nelem
        nelem = nsimplices*mesh.nelem
        X = np.zeros((nnode,mesh.points.shape[1]),dtype=np.float64)
        T = np.zeros((nelem,mesh.elements.shape[1]),dtype=np.int64)

        for ielem in range(mesh.nelem):
            X[ielem*nsize:(ielem+1)*nsize,:] = np.dot(Bases.T, mesh.points[mesh.elements[ielem,:],:])
            T[ielem*nsimplices:(ielem+1)*nsimplices,:] = simplices + ielem*nsize

        # REMOVE DUPLICATES
        repoints, idx_repoints, inv_repoints = remove_duplicates_2D(X, decimals=10)
        unique_reelements, inv_reelements = np.unique(T,return_inverse=True)
        unique_reelements = unique_reelements[inv_repoints]
        reelements = unique_reelements[inv_reelements]
        reelements = reelements.reshape(nelem,mesh.elements.shape[1])

        self.__reset__()
        self.elements = np.ascontiguousarray(reelements)
        self.points = np.ascontiguousarray(repoints)
        self.element_type = mesh.element_type
        self.nelem = self.elements.shape[0]
        self.nnode = self.points.shape[0]

        if self.element_type == "tri" or self.element_type == "quad":
            self.GetEdges()
            self.GetBoundaryEdges()
        elif self.element_type == "tet" or self.element_type == "hex":
            self.GetFaces()
            self.GetBoundaryFaces()
            self.GetBoundaryEdges()

        if CActual > 0:
            sys.stdout = open(os.devnull, "w")
            self.GetHighOrderMesh(p=CActual+1)
            # self.GetHighOrderMesh(p=CActual+1, equally_spaced=equally_spaced, check_duplicates=False)
            sys.stdout = sys.__stdout__



    def Partition(self, n=2, figure=None, show_plot=False, compute_boundary_info=True):
        """Partitions any type of mesh low and high order
            into a set of meshes.
            Returns a list of partitioned meshes and their index map
            into the origin mesh as well as list of node maps
        """

        num_par = n
        partitioned_indices = np.array_split(np.arange(self.nelem),num_par)
        nelems = [a.shape[0] for a in partitioned_indices]

        pmesh, partitioned_nodes_indices = [], []
        for i in range(len(nelems)):
            pmesh.append(self.GetLocalisedMesh(partitioned_indices[i], compute_boundary_info=compute_boundary_info))
            partitioned_nodes_indices.append(np.unique(self.elements[partitioned_indices[i],:]))


        if show_plot:

            import itertools
            colors = itertools.cycle(('#D1655B','#44AA66','#FACD85','#70B9B0','#72B0D7','#E79C5D',
                    '#4D5C75','#FFF056','#558C89','#F5CCBA','#A2AB58','#7E8F7C','#005A31'))

            if self.element_type == "tri" or self.element_type == "quad":
                try:
                    from Florence.PostProcessing import PostProcess
                except ImportError:
                    raise ImportError("This functionality requires florence's support")

                import matplotlib.pyplot as plt
                if figure is None:
                    figure = plt.figure()

                pp = PostProcess(2,2)
                for mesh in pmesh:
                    pp.CurvilinearPlot(mesh,figure=figure,show_plot=False,color=colors.next(),interpolation_degree=0)

                plt.show()

            elif self.element_type == "tet" or self.element_type == "hex" or self.element_type == "line":
                import os
                os.environ['ETS_TOOLKIT'] = 'qt4'
                from mayavi import mlab

                if figure is None:
                    figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

                for mesh in pmesh:
                    mesh.SimplePlot(figure=figure,show_plot=False,color=colors.next())

                mlab.show()

        return pmesh, partitioned_indices, partitioned_nodes_indices



    @staticmethod
    def TriangularProjection(c1=(0,0), c2=(2,0), c3=(2,2), points=None, npoints=10, equally_spaced=True):
        """Builds an instance of Mesh on a triangular region through FE interpolation
            given three vertices of the triangular region. Alternatively one can specify
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
                EvalOpt=1,equally_spaced=equally_spaced,Transform=1)[0]

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
            given four vertices of the quadrilateral region. Alternatively one can specify
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
    def TetrahedralProjection(c1=(0,0,0), c2=(2,0,0), c3=(0,2,0), c4=(0,0,2), points=None, npoints=10, equally_spaced=True):
        """Builds an instance of Mesh on a tetrahedral region through FE interpolation
            given four vertices of the tetrahedral region. Alternatively one can specify
            the vertices as numpy array of 4x3.

            This is a static immutable function, in that it does not modify self
        """

        if points is None or not isinstance(points,np.ndarray):
            if not isinstance(c1,tuple) or not isinstance(c2,tuple) or not isinstance(c3,tuple) \
                or not isinstance(c4,tuple):
                raise ValueError("coordinates should be given in tuples of two elements (x,y)")
            else:
                c1 = np.array(c1); c2 = np.array(c2); c3 = np.array(c3); c4 = np.array(c4)
                opoints = np.vstack((c1,c2,c3,c4))
        else:
            opoints = points

        from scipy.spatial import Delaunay
        from Florence.FunctionSpace import Tet
        from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPointsTet
        from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet

        if equally_spaced:
            points = EquallySpacedPointsTet(npoints)
        else:
            points = FeketePointsTet(npoints)

        BasesTet = np.zeros((4,points.shape[0]),dtype=np.float64)
        hpBases = Tet.hpNodal.hpBases
        for i in range(points.shape[0]):
            BasesTet[:,i] = hpBases(0,points[i,0],points[i,1],points[i,2],
                Transform=1,EvalOpt=1,equally_spaced=equally_spaced)[0]
        makezero(BasesTet,tol=1e-10)

        # func = Delaunay(points,qhull_options="QJ") # this does not produce the expected connectivity
        func = Delaunay(points)
        tets = func.simplices

        mesh = Mesh()
        mesh.element_type="tet"
        mesh.points = np.dot(BasesTet.T, opoints)
        mesh.elements = tets
        mesh.nelem = mesh.elements.shape[0]
        mesh.nnode = mesh.points.shape[0]
        mesh.GetBoundaryFaces()
        mesh.GetBoundaryEdges()

        return mesh



    @staticmethod
    def HexahedralProjection(c1=(0,0,0), c2=(2,0,0), c3=(2,2,0), c4=(0,2,0.),
        c5=(0,1.8,3.), c6=(0.2,0,3.), c7=(2,0.2,3.), c8=(1.8,2,3.),  points=None, npoints=6, equally_spaced=True):
        """Builds an instance of Mesh on a hexahedral region through FE interpolation
            given eight vertices of the quadrilateral region. Alternatively one can specify
            the vertices as numpy array of 8x3.

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

        # nnode = hmesh.nnode
        # nelem = hmesh.nelem
        # nsize = int((npoints+1)**3)

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

        self.__do_essential_memebers_exist__()
        self.points = np.ascontiguousarray(self.points.astype(np.float64))
        if isinstance(self.elements,np.ndarray):
            self.elements = np.ascontiguousarray(self.elements.astype(np.uint64))
        if hasattr(self, 'edges'):
            if isinstance(self.edges,np.ndarray):
                self.edges = np.ascontiguousarray(self.edges.astype(np.uint64))
        if hasattr(self, 'faces'):
            if isinstance(self.faces,np.ndarray):
                self.faces = np.ascontiguousarray(self.faces.astype(np.uint64))


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

        elif self.element_type == "line":
            for i in range(100):
                if int(i+1)==self.elements.shape[1]:
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


    @property
    def IsCurvilinear(self):

        self.__do_essential_memebers_exist__()

        ndim = self.InferSpatialDimension()
        p = self.InferPolynomialDegree()

        is_curvilinear = False

        if self.element_type != "line":
            # FOR LINE ELEMENTS THIS APPROACH DOES NOT WORK AS JACOBIAN IS NOT WELL DEFINED

            # GET CURVED VOLUME
            curved_vol = self.Sizes().sum()
            # GET PLANAR VOLUME
            if self.element_type == "tet" or self.element_type == "hex":
                planar_vols = self.Volumes()
            elif self.element_type == "tri" or self.element_type == "quad":
                planar_vols = self.Areas()
            elif self.element_type == "line":
                planar_vols = self.Lengths()
            planar_vol = planar_vols.sum()
            # COMPARE THE TWO
            if not np.isclose(planar_vol, curved_vol, rtol=1e-6, atol=1e-6):
                is_curvilinear = True

            return is_curvilinear

        # else:
        #     # ANOTHER PROMISING TECHNIQUE FOR ALL TYPES OF ELEMENTS.
        #     # BUT IT ALSO DOES NOT WORK FOR LINES
        #     # IT ALSO DOES NOT WORK FOR 3D ELEMENTS AS WE ONLY TESSELATE SURFACES NOT VOLUMES
        #     from Florence.PostProcessing import PostProcess
        #     tmesh = PostProcess(ndim,ndim).Tessellate(self,np.zeros_like(self.points),interpolation_degree=0)
        #     tmesh.RemoveDuplicateNodes()
        #     error = tmesh.Lengths().sum() - self.Lengths().sum()
        #     if not np.isclose(error, 0., rtol=1e-6, atol=1e-6):
        #         is_curvilinear = True


        # ACTIVE ONLY FOR LINE ELEMENTS RIGHT NOW,
        # ALTERNATIVELY FOR LINE WE CAN CHECK COLINEARITY
        # THAT WAY THIS FUNCTION WOULD NOT DEPEND ON IsEquallySpaced
        # VERY SIMPLE WAY TO CHECK IF A MESH IS CURVED
        if p == 1:
            is_curvilinear = False
        else:
            sys.stdout = open(os.devnull, "w")

            mesh = deepcopy(self)
            mesh = mesh.GetLinearMesh(remap=True)
            # NOTE THAT IF OTHER ARGS WERE PASSED TO GetHighOrderMesh
            # THIS WILL FAIL
            if not self.IsEquallySpaced:
                mesh.GetHighOrderMesh(p=p, equally_spaced=False)
                error = np.linalg.norm(mesh.points - self.points)
                if not np.isclose(error, 0., rtol=1e-6, atol=1e-6):
                    is_curvilinear = True
            else:
                mesh = mesh.GetLinearMesh(remap=True)
                mesh.GetHighOrderMesh(p=p, equally_spaced=True)
                error = np.linalg.norm(mesh.points - self.points)
                if not np.isclose(error, 0., rtol=1e-6, atol=1e-6):
                    is_curvilinear = True

            sys.stdout = sys.__stdout__


        return is_curvilinear



    @property
    def IsEquallySpaced(self):

        self.__do_essential_memebers_exist__()

        # FOR CURVILINEAR MESHES THIS STRATEGY WILL FAIL
        # HOWEVER AT THE MOMENT CALLING self.IsCurvilinear INTRODUCES
        # CYCLIC DEPENDENCY FOR LINES. IN ESSENCE, SINCE WE FEA ARE DONE
        # MAINLY ON FEKETE POINTS IF A CURVED MESH COMES IN IT WILL BE
        # AUTOMATICALLY FALSE
        is_equally_spaced = False
        p = self.InferPolynomialDegree()
        if p == 1:
            # FOR p=1 THIS NEEDS TO BE FALSE - AFFECTS ELSEWHERE (TRACTION COMPUTATION)
            is_equally_spaced = False
        elif p == 2:
            # FOR p=2 THIS IS ALWAYS TRUE
            is_equally_spaced = True
        elif p > 2:
            try:
                from Florence.QuadratureRules.EquallySpacedPoints import EquallySpacedPoints, EquallySpacedPointsTri, EquallySpacedPointsTet
                from Florence.FunctionSpace import Line, Tri, QuadES, Tet, HexES
                from Florence.MeshGeneration.NodeArrangement import NodeArrangementLine
            except ImportError:
                raise ValueError("This functionality requires Florence's support")


            if self.element_type == "line":
                ndim = 1
                nsize = 2
                eps = EquallySpacedPoints(ndim+1,p-1).ravel()
                # ARRANGE NODES FOR LINES HERE (DONE ONLY FOR LINES) - IMPORTANT
                node_aranger = NodeArrangementLine(p-1)
                eps = eps[node_aranger]
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(0,eps.shape[0]):
                    Neval[:,i] = Line.Lagrange(0,eps[i])[0]

            elif self.element_type == "tri":
                ndim = 2
                nsize = 3
                eps =  EquallySpacedPointsTri(p-1)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                hpBases = Tri.hpNodal.hpBases
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(0,eps.shape[0]):
                    Neval[:,i]  = hpBases(0,eps[i,0],eps[i,1],Transform=1,EvalOpt=1,equally_spaced=True)[0]

            elif self.element_type == "quad":
                ndim = 2
                nsize = 4
                eps = EquallySpacedPoints(ndim+1,p-1)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(0,eps.shape[0]):
                    Neval[:,i] = QuadES.Lagrange(0,eps[i,0],eps[i,1],arrange=1)[:,0]

            elif self.element_type == "tet":
                ndim = 3
                nsize = 4
                eps =  EquallySpacedPointsTet(p-1)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                hpBases = Tet.hpNodal.hpBases
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(0,eps.shape[0]):
                    Neval[:,i]  = hpBases(0,eps[i,0],eps[i,1],eps[i,2],Transform=1,EvalOpt=1,equally_spaced=True)[0]

            elif self.element_type == "hex":
                ndim = 3
                nsize = 8
                eps = EquallySpacedPoints(ndim+1,p-1)
                # COMPUTE BASES FUNCTIONS AT ALL NODAL POINTS
                Neval = np.zeros((nsize,eps.shape[0]),dtype=np.float64)
                for i in range(0,eps.shape[0]):
                    Neval[:,i] = HexES.Lagrange(0,eps[i,0],eps[i,1],eps[i,2],arrange=1)[:,0]


            linear_points = self.points[self.elements[0,:nsize],:]
            bench_points = np.dot(Neval.T,linear_points)

            candidate_points = self.points[self.elements[0,:],:]

            # CHECK IF MESH POINTS AND LINEAR PROJECTION POINTS MATCH
            error = np.linalg.norm(bench_points - candidate_points)

            if np.isclose(error, 0., rtol=1e-6, atol=1e-6):
                is_equally_spaced = True


        return is_equally_spaced


    def IsSimilar(self,other):
        """Checks if two meshes are similar.
            Similarity is established in terms of element type.
            This is not a property, since property methods can
            take variables
        """
        self.__do_essential_memebers_exist__()
        other.__do_essential_memebers_exist__()
        return self.element_type == other.element_type


    def IsEqualOrder(self,other):
        """Checks if two meshes are equal order.
            This is not a property, since property methods can
            take variables
        """
        return self.InferPolynomialDegree() == other.InferPolynomialDegree()


    def GetNumberOfElements(self):
        if self.nelem != None:
            return self.nelem
        assert self.elements is not None
        self.nelem = self.elements.shape[0]
        return self.nelem


    def GetNumberOfNodes(self):
        if self.nnode != None:
            return self.nnode
        assert self.points is not None
        self.nnode = self.points.shape[0]
        return self.nnode


    def InferSpatialDimension(self):
        """Infer the spatial dimension of the mesh"""

        assert self.points is not None
        # if self.points.shape[1] == 3:
        #     if self.element_type == "tri" or self.element_type == "quad":
        #         print("3D surface mesh of ", self.element_type)

        return self.points.shape[1]


    def InferNumberOfNodesPerElement(self, p=None, element_type=None):
        """Infers number of nodes per element. If p and element_type are
            not None then returns the number of nodes required for the given
            element type with the given polynomial degree"""

        if p is not None and element_type is not None:
            if element_type=="line":
                return int(p+1)
            elif element_type=="tri":
                return int((p+1)*(p+2)/2)
            elif element_type=="quad":
                return int((p+1)**2)
            elif element_type=="tet":
                return int((p+1)*(p+2)*(p+3)/6)
            elif element_type=="hex":
                return int((p+1)**3)
            else:
                raise ValueError("Did not understand element type")

        assert self.elements.shape[0] is not None
        return self.elements.shape[1]


    def InferNumberOfNodesPerLinearElement(self, element_type=None):
        """Infers number of nodes per element. If element_type are
            not None then returns the number of nodes required for the given
            element type"""

        if element_type is None and self.element_type is None:
            raise ValueError("Did not understand element type")
        if element_type is None:
            element_type = self.element_type

        tmp = self.element_type
        if element_type != self.element_type:
            self.element_type = element_type

        nodeperelem = None
        if element_type=="line":
            nodeperelem = 2
        elif element_type=="tri":
            nodeperelem = 3
        elif element_type=="quad":
            nodeperelem = 4
        elif element_type=="tet":
            nodeperelem = 4
        elif element_type=="hex":
            nodeperelem = 8
        else:
            raise ValueError("Did not understand element type")

        self.element_type = tmp

        return nodeperelem



    def InferElementType(self):

        if self.element_type is not None:
            return self.element_type

        assert self.elements is not None
        assert self.points is not None

        ndim = self.InferSpatialDimension()
        nodeperelem = self.InferNumberOfNodesPerElement()
        nn = 20

        if ndim==3:
            if nodeperelem in [int((i+1)*(i+2)*(i+3)/6) for i in range(1,nn)]:
                self.element_type = "tet"
            elif nodeperelem in [int((i+1)**3) for i in range(1,nn)]:
                self.element_type = "hex"
            else:
                if nodeperelem in [int((i+1)*(i+2)/2) for i in range(1,nn)]:
                    self.element_type = "tri"
                elif nodeperelem in [int((i+1)**2) for i in range(1,nn)]:
                    self.element_type = "quad"
                else:
                    raise ValueError("Element type not understood")
        elif ndim==2:
            if nodeperelem in [int((i+1)*(i+2)/2) for i in range(1,nn)]:
                self.element_type = "tri"
            elif nodeperelem in [int((i+1)**2) for i in range(1,nn)]:
                self.element_type = "quad"
            else:
                raise ValueError("Element type not understood")
        elif ndim==1:
            self.element_type = "line"
        else:
            raise ValueError("Element type not understood")

        # IF POINTS ARE CO-PLANAR THEN IT IS NOT TET BUT QUAD
        if ndim == 3 and self.element_type == "tet":
            a = self.points[self.elements[:,0],:]
            b = self.points[self.elements[:,1],:]
            c = self.points[self.elements[:,2],:]
            d = self.points[self.elements[:,3],:]

            det_array = np.dstack((a-d,b-d,c-d))
            # FIND VOLUME OF ALL THE ELEMENTS
            volume = 1./6.*np.linalg.det(det_array)
            if np.allclose(volume,0.0):
                self.element_type = "quad"

        return self.element_type


    def InferBoundaryElementType(self):

        self.InferElementType()
        if self.element_type == "hex":
            self.boundary_element_type = "quad"
        elif self.element_type == "tet":
            self.boundary_element_type = "tri"
        elif self.element_type == "quad" or self.element_type == "tri":
            self.boundary_element_type = "line"
        elif self.element_type == "line":
            self.boundary_element_type = "point"
        else:
            raise ValueError("Could not understand element type")

        return self.boundary_element_type



    def GetLinearMesh(self, solution=None, remap=False):
        """Returns the linear mesh from a high order mesh. If mesh is already linear returns the same mesh.
            Also maps any solution vector/tensor of high order mesh to the linear mesh, if supplied.
            For safety purposes, always makes a copy"""

        self.__do_essential_memebers_exist__()

        ndim = self.InferSpatialDimension()
        if ndim==2:
            if self.element_type == "tri" or self.element_type == "quad":
                assert self.edges is not None
        elif ndim==3:
            if self.element_type == "tet" or self.element_type == "hex":
                assert self.faces is not None


        if self.IsHighOrder is False:
            if solution is not None:
                return deepcopy(self), deepcopy(solution)
            return deepcopy(self)
        else:
            if not remap:
                # WORKS ONLY IF THE FIST COLUMNS CORRESPOND TO
                # LINEAR CONNECTIVITY
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
                    lmesh.faces = np.copy(self.faces[:,:4])
                    lmesh.nnode = int(np.max(lmesh.elements)+1)
                    lmesh.points = np.copy(self.points[:lmesh.nnode,:])
                lmesh.nelem = lmesh.elements.shape[0]

                if solution is not None:
                    solution = solution[np.unique(lmesh.elements),...]
                    return lmesh, solution

            else:
                # WORKS FOR ALL CASES BUT REMAPS (NO MAPPING BETWEEN LOW AND HIGH ORDER)
                nodeperelem = self.InferNumberOfNodesPerLinearElement()
                lmesh = Mesh()
                lmesh.element_type = self.element_type
                lmesh.nelem = self.nelem
                unnodes, inv = np.unique(self.elements[:,:nodeperelem], return_inverse=True)
                aranger = np.arange(lmesh.nelem*nodeperelem)
                lmesh.elements = inv[aranger].reshape(lmesh.nelem,nodeperelem)
                lmesh.points = self.points[unnodes,:]
                if lmesh.element_type == "hex" or lmesh.element_type == "tet":
                    lmesh.GetBoundaryFaces()
                    lmesh.GetBoundaryEdges()
                elif lmesh.element_type == "quad" or lmesh.element_type == "tri":
                    lmesh.GetBoundaryEdges()

                if solution is not None:
                    solution = solution[unnodes,...]
                    return lmesh, solution

        return lmesh


    def GetLocalisedMesh(self, elements, solution=None, compute_boundary_info=True):
        """Make a new Mesh instance from part of a big mesh.
            Makes a copy and does not modify self

            inputs:
                elements:           [int, tuple, list, 1D array] list of elements in big mesh (self)
                                    from which a small localised mesh needs to be extracted.
                                    It could also be an array of boolean of size self.nelem (in which case
                                    the dtype of array has to be strictly np.bool)
                solution            [1D array having the same length as big mesh points]
                                    if a solution also needs to be mapped over the localised element
        """

        self.__do_essential_memebers_exist__()

        elements = np.array(elements).flatten()

        if elements.dtype == np.bool:
            if elements.shape[0] != self.elements.shape[0]:
                raise ValueError("Boolean array should be the same size as number of elements")
                return
            elements = np.where(elements==True)[0]

        nodeperelem = self.elements.shape[1]
        tmesh = Mesh()
        tmesh.element_type = self.element_type
        unnodes, inv = np.unique(self.elements[elements,:nodeperelem], return_inverse=True)
        aranger = np.arange(elements.shape[0]*nodeperelem)
        tmesh.elements = inv[aranger].reshape(elements.shape[0],nodeperelem)
        tmesh.points = self.points[unnodes,:]
        tmesh.nelem = tmesh.elements.shape[0]
        tmesh.nnode = tmesh.points.shape[0]

        if compute_boundary_info:
            if tmesh.element_type == "hex" or tmesh.element_type == "tet":
                tmesh.GetBoundaryFaces()
                tmesh.GetBoundaryEdges()
            elif tmesh.element_type == "quad" or tmesh.element_type == "tri":
                tmesh.GetBoundaryEdges()

        if solution is not None:
            if self.nelem != solution.shape[0]:
                solution = solution[unnodes,...]
            else:
                if solution.ndim == 1:
                    solution = solution[elements]
                else:
                    solution = solution[elements,...]
            return tmesh, solution

        # MAKE MESH DATA CONTIGUOUS
        tmesh.ChangeType()

        return tmesh


    def ConvertToLinearMesh(self):
        """Convert a high order mesh to linear mesh.
            This is different from GetLinearMesh in that it converts a
            high order mesh to linear mesh by tessellation i.e. the number of
            points in the mesh do not change
        """

        self.__do_essential_memebers_exist__()
        p = self.InferPolynomialDegree()

        if self.element_type == "hex":
            if p!=2:
                raise NotImplementedError("Converting to linear mesh for hexahedral mesh with p/q>2 not implemented yet")

        lmesh = Mesh()
        elements = np.copy(self.elements)

        if self.element_type == "hex":

            a1 = [ 0,  8, 10,  9, 13, 17, 19, 18]
            a2 = [13, 17, 19, 18,  4, 22, 24, 23]
            a3 = [ 8,  1, 11, 10, 17, 14, 20, 19]
            a4 = [17, 14, 20, 19, 22,  5, 25, 24]
            a5 = [ 9, 10, 12,  3, 18, 19, 21, 16]
            a6 = [18, 19, 21, 16, 23, 24, 26,  7]
            a7 = [10, 11,  2, 12, 19, 20, 15, 21]
            a8 = [19, 20, 15, 21, 24, 25,  6, 26]

            lmesh.elements = np.concatenate(
               (elements[:,a1],
                elements[:,a2],
                elements[:,a3],
                elements[:,a4],
                elements[:,a5],
                elements[:,a6],
                elements[:,a7],
                elements[:,a8]
                ))

        elif self.element_type == "tet":

            from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet
            from scipy.spatial import Delaunay

            # BUILD DELAUNAY TRIANGULATION OF REFERENCE ELEMENTS
            gpoints = FeketePointsTet(p-1)
            Tfunc = Delaunay(gpoints)
            simplex = Tfunc.simplices.copy()

            lmesh.elements = np.zeros((1,4))
            for i in range(Tfunc.nsimplex):
                lmesh.elements = np.concatenate((lmesh.elements,elements[:,simplex[i,:]]))
            lmesh.elements = lmesh.elements[1:,:]

        else:
            raise NotImplementedError("Converting to linear mesh with not implemented yet")


        lmesh.elements = np.ascontiguousarray(lmesh.elements,dtype=np.int64)
        lmesh.points = np.copy(self.points)
        lmesh.degree = 1
        lmesh.element_type = self.element_type
        lmesh.nelem = lmesh.elements.shape[0]
        lmesh.nnode = lmesh.points.shape[0]
        lmesh.GetBoundaryFaces()
        lmesh.GetBoundaryEdges()

        return lmesh


    def ConvertTrisToQuads(self):
        """Converts a tri mesh to a quad mesh through refinement/splitting.
            This is a simpler version of the the Blossom-quad algorithm implemented in gmsh"""

        self.__do_memebers_exist__()
        if self.element_type == "quad":
            return
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
        #     quad0 = np.concatenate((self.points[self.elements[elem,0],:][None,:],mid0[elem,:][None,:],
                    # median[elem,:][None,:],mid2[elem,:][None,:]),axis=0)
        #     quad1 = np.concatenate((self.points[self.elements[elem,1],:][None,:],mid1[elem,:][None,:],
                    # median[elem,:][None,:],mid0[elem,:][None,:]),axis=0)
        #     quad2 = np.concatenate((self.points[self.elements[elem,2],:][None,:],mid2[elem,:][None,:],
                    # median[elem,:][None,:],mid1[elem,:][None,:]),axis=0)
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

        print("Triangular to quadrilateral mesh conversion took", time() - tconv, "seconds")


    def ConvertTetsToHexes(self):
        """Converts a tet mesh to a hex mesh through refinement/splitting
        """

        self.__do_memebers_exist__()
        if self.element_type == "hex":
            return
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

        print("Tetrahedral to hexahedral mesh conversion took", time() - tconv, "seconds")





    def ConvertQuadsToTris(self):
        """Converts a quad mesh to a tri mesh through refinement/splitting

            NOTE: If only linear elements are required conversion of quads to tris
            can be done using Delauney triangularation. The following implementation
            takes care of high order elements as well
        """

        self.__do_memebers_exist__()
        if self.element_type == "tri":
            return
        assert self.element_type == "quad"

        tconv = time()

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

        print("Quadrilateral to triangular mesh conversion took", time() - tconv, "seconds")


    def ConvertHexesToTets(self):
        """Converts a hex mesh to a tet mesh through refinement/splitting

            A hexahedron can be split into 5 or 6 tetrahedrons and there are
            many possible configuration without the edges/faces intersecting each
            other. This method splits a hex into 6 tets.

            Note that in principle, this splitting produces non-conformal meshes
        """

        self.__do_memebers_exist__()
        if self.element_type == "tet":
            return
        assert self.element_type == "hex"

        tconv = time()

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

        print("Hexahedral to tetrahedral mesh conversion took", time() - tconv, "seconds")


    def NodeArranger(self,C=None):
        """Calls NodeArrangment"""

        assert self.element_type is not None

        if C is None:
            C = self.InferPolynomialDegree() - 1

        if self.element_type == "hex":
            return NodeArrangementHex(C)
        elif self.element_type == "tet":
            return NodeArrangementTet(C)
        elif self.element_type == "quad":
            return NodeArrangementQuad(C)
        elif self.element_type == "tri":
            return NodeArrangementTri(C)
        elif self.element_type == "line":
            return NodeArrangementLine(C)
        else:
            raise ValueError("Element type not understood")



    def CreateDummyLowerDimensionalMesh(self):
        """Create a dummy lower dimensional mesh that would have some specific mesh attributes at least.
            The objective is that the lower dimensional mesh should have the same element type as the
            boundary faces/edges of the actual mesh and be the same order"""


        sys.stdout = open(os.devnull, "w")
        p = self.InferPolynomialDegree()
        mesh = Mesh()
        if self.element_type == "tet":
            mesh.Rectangle(nx=1,ny=1, element_type="tri")
            mesh.GetHighOrderMesh(p=p)
        elif self.element_type == "hex":
            mesh.Rectangle(nx=1,ny=1, element_type="quad")
            mesh.GetHighOrderMesh(p=p)
        elif self.element_type == "tri" or self.element_type == "quad":
            mesh.Line(n=1, p=p)
        sys.stdout = sys.__stdout__

        return mesh


    def CreateDummyUpperDimensionalMesh(self):
        """Create a dummy upper dimensional mesh that would have some specific mesh attributes at least.
            The objective is that the upper dimensional mesh should have its bounary element type the same as
            the element type of actual mesh and be the same order. For 1D (line) elements a quad mesh is generated"""


        sys.stdout = open(os.devnull, "w")
        p = self.InferPolynomialDegree()
        mesh = Mesh()
        if self.element_type == "tri":
            mesh.Parallelepiped(nx=1,ny=1,nz=1, element_type="tet")
            mesh.GetHighOrderMesh(p=p)
        elif self.element_type == "quad":
            mesh.Parallelepiped(nx=1,ny=1,nz=1, element_type="hex")
            mesh.GetHighOrderMesh(p=p)
        elif self.element_type == "line":
            mesh.Rectangle(nx=1,ny=1, element_type="quad")
            mesh.GetHighOrderMesh(p=p)
        sys.stdout = sys.__stdout__

        return mesh


    def CreateDummy3DMeshfrom2DMesh(self):
        """Create a dummy 3D mesh from the surfaces of 2D mesh. No volume elements are generated
            This is used for plotting and generating curvilinear elements for surface mesh using florence
        """

        sys.stdout = open(os.devnull, "w")

        p = self.InferPolynomialDegree()
        mm = Mesh()
        if self.element_type == "quad":
            mm.element_type = "hex"
            mm.elements = np.zeros((1,int((p+1)**3))).astype(np.int64)
        elif self.element_type == "tri":
            mm.element_type = "tet"
            mm.elements = np.zeros((1,int((p+1)*(p+2)*(p+3)/6))).astype(np.int64)
        mm.edges = np.zeros((1,p+1)).astype(np.int64)
        mm.nelem = 1
        mm.points = np.copy(self.points)
        mm.faces = np.copy(self.elements)
        mm.boundary_face_to_element = np.zeros((mm.faces.shape[0],2)).astype(np.int64)
        mm.boundary_face_to_element[:,0] = 1

        sys.stdout = sys.__stdout__

        return mm


    def MakeCoordinates3D(self):
        """Change the coordinates/points of the mesh to 3D by appending
            a zero Z-axis
        """

        self.points = np.concatenate((self.points, np.zeros((self.points.shape[0],1)) ), axis=1)
        self.points = np.ascontiguousarray(self.points)


    def SwapAxis(self, axis0, axis1):
        """Swap mesh axis axis0 with axis1, i.e. swap XYZ coordinates
        """

        axis0 = int(axis0)
        axis1 = int(axis1)

        self.points[:,[axis0,axis1]] = self.points[:,[axis1,axis0]]



    def __add__(self, other):
        """Add self with other without modifying self. Hybrid meshes not supported"""
        mesh = deepcopy(self)
        mesh.MergeWith(other)
        return mesh

    def __iadd__(self, other):
        """Add self with other. Hybrid meshes not supported"""
        self.MergeWith(other)
        return self


    def __lt__(self,other):
        """If mesh1 < mesh2 it means that mesh1 can be fit in mesh2
        """
        self_bounds = self.Bounds
        ndim = self.InferSpatialDimension()

        if isinstance(other,Mesh):
            other_bounds = other.Bounds
            mins = (self_bounds[0,:] > other_bounds[0,:]).all()
            maxs = (self_bounds[1,:] < other_bounds[1,:]).all()
            return mins and maxs
        elif isinstance(other,np.ndarray):
            # Otherwise check if an element is within a given bounds
            assert other.shape == (2,ndim)
            mins = (self_bounds[0,:] > other[0,:]).all()
            maxs = (self_bounds[1,:] < other[1,:]).all()
            return mins and maxs
        else:
            raise ValueError("Cannot compare mesh with {}".format(type(other)))

    def __le__(self,other):
        """If mesh1 <= mesh2 it means that mesh1 can be fit in mesh2
        """
        self_bounds = self.Bounds
        ndim = self.InferSpatialDimension()

        if isinstance(other,Mesh):
            other_bounds = other.Bounds
            mins = (self_bounds[0,:] >= other_bounds[0,:]).all()
            maxs = (self_bounds[1,:] <= other_bounds[1,:]).all()
            return mins and maxs
        elif isinstance(other,np.ndarray):
            # Otherwise check if an element is within a given bounds
            assert other.shape == (2,ndim)
            mins = (self_bounds[0,:] >= other[0,:]).all()
            maxs = (self_bounds[1,:] <= other[1,:]).all()
            return mins and maxs
        else:
            raise ValueError("Cannot compare mesh with {}".format(type(other)))

    def __gt__(self,other):
        """If mesh1 > mesh2 it means that mesh2 can be fit in mesh1
        """
        self_bounds = self.Bounds
        ndim = self.InferSpatialDimension()

        if isinstance(other,Mesh):
            other_bounds = other.Bounds
            mins = (self_bounds[0,:] < other_bounds[0,:]).all()
            maxs = (self_bounds[1,:] > other_bounds[1,:]).all()
            return mins and maxs
        elif isinstance(other,np.ndarray):
            # Otherwise check if an element is within a given bounds
            assert other.shape == (2,ndim)
            mins = (self_bounds[0,:] < other[0,:]).all()
            maxs = (self_bounds[1,:] > other[1,:]).all()
            return mins and maxs
        else:
            raise ValueError("Cannot compare mesh with {}".format(type(other)))

    def __ge__(self,other):
        """If mesh1 >= mesh2 it means that mesh2 can be fit in mesh1
        """
        self_bounds = self.Bounds
        ndim = self.InferSpatialDimension()

        if isinstance(other,Mesh):
            other_bounds = other.Bounds
            mins = (self_bounds[0,:] <= other_bounds[0,:]).all()
            maxs = (self_bounds[1,:] >= other_bounds[1,:]).all()
            return mins and maxs
        elif isinstance(other,np.ndarray):
            # Otherwise check if an element is within a given bounds
            assert other.shape == (2,ndim)
            mins = (self_bounds[0,:] <= other[0,:]).all()
            maxs = (self_bounds[1,:] >= other[1,:]).all()
            return mins and maxs
        else:
            raise ValueError("Cannot compare mesh with {}".format(type(other)))



    def __do_memebers_exist__(self):
        """Check if fundamental members exist"""
        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None
        if self.element_type == "tri" or self.element_type == "quad":
            assert self.edges is not None
        ndim = self.InferSpatialDimension()
        if self.element_type == "tet" or self.element_type == "hex":
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