from __future__ import division
import os, warnings
from time import time
import numpy as np 
from scipy.io import loadmat, savemat
from Florence.Tensor import makezero, itemfreq, unique2d, in2d
from vtk_writer import write_vtu
from NormalDistance import NormalDistance
try:
    import meshpy.triangle as triangle
    has_meshpy = True
except ImportError:
    has_meshpy = False
from SalomeMeshReader import ReadMesh
from HigherOrderMeshing import *
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

    def __init__(self):
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
        self.element_type = None

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
        else:
            bounds = np.array([[np.min(self.points[:,0]),
                        np.min(self.points[:,1])],
                        [np.max(self.points[:,0]),
                        np.max(self.points[:,1])]])
            makezero(bounds)
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
        all_edges = np.concatenate((self.elements[:,:2],self.elements[:,[1,2]],
                             self.elements[:,[2,0]]),axis=0)
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

        # Cython solution
        # faces = remove_duplicates(faces)
        # Much faster than the cython solution
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

        if self.faces is None or self.faces is list:
            raise AttributeError('Tetrahedral edges cannot be computed independent of tetrahedral faces. Compute faces first')

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

        if self.element_type != "tet" and self.element_type != "tri":
            raise NotImplementedError("Computing edge lengths for", self.element_type, "is not implemented yet")

        lengths = None
        if which_edges == 'boundary':
            if self.faces is None:
                if self.element_type == "tri":
                    self.GetBoundaryEdgesTri()
                elif self.element_type == "tet":
                    self.GetBoundaryEdgesTet()

            if self.element_type == "tri" or self.element_type == "tet":
                # THE ORDERING OF NODES FOR EDGES IS FOR TRIS AND TETS ONLY
                edge_coords = self.points[self.edges[:,:2],:]
                lengths = np.linalg.norm(edge_coords[:,1,:] - edge_coords[:,0,:],axis=1)

        elif which_edges == 'all':
            if self.all_faces is None:
                if self.element_type == "tri":
                    self.GetEdgesTri()
                elif self.element_type == "tet":
                    self.GetEdgesTet()

            if self.element_type == "tri" or self.element_type == "tet":
                # THE ORDERING OF NODES FOR EDGES IS FOR TRIS AND TETS ONLY
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
        else:
            raise NotImplementedError("Computing areas for", self.element_type, "elements not implemented yet")

        if with_sign is False:
            if self.element_type == "tri":
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
            raise ValueError("Computing volumes for 2D mesh is not possible")
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

        if self.element_type != "tet" and self.element_type != "tri":
            raise NotImplementedError("Computing aspect ratio of ", self.element_type, "is not implemented yet")

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

        elif algorithm == 'face_based':
            raise NotImplementedError("Face/area based aspect ratio is not implemented yet")

        return aspect_ratio


    def Median(self):
        """Computes median of the elements tri, tet, quad, hex based on the interpolation function

            retruns:
                median:             [ndarray] of median of elements
                bases_at_median:    [1D array] of bases at median            
        """

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        median = None

        if self.element_type == "tri":
            from Florence.FunctionSpace import Tri
            from Florence.QuadratureRules.FeketePointsTri import FeketePointsTri

            middle_point_isoparametric = FeketePointsTri(2)[6] # check
            if not np.isclose(sum(middle_point_isoparametric),-0.6666666):
                raise ValueError("Median of triangle does not match [-0.3333,-0.3333]. "
                    "Did you change your nodal spacing or interpolation functions?")

            # C = self.InferPolynomialDegree() - 1
            hpBases = Tri.hpNodal.hpBases
            bases_for_middle_point = hpBases(0,middle_point_isoparametric[0],
                middle_point_isoparametric[1])[0]

            median = np.einsum('ijk,j',self.points[self.elements[:,:3],:],bases_for_middle_point) 

        elif self.element_type == "tet":
            from Florence.FunctionSpace import Tet
            from Florence.QuadratureRules.FeketePointsTet import FeketePointsTet

            middle_point_isoparametric = FeketePointsTet(3)[21]
            if not np.isclose(sum(middle_point_isoparametric),-1.5):
                raise ValueError("Median of tetrahedral does not match [-0.5,-0.5,-0.5]. "
                    "Did you change your nodal spacing or interpolation functions?")

            # C = self.InferPolynomialDegree() - 1
            hpBases = Tet.hpNodal.hpBases
            bases_for_middle_point = hpBases(0,middle_point_isoparametric[0],
                middle_point_isoparametric[1],middle_point_isoparametric[2])[0]

            median = np.einsum('ijk,j',self.points[self.elements[:,:4],:],bases_for_middle_point) 

        return median, bases_for_middle_point
  


    def CheckNodeNumbering(self,change_order_to='retain'):
        """Checks for node numbering order of the imported mesh. Mesh can be tri or tet

        input:

            change_order_to:            [str] {'clockwise','anti-clockwise','retain'} changes the order to clockwise, 
                                        anti-clockwise or retains the numbering order - default is 'retain' 

        output: 

            original_order:             [str] {'clockwise','anti-clockwise','retain'} returns the original numbering order"""


        assert self.elements is not None
        assert self.points is not None

        # CHECK IF IT IS LINEAR MESH
        quantity = np.array([])
        if self.element_type == "tri":
            assert self.elements.shape[1]==3
            quantity = self.Areas(with_sign=True)
        elif self.element_type == "tet":
            assert self.elements.shape[1]==4
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
            At most a triangle can have all its four edges on the boundary.

        output: 

            edge_elements:              [1D array] array containing elements which have face
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

        _,idx = unique2d(all_edges,consider_sort=True,order=False, return_index=True)
        edge_elements = np.zeros((self.all_faces.shape[0],2),dtype=np.int64)

        edge_elements[:,0] = idx % self.elements.shape[0]
        edge_elements[:,1] = idx // self.elements.shape[0]

        self.edge_to_element = edge_elements
        return self.edge_to_element


    def GetElementsWithBoundaryEdgesTri(self):
        """Finds elements which have edges on the boundary.
            At most a triangle can have all its four edges on the boundary.

        output: 

            edge_elements:              [2D array] array containing elements which have face
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
        elif self.reader_type is 'UniformHollowCircle':
            # self.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=4,ncirc=12)
            # self.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=True,nrad=7,ncirc=7) # isotropic
            self.UniformHollowCircle(inner_radius=0.5,outer_radius=2.,isotropic=False,nrad=7,ncirc=7)
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
        # else:
            # print("Did not read anything")
            # return

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
                if "elements" in DictOutput[key] or "edge" in DictOutput[key] \
                or "face" in DictOutput[key]:
                    setattr(self, key, np.ascontiguousarray(value).astype(np.uint64))
                else:
                    setattr(self, key, np.ascontiguousarray(value))
            else:
                setattr(self, key, value)


        # CUSTOM READ - OLD
        # self.elements = np.ascontiguousarray(DictOutput['elements']).astype(np.uint64)
        # self.points = np.ascontiguousarray(DictOutput['points'])
        # self.nelem = self.elements.shape[0]
        # # self.element_type = str(DictOutput['element_type'][0])
        # self.element_type = element_type
        # self.edges = np.ascontiguousarray(DictOutput['edges']).astype(np.uint64)
        # if self.element_type == "tet":
        #     self.faces = np.ascontiguousarray(DictOutput['faces']).astype(np.uint64)

        # self.all_faces = np.ascontiguousarray(DictOutput['all_faces'])
        # self.all_edges = np.ascontiguousarray(DictOutput['all_edges'])
        # self.face_to_element = np.ascontiguousarray(DictOutput['face_to_element'])
        # # self.edge_to_element = np.ascontiguousarray(DictOutput['edge_to_element'])
        # self.boundary_face_to_element = np.ascontiguousarray(DictOutput['boundary_face_to_element'])
        # self.boundary_edge_to_element = np.ascontiguousarray(DictOutput['boundary_edge_to_element'])

        # Tri
        # self.elements = np.ascontiguousarray(DictOutput['elements'])
        # self.points = np.ascontiguousarray(DictOutput['points'])
        # self.nelem = self.elements.shape[0]
        # self.element_type = str(DictOutput['element_type'][0])
        # self.edges = np.ascontiguousarray(DictOutput['edges'])
        # self.all_edges = np.ascontiguousarray(DictOutput['all_edges'])

        # # self.edge_to_element = np.ascontiguousarray(DictOutput['edge_to_element'])
        # # self.boundary_edge_to_element = np.ascontiguousarray(DictOutput['boundary_edge_to_element'])




    def SimplePlot(self,to_plot='faces',color=None,save=False,filename=None,figure=None,show_plot=True):
        """Simple mesh plot

            to_plot:        [str] only for 3D. 'faces' to plot only boundary faces
                            or 'all_faces' to plot all faces
            """

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        import matplotlib.pyplot as plt 
        if self.element_type == "tri":
            fig = plt.figure()
            plt.triplot(self.points[:,0],self.points[:,1], self.elements[:,:3],color='k')
            plt.axis("equal")

        elif self.element_type == "tet":
            if self.faces is None and self.all_faces is None:
                raise ValueError('Mesh faces not available. Compute them first')
            # from mpl_toolkits.mplot3d import Axes3D
            # fig = plt.figure()

            # FOR PLOTTING ELEMENTS
            # for elem in range(self.elements.shape[0]):
            #   coords = self.points[self.elements[elem,:],:]
            #   plt.gca(projection='3d')
            #   plt.plot(coords[:,0],coords[:,1],coords[:,2],'-bo')

            # FOR PLOTTING ONLY BOUNDARY FACES - MATPLOTLIB SOLUTION
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

            # FOR PLOTTING ONLY BOUNDARY FACES - MAYAVI.MLAB SOLUTION
            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab

            if to_plot == 'all_faces':
                faces = self.all_faces
                if self.all_faces is None:
                    raise ValueError("Boundary faces not available")
            else:
                faces = self.faces
                if self.faces is None:
                    raise ValueError("Faces not available")


            if figure is None:
                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

            if color is None:
                color=(197/255.,241/255.,197/255.)

            mlab.triangular_mesh(self.points[:,0],self.points[:,1],
                self.points[:,2],self.faces[:,:3],color=color)
            radius = 1e-00
            mlab.triangular_mesh(self.points[:,0],self.points[:,1],self.points[:,2], faces[:,:3],
                line_width=radius,tube_radius=radius,color=(0,0,0),
                representation='wireframe')

            # svpoints = self.points[np.unique(self.faces),:]
            # mlab.points3d(svpoints[:,0],svpoints[:,1],svpoints[:,2],color=(0,0,0),mode='sphere',scale_factor=0.005)

            if show_plot:
                mlab.show()

        else:
            raise NotImplementedError("SimplePlot for "+self.element_type+" not implemented yet")

        if save:
            if filename is None:
                raise KeyError('File name not given. Supply one')
            else:
                if filename.split(".")[-1] == filename:
                    filename += ".eps"
                plt.savefig(filename,format="eps",dpi=300)

        plt.show()



    def PlotMeshNumbering(self):
        """Plots element and node numbers on top of the triangular mesh"""

        import matplotlib.pyplot as plt

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

        elif self.element_type == "tet":

            import matplotlib as mpl
            import os
            os.environ['ETS_TOOLKIT'] = 'qt4'
            from mayavi import mlab

            figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(800,600))
            view = mlab.view()
            figure.scene.disable_render = True

            color = mpl.colors.hex2color('#F88379')

            linewidth = 10.
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
            #     # text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=2)
            #     # if i==0:
            #     #     text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=500)
            #     # else:
            #     #     text_obj.position = self.points[i,:]
            #     #     text_obj.text = str(i)
            #     #     text_obj.scale = [2,2,2]

            #     if self.points[i,2] == 0:
            #         text_obj = mlab.text3d(self.points[i,0],self.points[i,1],self.points[i,2],str(i),color=(0,0,0.),scale=0.5)

            for i in range(self.faces.shape[0]):
                if i==3:# or i==441:
                    for j in self.faces[i,:]:
                        text_obj = mlab.text3d(self.points[j,0],self.points[j,1],self.points[j,2],str(j),color=(0,0,0.),scale=10.5)


            figure.scene.disable_render = False

            # mlab.view(*view)
            mlab.show()



    def WriteVTK(self,*args,**kwargs):
        """Write mesh/results to vtu"""

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
                # CHANGE NUMBERING ORDER FOR PARAVIEW
                # para_arange = [0,4,1,6,2,5,7,8,9,3]
                # self.elements = self.elements[:,para_arange]    # NOTE: CHANGES MESH ELEMENT ORDERING
                cellflag = 24
        elif self.element_type == 'hex':
            cellflag = 12
            if self.elements.shape[1] == 20:
                cellflag = 25

        # CHECK IF THE OUTPUT FILE NAME IS SPECIFIED 
        FNAME = False
        for i in kwargs.items():
            if 'fname' in i:
                FNAME = True
                fname = i
        for i in range(len(args)):
            if isinstance(args[i],str):
                FNAME = True 
                fname = args[i]
        if not FNAME:
            # IF NOT WRITE TO THE TOP LEVEL DIRECTORY
            pathvtu = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..'))
            print pathvtu
            warnings.warn('No name given to the vtu output file. I am going to write one at the top level directory')
            fname = pathvtu+str('/output.vtu')
            write_vtu(Verts=self.points, Cells={cellflag:self.elements},fname=fname,**kwargs)
        else:
                write_vtu(Verts=self.points, Cells={cellflag:self.elements},**kwargs)


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


    def Rectangle(self,lower_left_point=(0,0),upper_right_point=(2,1),nx=5,ny=5):
        """Create a triangular mesh of on rectangle"""

        if self.elements is not None and self.points is not None:
            self.__reset__()

        if (lower_left_point[0] > upper_right_point[0]) or \
            (lower_left_point[1] > upper_right_point[1]):
            raise ValueError("Incorrect coordinate for lower left and upper right vertices")


        from scipy.spatial import Delaunay

        x=np.linspace(lower_left_point[0],upper_right_point[0],nx)
        y=np.linspace(lower_left_point[1],upper_right_point[1],ny)

        X,Y = np.meshgrid(x,y)
        coordinates = np.dstack((X.ravel(),Y.ravel()))[0,:,:]

        tri_func = Delaunay(coordinates)
        self.element_type = "tri"
        self.elements = tri_func.simplices
        self.nelem = self.elements.shape[0] 
        self.points = tri_func.points
        self.GetBoundaryEdgesTri()


    def Square(self,lower_left_point=(0,0),side_length=1,n=5):
        """Create a triangular mesh on a square

            input:
                lower_left_corner           [tuple] of lower left vertex of the square
                side_length:                [int] length of side 
                n:                          [int] number of discretisation
            """

        upper_right_point = (side_length+lower_left_point[0],side_length+lower_left_point[1])
        self.Rectangle(lower_left_point=lower_left_point,
            upper_right_point=upper_right_point,nx=n,ny=n)


    def UniformHollowCircle(self,center=(0,0),inner_radius=1.0,outer_radius=2.,element_type='tri',isotropic=True,nrad=5,ncirc=10):
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
        # cost = np.cos(t)
        # sint = np.sin(t)
        for i in range(0,radii.shape[0]):
            # xy[i*t.shape[0]:(i+1)*t.shape[0],0] = radii[i]*cost
            # xy[i*t.shape[0]:(i+1)*t.shape[0],1] = radii[i]*sint
            xy[i*t.shape[0]:(i+1)*t.shape[0],0] = radii[i]*np.cos(t)
            xy[i*t.shape[0]:(i+1)*t.shape[0],1] = radii[i]*np.sin(t)


        # REMOVE DUPLICATES GENERATED BY SIN/COS OF LINSPACE
        xy = xy[np.setdiff1d( np.arange(xy.shape[0]) , np.linspace(t.shape[0]-1,xy.shape[0]-1,radii.shape[0]).astype(int) ),:]

        connec = np.zeros((1,4),dtype=np.int64)
        # connec = ((),)

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
            warn('Edges are not computed as GetBoundaryEdgesQuad is not yet implemented')


        # ASSIGN NODAL COORDINATES
        self.points = xy
        # IF CENTER IS DIFFERENT FROM (0,0)
        self.points[:,0] += center[0]
        self.points[:,1] += center[1]
        # ASSIGN PROPERTIES
        self.nnode = self.points.shape[0]



    def Sphere(self,radius=1.0, points=10):
        """Create a tetrahedral mesh on an sphere

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



    def RemoveElements(self,(x_min,y_min,x_max,y_max),element_removal_criterion="all",keep_boundary_only=False,
            compute_edges=True,compute_faces=True,plot_new_mesh=True):
        """Removes elements with some specified criteria

        input:              
            (x_min,y_min,x_max,y_max)       [tuple of doubles] box selection. Deletes all the elements apart 
                                            from the one within this box 
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

        return lmesh





    def BoundaryEdgesfromPhysicalParametrisation(self,points,facets,mesh_points,mesh_edges):
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
        """TODO: EXLUDE EXTERIOR REGIONS"""

        assert self.element_type is not None
        assert self.elements is not None
        assert self.points is not None

        assert other.element_type is not None
        assert other.elements is not None
        assert other.points is not None

        if self.element_type != other.element_type:
            raise NotImplementedError("Hybrid meshes are not supported")

        from scipy.spatial import Delaunay

        self.points = np.concatenate((self.points,other.points),axis=0)
        self.points = unique2d(self.points,consider_sort=False,order=False)

        tri_func = Delaunay(self.points)
        self.elements = tri_func.simplices
        self.points = tri_func.points

        self.SimplePlot()


    def __reset__(self):
        """Class resetter. Resets all elements of the class
        """

        for i in self.__dict__.keys():
            self.__dict__[i] = None




##############===============================
    def GetBoundaryFacesHigherTet(self):
        """Get high order planar tetrahedral faces given high order element connectivity
            and nodal coordinates

            NOT TESTED

            """

        assert self.elements is not None
        assert self.points is not None
        assert self.nelem is not None

        from Florence.Tensor import itemfreq
        from Florence.QuadratureRules.NodeArrangement import NodeArrangementTet


        # CHANGE TYPE 
        if isinstance(self.elements[0,0],np.uint64):
            self.elements = self.elements.astype(np.int64)
            self.faces = self.faces.astype(np.int64)

        # print self.faces.shape[1]
        if self.faces.shape[1] == 0:
            newMesh = Mesh()
            newMesh.nelem = self.elements.shape[0]
            newMesh.elements = self.elements[:,:4]
            newMesh_nnode = np.max(newMesh.elements)+1 # NO MONKEY PATCHING
            newMesh.points = self.points[:newMesh_nnode]
            newMesh.GetBoundaryFacesTet()
            self.faces = newMesh.faces

        if self.faces.shape[1] > 3:
            raise UserWarning("High order tetrahedral faces seem to be already computed. Do you want me to recompute them?")
            pass


        # INFER POLYNOMIAL DEGREE AND ORDER OF CONTINUITY
        C = self.InferPolynomialDegree()-1
        # GET NUMBER OF COLUMNS FOR HIGHER ORDER FACES
        fsize = int((C+2.)*(C+3.)/2.)
        # GET THE NODE ARRANGEMENT FOR FACES
        face_node_arrangment = NodeArrangementTet(C)[0]

        higher_order_faces = np.zeros((self.faces.shape[0],fsize),dtype = np.int64)
        elements_containing_faces = []
        counter = 0
        # FIND IF THE 3 NODES OF THE LINEAR FACE IS IN THE ELEMENT
        for i in range(self.faces.shape[0]):
            face_nodes = [];  face_cols = []
            for j in range(3):
                # rows, cols = np.where(self.elements[:,:4]==self.faces[i,j])
                rows, cols = whereEQ(self.elements[:,:4],self.faces[i,j])
                # STORE ALL THE OCCURENCES OF CURRENT NODAL FACE IN THE ELEMENT CONNECTIVITY
                face_nodes = np.append(face_nodes,rows)
                face_cols = np.append(face_cols,cols)
            # CHANGE TYPE
            face_nodes = face_nodes.astype(np.int64)
            face_cols = face_cols.astype(np.int64)
            # COUNT THE NUMBER OF OCCURENCES
            occurence_within_an_element = itemfreq(face_nodes)
            # SEE WHICH ELEMENT CONTAINS ALL THE THREE NODES OF THE FACE
            counts = np.where(occurence_within_an_element[:,1]==3)[0]
            if counts.shape[0] != 0:
                # COMPUTE ONLY IF NEEDED LATER ON
                # elements_containing_faces = np.append(elements_containing_faces, 
                #   occurence_within_an_element[counts,0])

                # GET THE POSITIONS FROM CONCATENATION
                inv_uniques = np.unique(face_nodes,return_inverse=True)[1]
                which_connectivity_cols_idx = np.where(inv_uniques==counts)[0]
                # FROM THE POSITIONS GET THE COLUMNS AT WHICH THESE NODES OCCUR IN THE CONNECTIVITY
                which_connectivity_cols =  face_cols[which_connectivity_cols_idx]
                if which_connectivity_cols_idx.shape[0] != 0:
                    # BASED ON THE OCCURENCE, DECIDE WHICH FACE OF THAT ELEMENT WE ARE ON
                    if which_connectivity_cols[0]==0 and which_connectivity_cols[1]==1 \
                        and which_connectivity_cols[2]==2:
                        higher_order_faces[counter,:] = self.elements[
                        occurence_within_an_element[counts,0],face_node_arrangment[0,:]]
                        counter += 1
                    elif which_connectivity_cols[0]==0 and which_connectivity_cols[1]==1 \
                        and which_connectivity_cols[2]==3:
                        higher_order_faces[counter,:] = self.elements[
                        occurence_within_an_element[counts,0],face_node_arrangment[1,:]]
                        counter += 1
                    elif which_connectivity_cols[0]==0 and which_connectivity_cols[1]==2 \
                        and which_connectivity_cols[2]==3:
                        higher_order_faces[counter,:] = self.elements[
                        occurence_within_an_element[counts,0],face_node_arrangment[2,:]]
                        counter += 1
                    elif which_connectivity_cols[0]==1 and which_connectivity_cols[1]==2 \
                        and which_connectivity_cols[2]==3:
                        higher_order_faces[counter,:] = self.elements[
                        occurence_within_an_element[counts,0],face_node_arrangment[3,:]]
                        counter += 1

        # CHECK WITH AN ALREADY EXISTING MESH GENERATOR
        # print np.linalg.norm(np.sort(higher_order_faces,axis=1) - np.sort(self.faces,axis=1))
                
        self.faces = higher_order_faces
        self.ChangeType()