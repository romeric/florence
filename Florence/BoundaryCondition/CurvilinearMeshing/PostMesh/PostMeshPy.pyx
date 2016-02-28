#cython: profile=False
#cython: infer_types=True
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False

from cython import boundscheck, nonecheck, wraparound, profile, double
import numpy as np
cimport numpy as np

from warnings import warn



cdef class PostMeshBasePy:
    """
    PostMesh base class. Provides most of the common functionality for 
    point projection, point inversion and obtaining Dirichlet data.
    This class should not be directly.
    """

    cdef UInteger ndim 
    # USE CYTHONS TYPED MEMORY VIEWS AS NUMPY
    # ARRAYS CANNOT BE DECLARED AT MODULE/CLASS LEVEL
    cdef Real[:,:] boundary_fekete 

    # CREATE A POINTER TO CPP BASE CLASS
    cdef PostMeshBase *baseptr

    def __cinit__(self, bytes py_element_type, UInteger dimension=2):

        self.ndim = dimension
        # CONVERT TO CPP STRING EXPLICITLY
        cdef string cpp_element_type = py_element_type
        # CREATE A NEW CPP OBJECT BY CALLING ITS CONSTRUCTOR
        self.baseptr = new PostMeshBase(cpp_element_type,dimension)
        # CHECK IF THE OBJECT WAS CREATED
        if self.baseptr is NULL:
            raise MemoryError("Could not create an instance of PostMesh")

    def Init(self, bytes py_element_type, UInteger dimension=2):
        # CONVERT TO CPP STRING EXPLICITLY
        cdef string cpp_element_type = py_element_type
        self.baseptr.Init(cpp_element_type, dimension)

    def SetScale(self,Real scale):
        """Scales the mesh to match the scaling of CAD"""
        self.baseptr.SetScale(scale)

    def SetCondition(self,Real radius=1.0e20):
        """Set the condition for projection. This is useful in situations
        when the far-field does not need projection. Everything in the mesh 
        within the 'radius' gets projected
        """
        self.baseptr.SetCondition(radius)

    def SetProjectionPrecision(self, Real precision):
        """Set projection precision. A linear mesh can be inaccurate compared
        to the actual CAD geometry. If the geometrical point and the mesh point
        are within the 'precision' they are treated to be the same point
        """
        self.baseptr.SetProjectionPrecision(precision)

    def SetProjectionCriteria(self, UInteger[:,::1] criteria):
        """Set projection criteria for complex situations when specifying a radius 
        is not enough. 'criteria' is an array containing either 0s or 1s with a 
        dimension of (nfaces/nedges x 1) where nfaces/nedges is number of boundary
        edges/faces
        """
        self.baseptr.SetProjectionCriteria(&criteria[0,0],criteria.shape[0],criteria.shape[1])
    
    def ComputeProjectionCriteria(self):
        """Convenience function for computing projection criteria based on
        radius specifed through SetCondition""" 
        self.baseptr.ComputeProjectionCriteria()

    def SetMeshElements(self,UInteger[:,::1] elements):
        """Set up elements of the linear mesh"""
        self.baseptr.SetMeshElements(&elements[0,0],elements.shape[0],elements.shape[1])
    
    def SetMeshPoints(self,Real[:,::1] points):
        """Set up nodal coordinates of the linear mesh"""
        self.baseptr.SetMeshPoints(&points[0,0],points.shape[0],points.shape[1])
    
    def SetMeshEdges(self,UInteger[:,::1] edges):
        """Set up boundary edges of the linear mesh"""
        self.baseptr.SetMeshEdges(&edges[0,0],edges.shape[0],edges.shape[1])

    def SetMeshFaces(self,UInteger[:,::1] faces):
        """Set up boundary faces of the linear mesh"""
        self.baseptr.SetMeshFaces(&faces[0,0],faces.shape[0],faces.shape[1])

    def ScaleMesh(self):
        """Scale mesh to match the CAD geometry"""
        self.baseptr.ScaleMesh()

    def GetMeshElementType(self):
        cdef string cpp_element_type = self.baseptr.GetMeshElementType()
        cdef bytes py_element_type = cpp_element_type
        return py_element_type

    def SetNodalSpacing(self, Real[:,::1] spacing):
        """Set nodal spacing of high order points in the mesh.
        spacing has to be given in the isoparametric domain of finite element, 
        for instance equally-spaced, Gauss-Lobatto or Fekete point spacing
        """
        self.baseptr.SetNodalSpacing(&spacing[0,0],spacing.shape[0],spacing.shape[1])

    def SetMesh(self,UInteger[:,::1] elements, Real[:,::1] points,
        UInteger[:,::1] edges, UInteger[:,::1] faces, Real[:,::1] spacing, scale_mesh=True):
        """Convenience method for Python interface for setting up the linear mesh

            input:
                elements:               [ndarray] element connectivity of the linear mesh
                points:                 [ndarray] nodal coordinates of the linear mesh
                edges:                  [ndarray] boundary edges of the linear mesh
                faces:                  [ndarray] boundary faces of the linear mesh
                spacing:                [ndarray] nodal spacing of high order points 
                                        in the mesh. spacing has to be given in the 
                                        isoparametric domain of finite element, for 
                                        instance equally-spaced, Gauss-Lobatto or Fekete 
                                        point spacing
                scale_mesh:             [bool] True/False, to scale or not to scale the
                                        mesh based on CAD geometry
                """

        self.baseptr.SetMeshElements(&elements[0,0],elements.shape[0],elements.shape[1])    
        self.baseptr.SetMeshPoints(&points[0,0],points.shape[0],points.shape[1])    
        self.baseptr.SetMeshEdges(&edges[0,0],edges.shape[0],edges.shape[1])
        self.baseptr.SetMeshFaces(&faces[0,0],faces.shape[0],faces.shape[1])
        if scale_mesh:
            self.baseptr.ScaleMesh()
        self.baseptr.SetNodalSpacing(&spacing[0,0],spacing.shape[0],spacing.shape[1])

    def ReadIGES(self, bytes filename):
        """Read IGES files"""
        self.baseptr.ReadIGES(<const char*>filename)

    def ReadSTEP(self, bytes filename):
        """Read STEP files"""
        self.baseptr.ReadSTEP(<const char*>filename)

    @wraparound(True)
    def ReadGeometry(self, bytes filename):
        """Read geometry from IGES or STEP files"""
        suffix = filename.split(".")[-1].upper().lower()
        if suffix == "iges" or suffix == "igs":
            self.baseptr.ReadIGES(<const char*>filename)
        if suffix == "step" or suffix == "stp":
            self.baseptr.ReadSTEP(<const char*>filename)

    def GetGeomVertices(self):
        self.baseptr.GetGeomVertices()
        cdef vector[Real] geom_points = self.baseptr.ObtainGeomVertices()
        cdef np.ndarray geometry_points = np.array(geom_points,copy=False)
        return geometry_points.reshape(int(geometry_points.shape[0]/self.ndim),self.ndim)

    def GetGeomEdges(self):
        """Obtain geometrical curves and topological edges from the imported CAD model"""
        self.baseptr.GetGeomEdges()

    def GetGeomFaces(self):
        """Obtain geometrical surfaces and topological faces from the imported CAD model"""
        self.baseptr.GetGeomFaces()

    @property
    def NbPoints(self):
        """Retruns number of geometrical points"""
        return self.baseptr.NbPoints()

    @property
    def NbCurves(self):
        """Retruns number of geometrical curves"""
        return self.baseptr.NbCurves()

    @property
    def NbSurfaces(self):
        """Retruns number of geometrical surfaces"""
        return self.baseptr.NbSurfaces()

    def SetGeometry(self, bytes filename):
        """Convenience method for Python API for setting up the CAD geometry"""
        self.ReadGeometry(filename)
        self.baseptr.GetGeomVertices()
        cdef vector[Real] geom_points = self.baseptr.ObtainGeomVertices()
        cdef np.ndarray geometry_points = np.array(geom_points,copy=False)
        
        self.baseptr.GetGeomEdges()
        self.baseptr.GetGeomFaces()
        return geometry_points.reshape(int(geometry_points.shape[0]/self.ndim),self.ndim)

    @boundscheck(False)
    def GetDirichletData(self):
        """Obtain Dirichlet boundary condition for higher order nodes of the mesh"""
        cdef: 
            DirichletData struct_to_python = self.baseptr.GetDirichletData()
            np.ndarray[np.int64_t, ndim=2, mode='c'] nodes_dir = np.zeros((struct_to_python.nodes_dir_size,1),dtype=np.int64)
            Integer i
            UInteger j 

        for i in range(struct_to_python.nodes_dir_size):
            nodes_dir[i] = struct_to_python.nodes_dir_out_stl[i]
            
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] displacements_BC = \
            np.zeros((self.baseptr.ndim*struct_to_python.nodes_dir_size,1),dtype=np.float64) 
        for j in range(self.baseptr.ndim*struct_to_python.nodes_dir_size):
            displacements_BC[j] = struct_to_python.displacement_BC_stl[j]

        return nodes_dir, displacements_BC.reshape(struct_to_python.nodes_dir_size,self.baseptr.ndim) 


    def __dealloc__(self):
        if self.baseptr != NULL:
            del self.baseptr




cdef class PostMeshCurvePy(PostMeshBasePy):
    """
    Point projection and inversion onto curves, for 2D finite elements
    such as triangular and quad elements 
    """

    def __cinit__(self, bytes py_element_type, UInteger dimension=2):

        self.ndim = 2
        # CONVERT TO CPP STRING EXPLICITLY
        cdef string cpp_element_type = py_element_type
        # CREATE A CPP PostMeshCurve OBJECT AND CAST IT TO CPP PostMeshBase 
        self.baseptr = <PostMeshBase*> new PostMeshCurve(cpp_element_type,dimension)
    
    def DiscretiseCurves(self, Integer npoints):
        """Discretise geometrical curves. This is only useful for post-processing"""
        cdef:
            vector[vector[Real]] discretised_points
            Integer i
        discretised_points = (<PostMeshCurve*>self.baseptr).DiscretiseCurves(npoints)
        discretised_points_py = []
        for i in range(len(discretised_points)):
            discretised_points_py.append(np.array(discretised_points[i]).reshape(len(discretised_points[0])/3,3))
        return discretised_points_py

    def GetCurvesParameters(self):
        """Get curves isoparametric parameters"""
        (<PostMeshCurve*>self.baseptr).GetCurvesParameters()

    def GetCurvesLengths(self):
        """Get lengths of all geometrical curves"""
        (<PostMeshCurve*>self.baseptr).GetCurvesLengths()
    
    def GetBoundaryPointsOrder(self):
        """Order high order nodes in VTK style"""
        (<PostMeshCurve*>self.baseptr).GetBoundaryPointsOrder()

    def GetGeomPointsOnCorrespondingEdges(self):
        """Find which geometrical points lie on which geometrical curves/topological edges"""
        (<PostMeshCurve*>self.baseptr).GetGeomPointsOnCorrespondingEdges()

    def IdentifyCurvesContainingEdges(self):
        """Identify which geometrical curves contain which mesh edges"""
        (<PostMeshCurve*>self.baseptr).IdentifyCurvesContainingEdges()

    def ProjectMeshOnCurve(self):
        """Project linear mesh nodes on to the true CAD curves"""
        (<PostMeshCurve*>self.baseptr).ProjectMeshOnCurve()

    def RepairDualProjectedParameters(self):
        """Repair/fix projected points, if projection happens to be inaccurate.
        This is typically the case for closed/periodic curves
        """
        (<PostMeshCurve*>self.baseptr).RepairDualProjectedParameters()

    def MeshPointInversionCurve(self):
        """Perform point inversion of high order nodes in the mesh on
        to the true CAD geometry using an orthogonal projection
        """
        (<PostMeshCurve*>self.baseptr).MeshPointInversionCurve()

    def MeshPointInversionCurveArcLength(self):
        """Perform point inversion of high order nodes in the mesh on
        to the true CAD geometry using an arc-length-based projection.
        This is an isometric projection, in that the spacing of nodes
        in the curved mesh remain the same i.e. equally-spaced, Fekete
        etc
        """
        (<PostMeshCurve*>self.baseptr).MeshPointInversionCurveArcLength()

    def PerformPointProjectionInversionCurve(self, str curve_identification_algorithm="projection", 
        str projection_type="arc_length"):
        """Convenience function for Python API. Performs point projection and
            point inversion at one go

            input:
                curve_identification_algorithm          [str] algorithm to use for identifying
                                                        which mesh edges lie on which geometrical 
                                                        curves, either "projection" or "minimisation".
                                                        Note that this input argument can be ignored,
                                                        as it is only kept for API consistency at the 
                                                        moment, as "minimisation" algorithm is not
                                                        implemented for curves 

                projection_type:                        [str] type of prjection to use for projecting
                                                        high order nodes to CAD geometry, should be
                                                        either "arc_length" or "orthogonal".
                                                        The default for 2D meshes is arc_length, as
                                                        it results in better quality high order meshes
        """

        (<PostMeshCurve*>self.baseptr).GetCurvesParameters()
        (<PostMeshCurve*>self.baseptr).GetCurvesLengths()
        (<PostMeshCurve*>self.baseptr).GetBoundaryPointsOrder()
        (<PostMeshCurve*>self.baseptr).GetGeomPointsOnCorrespondingEdges()
        if curve_identification_algorithm != "projection":
            warn("Only projection algorithm is available for curve identification")
        (<PostMeshCurve*>self.baseptr).IdentifyCurvesContainingEdges()
        (<PostMeshCurve*>self.baseptr).ProjectMeshOnCurve()
        (<PostMeshCurve*>self.baseptr).RepairDualProjectedParameters()
        if projection_type == "orthogonal":
            (<PostMeshCurve*>self.baseptr).MeshPointInversionCurve()
        else:
            (<PostMeshCurve*>self.baseptr).MeshPointInversionCurveArcLength()

    def ReturnModifiedMeshPoints(self,Real[:,::1] points):
        """Return modified points in the linear mesh. Needs nodal coordinates
        as input argument
        """
        (<PostMeshCurve*>self.baseptr).ReturnModifiedMeshPoints(&points[0,0])

    def __dealloc__(self):
        # CREATE A TEMPORARY DERIVED CPP OBJECT
        cdef PostMeshCurve *tmpptr
        if self.baseptr != NULL:
            # CAST IT TO BASE
            tmpptr = <PostMeshCurve*>self.baseptr
            # DELETE IT
            del tmpptr
            # SET BASE POINTER TO NULL
            self.baseptr = NULL



cdef class PostMeshSurfacePy(PostMeshBasePy):
    """
    Point projection and inversion onto surfaces and curves 
    (i.e. surface intersections), for 3D finite elements
    such as tetrahedra and hexahedra  
    """

    def __cinit__(self, bytes py_element_type, UInteger dimension=3):
        
        self.ndim = 3
        # CONVERT TO CPP STRING EXPLICITLY
        cdef string cpp_element_type = py_element_type
        self.baseptr = <PostMeshBase*>new PostMeshSurface(cpp_element_type,dimension)

    def GetSurfacesParameters(self):
        """Get surfaces isoparametric parameters""" 
        (<PostMeshSurface*>self.baseptr).GetSurfacesParameters()

    def GetGeomPointsOnCorrespondingFaces(self):
        """Find which geometrical points lie on which geometrical surfaces/topological faces"""
        (<PostMeshSurface*>self.baseptr).GetGeomPointsOnCorrespondingFaces()
    
    def IdentifySurfacesContainingFaces(self):
        """Identify which geometrical surfaces contain which mesh faces, 
        by solving a minimisation problem"""
        (<PostMeshSurface*>self.baseptr).IdentifySurfacesContainingFaces()

    def IdentifyRemainingSurfacesByProjection(self):
        """If identifying which geometrical surfaces contain which mesh faces, 
        fails use proejction to identify the remaining surfaces"""
        (<PostMeshSurface*>self.baseptr).IdentifyRemainingSurfacesByProjection()
    
    def IdentifySurfacesContainingFacesByPureProjection(self):
        """Identify which geometrical surfaces contain which mesh faces, 
        solely by relying on projection"""
        (<PostMeshSurface*>self.baseptr).IdentifySurfacesContainingFacesByPureProjection()

    def IdentifySurfacesIntersections(self):
        """Identify which geometrical surfaces contain which mesh faces"""
        (<PostMeshSurface*>self.baseptr).IdentifySurfacesIntersections()

    def SupplySurfacesContainingFaces(self, Integer[::1] face_to_surface_map, 
        Integer already_mapped=0, 
        str internal_surface_identification_algorithm="minimisation"):
        """Supply which mesh faces lie on which geometrical surfaces, from an external sources. 
        Typically linear mesh generators are able to provide this information. 
        This is due to the face that in some extreme cases OpenCascade might fail to identify
        the right surface that a mesh face lies. 

            input:
                face_to_surface_map:        [1D array] of the same size as number of boundary
                                            faces, each entry corresponding to a geometrical surface
                                            that contains the corresponding mesh boundary face

                already_mapped:             [int] 0 or 1.  Due to different CAD standards 
                                            the external mapping provided for the geometry 
                                            and that geometry read internally might have 
                                            different face numbering, in which case supply
                                            already_map as zero.

                internal_surface_identification_algorithm
                                            [str] either "minimisation" of "projection". 
                                            If the internal and externally supplied numbering of
                                            entities do not match, surfaces will have to be
                                            identified internally to gaurantee, the validity
                                            of external mapping 
        """ 

        cdef Integer caller = 0
        if internal_surface_identification_algorithm != "minimisation":
            caller = 1
        (<PostMeshSurface*>self.baseptr).SupplySurfacesContainingFaces(&face_to_surface_map[0], 
            face_to_surface_map.shape[0], already_mapped, caller)

    def ProjectMeshOnSurface(self):
        """Project linear mesh nodes on to the true CAD curves"""
        (<PostMeshSurface*>self.baseptr).ProjectMeshOnSurface()

    def RepairDualProjectedParameters(self):
        """Repair/fix projected points, if projection happens to be inaccurate.
        This is typically the case for closed/periodic iso-curves
        """
        (<PostMeshSurface*>self.baseptr).RepairDualProjectedParameters()

    def MeshPointInversionSurface(self,Integer project_on_curves, Integer modify_linear_mesh=0):
        """Perform point inversion of high order nodes in the mesh on
        to the true CAD geometry using an orthogonal projection

            input:
                project_on_curves:          [int] 0 or 1, project of surface intersections

                modify_linear_mesh          [int] 0 or 1, modify linear mesh, if it has
                                            inaccuracy in nodal coordinates
        """
        (<PostMeshSurface*>self.baseptr).MeshPointInversionSurface(project_on_curves, modify_linear_mesh)

    def MeshPointInversionSurfaceArcLength(self, Integer project_on_curves, 
        Real orth_tol, Real[:,::1] FEbases):
        """Perform point inversion of high order nodes in the mesh on
        to the true CAD geometry using an arc-length-based projection.
        This is an isometric projection, in that the spacing of nodes
        in the curved mesh remain the same i.e. equally-spaced, Fekete
        etc

            input:
                project_on_curves:          [int] 0 or 1, project of surface intersections

                orth_tol                    [double] 0 or 1, tolerance to fall back to
                                            orthogonal projection if arc length fails.
                                            This is a a tolerance defined in the physical space
                                            such that (L_arc - L_orth)/L_orth < orth_tol, where
                                            L represents the amount of displacement that the 
                                            boundary node undergoes. Typically a value between
                                            0 and 1 should be given

                FEbases                     [2D array] of finite element shape functions 
                                            evaluated at each isoparamertric point. Arc length 
                                            projection works by placing the points of 
                                            isoparametric finite element in the 
                                            isoparametric domain of NURBS
        """
        (<PostMeshSurface*>self.baseptr).MeshPointInversionSurfaceArcLength(project_on_curves,orth_tol,
            &FEbases[0,0],FEbases.shape[0],FEbases.shape[1])

    def PerformPointProjectionInversionSurface(self, str surface_identification_algorithm="minimisation", 
        str projection_type="orthogonal", Integer[::1] face_to_surface_map=None, 
        Integer already_mapped=0, str internal_surface_identification_algorithm="minimisation",
        repair_dual_projection=False, Integer project_on_curves=1, 
        Integer modify_linear_mesh=0, Real orth_tol=1, Real[:,::1] FEbases=None):
        """Convenience function for Python API. Performs point projection and
            point inversion at one go

            input:
                surface_identification_algorithm          
                                            [str] algorithm to use for identifying
                                            which mesh faces lie on which geometrical 
                                            surfaces, either "projection" or "minimisation"
                                            or "supplied" in case it is supplied externally.
                                            Default is minimisation in 3D, unless supplied            

                projection_type:            [str] type of prjection to use for projecting
                                            high order nodes to CAD geometry, should be
                                            either "arc_length" or "orthogonal".
                                            The default for 3D meshes is orhthogonal, as
                                            it is more robust for surfaces

                face_to_surface_map:        [1D array] of the same size as number of boundary
                                            faces, each entry corresponding to a geometrical surface
                                            that contains the corresponding mesh boundary face

                already_mapped:             [int] 0 or 1.  Due to different CAD standards 
                                            the external mapping provided for the geometry 
                                            and that geometry read internally might have 
                                            different face numbering, in which case supply
                                            already_map as zero.

                internal_surface_identification_algorithm:
                                            [str] either "minimisation" of "projection". 
                                            If the internal and externally supplied numbering of
                                            entities do not match, surfaces will have to be
                                            identified internally to gaurantee the validity
                                            of external mapping

                repair_dual_projection:     [bool] True/False to repair/fix projected points, 
                                            if projection happens to be inaccurate.
                                            This is typically the case for closed/periodic 
                                            iso-curves

                project_on_curves:          [int] 0 or 1, project of surface intersections

                modify_linear_mesh          [int] 0 or 1, modify linear mesh, if it has
                                            inaccuracy in nodal coordinates

                orth_tol                    [double] 0 or 1, tolerance to fall back to
                                            orthogonal projection if arc length fails.
                                            This is a a tolerance defined in the physical space
                                            such that (L_arc - L_orth)/L_orth < orth_tol, where
                                            L represents the amount of displacement that the 
                                            boundary node undergoes. Typically a value between
                                            0 and 1 should be given 

                FEbases                     [2D array] of finite element shape functions 
                                            evaluated at each isoparamertric point. Arc length 
                                            projection works by placing the points of 
                                            isoparametric finite element in the 
                                            isoparametric domain of NURBS
        """

        cdef Integer caller = 0
        if internal_surface_identification_algorithm != "minimisation":
            caller = 1

        (<PostMeshSurface*>self.baseptr).GetSurfacesParameters()
        (<PostMeshSurface*>self.baseptr).GetGeomPointsOnCorrespondingFaces()
        if surface_identification_algorithm == "minimisation":
            (<PostMeshSurface*>self.baseptr).IdentifySurfacesContainingFaces()
            (<PostMeshSurface*>self.baseptr).IdentifyRemainingSurfacesByProjection()
        elif surface_identification_algorithm == "projection":
            (<PostMeshSurface*>self.baseptr).IdentifySurfacesContainingFacesByPureProjection()
        elif surface_identification_algorithm == "supplied":
            if face_to_surface_map is not None:
                (<PostMeshSurface*>self.baseptr).SupplySurfacesContainingFaces(&face_to_surface_map[0], 
            face_to_surface_map.shape[0], already_mapped, caller)
            else:
                raise ValueError("A mapping between mesh faces and geometrical surfaces was not supplied")


        if projection_type == "orthogonal":
            (<PostMeshCurve*>self.baseptr).MeshPointInversionCurve()
        else:
            (<PostMeshSurface*>self.baseptr).ProjectMeshOnSurface()
            if repair_dual_projection:
                (<PostMeshSurface*>self.baseptr).RepairDualProjectedParameters()
            if FEbases is None:
                raise ValueError("Finite element bases functions were not supplied "
                    "for arc length based projection")
            (<PostMeshSurface*>self.baseptr).MeshPointInversionSurfaceArcLength(project_on_curves,orth_tol,
            &FEbases[0,0],FEbases.shape[0],FEbases.shape[1])

    def ReturnModifiedMeshPoints(self,Real[:,::1] points):
        (<PostMeshSurface*>self.baseptr).ReturnModifiedMeshPoints(&points[0,0])

    @boundscheck(False)
    def GetMeshFacesOnPlanarSurfaces(self):
        """Get boundary mesh faces that lie on planar CAD surfaces. 
        These faces typically require in-plane translation of nodes, 
        that is separate 2D analyses"""
        # TRANSPOSE AND COPY 
        return np.array((<PostMeshSurface*>self.baseptr).GetMeshFacesOnPlanarSurfaces(),
            copy=False).T.copy(order="c")

    @boundscheck(False)
    def GetDirichletFaces(self):
        """Get boundary faces that are being projected on to the CAD surfaces"""
        cdef vector[Integer] dirichlet_faces = (<PostMeshSurface*>self.baseptr).GetDirichletFaces()
        return np.array(dirichlet_faces,
            copy=False).reshape(dirichlet_faces.size()/(self.ndim+1),self.ndim+1)

    def __dealloc__(self):
        # CREATE A TEMPORARY DERIVED CPP OBJECT
        cdef PostMeshSurface *tmpptr
        if self.baseptr != NULL:
            # CAST IT TO BASE
            tmpptr = <PostMeshSurface*>self.baseptr
            # DELETE IT
            del tmpptr
            # SET BASE POINTER TO NULL
            self.baseptr = NULL