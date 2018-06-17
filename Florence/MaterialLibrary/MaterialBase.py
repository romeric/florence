from __future__ import print_function
import numpy as np
from Florence.Utils import insensitive
from warnings import warn

# BASE CLASS FOR ALL MATERIAL MODELS - SHOULD NOT BE USED DIRECTLY
class Material(object):
    """Base class for all material models"""

    def __init__(self, mtype, ndim, energy_type="internal_energy",
        lame_parameter_1=None, lame_parameter_2=None, poissons_ratio=None, youngs_modulus=None,
        shear_modulus=None, transverse_iso_youngs_modulus=None, transverse_iso_shear_modulus=None,
        bulk_modulus=None, density=None, permittivity=None, permeability=None,
        is_compressible=True, is_incompressible=False, is_nearly_incompressible=False,
        is_nonisotropic=True,is_anisotropic=False,is_transversely_isotropic=False, anisotropic_orientations=None,
        **kwargs):


        # SAFETY CHECKS
        if not isinstance(mtype, str):
            raise TypeError("Type of material model should be given as a string")
        if not isinstance(energy_type, str):
            raise TypeError("Material energy can either be 'internal_energy' or 'enthalpy'")

        self.energy_type = energy_type

        # MATERIAL CONSTANTS
        self.mu = lame_parameter_1
        self.lamb = lame_parameter_2
        self.nu = poissons_ratio
        self.E = youngs_modulus
        self.E_A = transverse_iso_youngs_modulus
        self.G_A = transverse_iso_shear_modulus
        self.K = bulk_modulus
        self.rho = density
        # if self.rho is None:
        #     self.rho = 0.0

        self.e = permittivity
        self.u = permeability


        # SET ALL THE OPTIONAL KEYWORD ARGUMENTS
        for i in kwargs.keys():
            if "__" not in i:
                setattr(self,i,kwargs[i])

        self.mtype = mtype
        self.ndim = ndim


        if 'elec' not in insensitive(self.mtype):
            if 'magnet' not in insensitive(self.mtype):
                self.nvar = self.ndim
        elif 'elec' in insensitive(self.mtype) and 'magnet' not in insensitive(self.mtype):
            self.nvar = self.ndim + 1
        elif 'elec' not in insensitive(self.mtype) and 'magnet' in insensitive(self.mtype):
            self.nvar = self.ndim + 1
        elif 'elec' in insensitive(self.mtype) and 'magnet' in insensitive(self.mtype):
            self.nvar = self.ndim + 2
        else:
            self.nvar = self.ndim

        self.H_Voigt = None

        if self.mu is None or self.lamb is None:
            if self.E is not None and self.nu is not None:
                self.GetLameParametersFromYoungsPoisson()
            # else:
            #     warn("You must set the material constants for problem")

        try:
            if self.mtype == 'LinearElastic' or \
                self.mtype == 'IncrementalLinearElastic':

                if self.ndim == 2:
                    self.H_Voigt = self.lamb*np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]]) +\
                     self.mu*np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
                else:
                    block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
                    block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
                    self.H_Voigt = self.lamb*block_1 + self.mu*block_2
            else:
                if self.ndim == 2:
                    self.vIijIkl = np.array([[1.,1.,0.],[1.,1.,0],[0.,0.,0.]])
                    self.vIikIjl = np.array([[2.,0.,0.],[0.,2.,0],[0.,0.,1.]])
                else:
                    block_1 = np.zeros((6,6),dtype=np.float64); block_1[:3,:3] = np.ones((3,3))
                    block_2 = np.eye(6,6); block_2[0,0],block_2[1,1],block_2[2,2]=2.,2.,2.
                    self.vIijIkl = block_1
                    self.vIikIjl = block_2

                I = np.eye(self.ndim,self.ndim)
                self.Iijkl = np.einsum('ij,kl',I,I)
                self.Iikjl = np.einsum('ik,jl',I,I) + np.einsum('il,jk',I,I)

        except TypeError:
            # CATCH ONLY TypeError. OTHER MATERIAL CONSTANT RELATED ERRORS ARE SELF EXPLANATORY
            raise ValueError("Material constants for {} does not seem correct".format(self.mtype))

        if self.H_Voigt is not None:
            self.H_VoigtSize = self.H_Voigt.shape[0]

        self.is_compressible = is_compressible
        self.is_nearly_incompressible = is_nearly_incompressible
        self.is_incompressible = is_incompressible

        self.is_anisotropic = is_anisotropic
        self.is_transversely_isotropic = is_transversely_isotropic
        self.is_nonisotropic = is_nonisotropic
        self.anisotropic_orientations = anisotropic_orientations

        self.has_low_level_dispatcher = False




    def SetFibresOrientation(self,anisotropic_orientations):
        self.anisotropic_orientations = anisotropic_orientations


    def GetFibresOrientation(self, mesh, interior_orientation=None, plot=False):
        """Convenience function for computing anisotropic orientations of fibres
            in a transversely isotropic material.
            The orientation is computed based on the popular concept of reinforced composites
            where for the elements at the boundary, the fibres are perpendicular to the boundary
            edge/face

            input:
                mesh:                           [Mesh]
                interior_orientation:           [1D numpy.array or list] orientation of all interior
                                                fibres. Default is negative X-axis i.e [-1.,0.] for 2D
                                                and [-1.,0.,0.] for 3D
        """

        ndim = mesh.InferSpatialDimension()
        if self.ndim != ndim:
            raise ValueError('Mesh object and material model do not have the same spatial dimension')

        if self.ndim == 2:

            edge_elements = mesh.GetElementsWithBoundaryEdges()

            self.anisotropic_orientations = np.zeros((mesh.nelem,self.ndim),dtype=np.float64)
            for iedge in range(edge_elements.shape[0]):
                coords = mesh.points[mesh.edges[iedge,:],:]
                min_x = min(coords[0,0],coords[1,0])
                dist = (coords[0,0:]-coords[1,:])/np.linalg.norm(coords[0,0:]-coords[1,:])

                if min_x != coords[0,0]:
                    dist *= -1

                self.anisotropic_orientations[edge_elements[iedge],:] = dist

            if interior_orientation is None:
                interior_orientation = [-1.,0.]
            for i in range(mesh.nelem):
                if np.allclose(self.anisotropic_orientations[i,:],0.):
                    self.anisotropic_orientations[i,:] = interior_orientation


            if plot:

                Xs,Ys = [],[]
                for i in range(mesh.nelem):
                    x_avg = np.sum(mesh.points[mesh.elements[i,:],0])/mesh.points[mesh.elements[i,:],0].shape[0]
                    y_avg = np.sum(mesh.points[mesh.elements[i,:],1])/mesh.points[mesh.elements[i,:],1].shape[0]

                    Xs=np.append(Xs,x_avg)
                    Ys=np.append(Ys,y_avg)

                import matplotlib.pyplot as plt
                figure = plt.figure()
                q = plt.quiver(Xs, Ys, self.anisotropic_orientations[:,0],
                    self.anisotropic_orientations[:,1], color='Teal',
                           headlength=5,width=0.004)

                if mesh.element_type == "tri":
                    plt.triplot(mesh.points[:,0],mesh.points[:,1], mesh.elements[:,:3],color='k')
                else:
                    from Florence.MeshGeneration.NodeArrangement import NodeArrangementQuad
                    C = mesh.InferPolynomialDegree() - 1
                    reference_edges = NodeArrangementQuad(C)[0]
                    reference_edges = np.concatenate((reference_edges,reference_edges[:,1,None]),axis=1)
                    reference_edges = np.delete(reference_edges,1,1)

                    all_edge_elements = mesh.GetElementsEdgeNumberingQuad()
                    mesh.GetEdgesQuad()
                    x_edges = np.zeros((C+2,mesh.all_edges.shape[0]))
                    y_edges = np.zeros((C+2,mesh.all_edges.shape[0]))

                    BasesOneD = np.eye(2,2)
                    for iedge in range(mesh.all_edges.shape[0]):
                        ielem = all_edge_elements[iedge,0]
                        edge = mesh.elements[ielem,reference_edges[all_edge_elements[iedge,1],:]]
                        x_edges[:,iedge], y_edges[:,iedge] = mesh.points[edge,:].T

                    plt.plot(x_edges,y_edges,'-k')

                plt.axis('equal')
                plt.axis('off')
                plt.show()

        elif self.ndim == 3:

            face_elements = mesh.GetElementsWithBoundaryFaces()

            self.anisotropic_orientations = np.zeros((mesh.nelem,self.ndim),dtype=np.float64)

            for iface in range(face_elements.shape[0]):
                coords = mesh.points[mesh.faces[iface,:],:]
                min_x = min(coords[0,0], coords[1,0], coords[2,0])

                # ORIENTS THE FIBRE TO ONE OF THE EDGES OF THE FACE
                fibre = (coords[0,:]-coords[1,:])/np.linalg.norm(coords[0,:]-coords[1,:])

                if min_x != coords[0,0]:
                    fibre *= -1

                self.anisotropic_orientations[face_elements[iface],:] = fibre

            if interior_orientation is None:
                interior_orientation = [-1.,0.,0.]
            for i in range(mesh.nelem):
                if np.allclose(self.anisotropic_orientations[i,:],0.):
                    self.anisotropic_orientations[i,:] = interior_orientation

            if plot:

                # all_face_elements = mesh.GetElementsFaceNumbering()
                Xs = np.zeros(mesh.elements.shape[0])
                Ys = np.zeros(mesh.elements.shape[0])
                Zs = np.zeros(mesh.elements.shape[0])

                # divider = mesh.points[mesh.elements[0,:],0].shape[0]
                # for i in range(mesh.nelem):
                #     Xs[i] = np.sum(mesh.points[mesh.elements[i,:],0])/divider
                #     Ys[i] = np.sum(mesh.points[mesh.elements[i,:],1])/divider
                #     Zs[i] = np.sum(mesh.points[mesh.elements[i,:],2])/divider

                divider = mesh.points[mesh.faces[0,:],0].shape[0]
                for i in range(mesh.faces.shape[0]):
                    Xs[face_elements[i,0]] = np.sum(mesh.points[mesh.faces[i,:],0])/divider
                    Ys[face_elements[i,0]] = np.sum(mesh.points[mesh.faces[i,:],1])/divider
                    Zs[face_elements[i,0]] = np.sum(mesh.points[mesh.faces[i,:],2])/divider



                import os
                os.environ['ETS_TOOLKIT'] = 'qt4'
                from mayavi import mlab
                from Florence.PostProcessing import PostProcess

                if mesh.element_type == "tet":
                    tmesh = PostProcess.TessellateTets(mesh,np.zeros_like(mesh.points),
                        interpolation_degree=0)
                elif mesh.element_type == "hex":
                    tmesh = PostProcess.TessellateHexes(mesh,np.zeros_like(mesh.points),
                        interpolation_degree=0)

                figure = mlab.figure(bgcolor=(1,1,1),fgcolor=(1,1,1),size=(1000,800))

                mlab.quiver3d(Xs, Ys, Zs, self.anisotropic_orientations[:,0],
                    self.anisotropic_orientations[:,1], self.anisotropic_orientations[:,2],
                    color=(0.,128./255,128./255),line_width=2)

                src = mlab.pipeline.scalar_scatter(tmesh.x_edges.T.copy().flatten(),
                tmesh.y_edges.T.copy().flatten(), tmesh.z_edges.T.copy().flatten())
                src.mlab_source.dataset.lines = tmesh.connections
                lines = mlab.pipeline.stripper(src)
                h_edges = mlab.pipeline.surface(lines, color = (0,0,0), line_width=2)

                mlab.show()


    def Linearise(self,energy):
        """Linearises a material model, by dispatching invariants to self.LineariseInvariant"""
        pass


    def LineariseInvariant(self,invariant_to_linearise):
        """Give an invariant in the form of a dictionary in order to linearise it.

            input:
                invariant_to_linearise:         [dict] must contain the following keys:
                                                    invariant [str] the invariant to linearise
                                                    cofficient [str] a material coefficient
                                                    kinematics [str] for instance deformation gradient tensor F
                                                    constants [str] constants like kronecker delta d_ij

            Linearisation is always carried out with respect to the right Cauchy-Green tensor (C)


        >>> material = Material("MooneyRivlin",3)
        >>> material.Linearise({'invariant':'uC:I','coefficient':'u','kinematics':'C','constants':'I'})
        Cauchy stress:      2*u*I
        Spatial Hessian:        0

        """

        if not isinstance(invariant_to_linearise,dict):
            raise ValueError('invariant_to_linearise should be a dictionary')

        if 'invariant' not in invariant_to_linearise.keys():
            raise ValueError("invariant_to_linearise should have at least one key named 'invariant' with no spaces")

        strip_invariant = "".join(invariant_to_linearise['invariant'].split())

        if 'coefficient' not in invariant_to_linearise.keys():
            coefficient = ''
            invariant = strip_invariant
        else:
            coefficient = "".join(invariant_to_linearise['coefficient'].split())
            invariant = strip_invariant.split(coefficient)

        if len(invariant) > 1:
            if invariant[0] == '':
                if invariant[1][0] == '*':
                    invariant = invariant[1][1:]
            elif invariant[1] == '':
                if invariant[0][-1] == '*':
                    invariant = invariant[0][:-1]


        delta = u'\u03B4'
        delta = delta.encode('utf-8')

        if "C:I" in invariant or "F:F" in invariant or "trC" in invariant or "II_F" in invariant or "I_C" in invariant:
            cauchy = "2.0/J*"+coefficient+"*I"
            elasticity = "0"

        if "G:I" in invariant or "H:H" in invariant or "trG" in invariant or "II_H" in invariant or "I_G" in invariant:
            cauchy = "2.0/J*"+coefficient+"*(trace(b)*I-b)*b"
            elasticity = "4.0/J*"+coefficient+"*(b_ij*b_kl - b_ikb_jl)"

        if "lnJ" in invariant:
            cauchy = "2.0/J*"+coefficient+"*I"
            elasticity = "4.0/J*"+coefficient+"*"+delta+"_ik*"+delta+"_jl"

        if "(J-1)**2" in invariant:
            cauchy = "2.0*"+coefficient+"*(J-1)*I"
            elasticity = "2.0*"+coefficient+"*(2*J-1)"+delta+"_ij*"+delta+"_jk"+\
                "-4.0*"+coefficient+"*(J-1)"+delta+"_ik*"+delta+"_jl"

        if "NCN" in invariant or "FNFN" in invariant or "II_FN" in invariant:
            cauchy = "2.0/J*"+coefficient+"*(FN)_i(FN)_j"
            elasticity = "0"

        if "NGN" in invariant or "HNHN" in invariant or "II_HN" in invariant:
            cauchy = "2.0/J*"+coefficient+"*( ((HN)_k(HN)_k)*I -(HN)_i(HN)_j )"
            elasticity = "4.0/J*"+coefficient+"*( -"+delta+"_ij*(HN)_k(HN)_l +(HN)_i(HN)_j*"+\
            delta+"_kl" + "(HN)_m(HN)_m*"+delta+"_ij"+delta+"_kl" + "-(HN)_m(HN)_m*"+delta+"_ik"+delta+"_jl" +\
            delta+"_il"+"(HN)_j(HN)_k"+delta+"_jl"+"(HN)_i(HN)_k )"

        if "cauchy" not in locals() or "elasticity" not in locals():
            cauchy = "NIL"
            elasticity = "NIL"
            warn("I could not linearise the invariant %s" % invariant)

        print("Cauchy stress tensor:\t\t\t", cauchy)
        print("Spatial Hessian:\t\t\t\t", elasticity)

        return cauchy, elasticity


    def GetYoungsPoissonsFromLameParameters(self):

        assert self.mu != None
        assert self.lamb != None

        self.E = self.mu*(3.0*self.lamb + 2.*self.mu)/(self.lamb + self.mu)
        self.nu = self.lamb/2.0/(self.lamb + self.mu)

    def GetLameParametersFromYoungsPoisson(self):

        assert self.nu != None
        assert self.E != None

        self.lamb = self.E*self.nu/(1.+self.nu)/(1.-2.0*self.nu)
        self.mu = self.E/2./(1+self.nu)


    @property
    def Types(self):
        """Returns available material types"""
        import os
        pwd = os.path.dirname(os.path.realpath(__file__))
        list_of_materials = os.listdir(pwd)

        list_of_materials = [list_of_materials[i].split(".")[0] for i in range(len(list_of_materials))]
        list_of_materials = list(np.unique(list_of_materials))
        if "__init__" in list_of_materials:
            idx = list_of_materials.index("__init__")
            del list_of_materials[idx]

        return np.asarray(list_of_materials).reshape(-1,1)


    def GetType(self):
        """Get the type of material used"""
        if self.mtype is None:
            raise ValueError("You have not specified a material type. "
                "Call the 'Types' property for a list of available material models")
        return self.mtype


    def SetType(self,mtype):
        """Set the type of material to be used"""
        self.mtype = mtype


    def pprint(self):
        """Pretty print"""

        import pandas
        from copy import deepcopy
        Dict = deepcopy(self.__dict__)
        for key in Dict.keys():
            if Dict[key] is None:
                Dict[key] = np.NAN
            if isinstance(Dict[key],np.ndarray):
                del Dict[key]

        print(pandas.DataFrame(Dict,index=["Available parameters:"]))






