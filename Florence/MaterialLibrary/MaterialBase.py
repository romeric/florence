import numpy as np
from Florence.Utils import insensitive
from warnings import warn

# BASE CLASS FOR ALL MATERIAL MODELS - SHOULD NOT BE USED DIRECTLY 
class Material(object):
    """Base class for all material models"""

    def __init__(self, mtype, ndim, energy_type="internal_energy", 
        lame_parameter_1=None, lame_parameter_2=None, poissons_ratio=None, youngs_modulus=None,
        shear_modulus=None, transverse_iso_youngs_modulus=None, transverse_iso_shear_modulus=None,
        bulk_modulus=None, density=None, permittivity=None, permeability=None, **kwargs):


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

        if self.mtype == 'LinearModel' or \
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

        if self.H_Voigt is not None:
            self.H_VoigtSize = self.H_Voigt.shape[0]

        self.is_compressible = True
        self.is_incompressible = False
        self.is_nearly_incompressible = False

        self.is_anisotropic = False
        self.is_transversely_isotropic = False
        self.is_nonisotropic = False
        self.anisotropic_orientations = None




    def SetFibresOrientation(self,anisotropic_orientations):
        self.anisotropic_orientations = anisotropic_orientations


    def GetFibresOrientation(self, mesh, plot=False):
        """Convenience function for computing anisotropic orientations of fibres
            in a transversely isotropic material.
            The orientation is computed based on the popular concept of re-inforced composites
            where the elements at the boundary the fibre is perpendicular to boundary edge"""
            
        if self.ndim == 2:

            edge_elements = mesh.GetElementsWithBoundaryEdgesTri()

            self.anisotropic_orientations = np.zeros((mesh.nelem,self.ndim),dtype=np.float64)
            for iedge in range(edge_elements.shape[0]):
                coords = mesh.points[mesh.edges[iedge,:],:]
                min_x = min(coords[0,0],coords[1,0])
                dist = (coords[0,0:]-coords[1,:])/np.linalg.norm(coords[0,0:]-coords[1,:])

                if min_x != coords[0,0]:
                    dist *= -1 

                self.anisotropic_orientations[edge_elements[iedge],:] = dist

            for i in range(mesh.nelem):
                if self.anisotropic_orientations[i,0]==0. and self.anisotropic_orientations[i,1]==0:
                    self.anisotropic_orientations[i,0] = -1. 


            Xs,Ys = [],[]
            for i in range(mesh.nelem):
                x_avg = np.sum(mesh.points[mesh.elements[i,:],0])/mesh.points[mesh.elements[i,:],0].shape[0]
                y_avg = np.sum(mesh.points[mesh.elements[i,:],1])/mesh.points[mesh.elements[i,:],1].shape[0]

                Xs=np.append(Xs,x_avg)
                Ys=np.append(Ys,y_avg)


            if plot:
                import matplotlib.pyplot as plt
                q = plt.quiver(Xs, Ys, self.anisotropic_orientations[:,0], self.anisotropic_orientations[:,1], 
                           color='Teal', 
                           headlength=5,width=0.004)

                plt.triplot(mesh.points[:,0],mesh.points[:,1], mesh.elements[:,:3],color='k')
                plt.axis('equal')
                plt.show()

        elif self.ndim == 3:
            self.anisotropic_orientations = np.zeros((mesh.nelem,self.ndim))
            self.anisotropic_orientations[:,0] = 1.0


    def Linearise(self,invariant_to_linearise):
        """Give an invariant in the form of a dictionary in order to linearise it.

            input:
                invariant_to_linearise:         [dict] must contain the following keys:
                                                    invariant [str] the invariant to linearise
                                                    cofficient [str] a material coefficient
                                                    kinematics [str] for instance deformation gradient tensor F
                                                    constants [str] constants like kronecker delta d_ij


        >>> material = Material("MooneyRivlin",3)
        >>> material.Linearise({'invariant':'uC:I','coefficient':'u','kinematics':'C','constants':'I'})
        Cauchy stress:      2*u*I
        Spatial Hessian:        0

        """

        delta = u'\u03B4'
        delta = delta.encode('utf-8')

        if "C:I" in invariant_to_linearise['invariant']:
            coefficient = invariant_to_linearise['invariant'].split("C:I")[0]
            print "Cauchy stress:\t\t","2*"+coefficient+"*I"
            print "Spatial Hessian:\t\t","0"

        if "F:F" in invariant_to_linearise['invariant']:
            coefficient = invariant_to_linearise['invariant'].split("F:F")[0]
            print "Cauchy stress:\t\t","2*"+coefficient+"*I"
            print "Spatial Hessian:\t\t","0"

        if "H:H" in invariant_to_linearise['invariant']:
            coefficient = invariant_to_linearise['invariant'].split("H:H")[0]
            print "Cauchy stress:\t\t\t",coefficient+"*b*(b x I)", "or", coefficient+"*b*(b.T - trace(b)*I)"
            # CHECK THIS
            print "Spatial Hessian:\t\t",\
                coefficient+"*(b_ij*"+delta+"_kl+b_kl*"+delta+\
                "_ij+trace(b)*("+delta+"_ik*"+delta+"_jl+trace(b)*"+delta+"_il*"+delta+"_jk))"
                

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






