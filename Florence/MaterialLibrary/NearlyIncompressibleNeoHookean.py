from numpy import einsum, asarray, eye
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt


#####################################################################################################
                                    # NEARLY INCOMPRESSIBLE NEOHOOKEAN 
#####################################################################################################


class NearlyIncompressibleNeoHookean(Material):
    """Material model for nearly incompressible neo-Hookean with the following internal energy:

        W(C) = mu/2*J**(-2/3)*(C:I)     # for isochoric part
        U(J) = k/2*(J-1)**2             # for volumetric part

        """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(NearlyIncompressibleNeoHookean, self).__init__(mtype, ndim, **kwargs)

        self.is_nearly_incompressible = True
        self.is_compressible = False

        self.is_transversely_isotropic = False
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False
        

    def Hessian(self, StrainTensors, ElectricFieldx=0, elem=0, gcounter=0):
        """Hessian split into isochoroic and volumetric parts"""
        
        I = StrainTensors['I']
        b = StrainTensors['b'][gcounter]
        J = StrainTensors['J'][gcounter]
        mu = self.mu

        # ISOCHORIC
        H_Voigt = 2*mu*J**(-5./3.)*(1./9.*trace(b)*einsum('ij,kl',I,I) - \
            1./3.*(einsum('ij,kl',b,I) + einsum('ij,kl',I,b)) +\
            1./6.*trace(b)*(einsum('ik,jl',I,I) + einsum('il,jk',I,I)) )
        # VOLUMETRIC
        H_Voigt += self.pressure[elem]*(einsum('ij,kl',I,I) - (einsum('ik,jl',I,I) + einsum('il,jk',I,I))) 

        H_Voigt = Voigt(H_Voigt,1)

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt


    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        mu = self.mu
        stress = mu*J**(-5./3.)*(b - 1./3.*trace(b)*I) 
        stress += self.pressure[elem]*I
            
        return stress












########################################################
# import numpy as np
# from Florence.Tensor import trace
# #####################################################################################################
#                                 # NEARLY INCOMPRESSIBLE NEOHOOKEAN
#                                 # W = mu/2*C:I + k/2*(J-1)**2                               
# #####################################################################################################


# class NearlyIncompressibleNeoHookean(object):
#     """ A nearly incompressible neo-Hookean material model whose energy functional is given by:

#                 W = mu/2*C:I + k/2*(J-1)**2

#             This is an incorrect internal energy for incompressibility as C:I is not pure 
#             deviatoric. It is missing a factor J^{-2/3}


#         """

#     def __init__(self, ndim):
#         super(NearlyIncompressibleNeoHookean, self).__init__()
#         self.ndim = ndim
#         self.nvar = self.ndim

#     def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#         # Using Einstein summation (using numpy einsum call)
#         d = np.einsum

#         # Get material constants (5 in this case)
#         mu = MaterialArgs.mu
#         lamb = MaterialArgs.lamb

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]

#         # Update Lame constants
#         kappa = lamb+2.0*mu/3.0


#         H_Voigt = Voigt( kappa*(2.0*J-1)*d('ij,kl',I,I)-kappa*(J-1)*(d('ik,jl',I,I)+d('il,jk',I,I)) ,1)
        
#         MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#         return H_Voigt



#     def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#         I = StrainTensors['I']
#         J = StrainTensors['J'][gcounter]
#         b = StrainTensors['b'][gcounter]

#         mu = MaterialArgs.mu
#         lamb = MaterialArgs.lamb
#         kappa = lamb+2.0*mu/3.0

#         return 1.0*mu/J*b+(kappa*(J-1.0))*I 


#     def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):
#         ndim = StrainTensors['I'].shape[0]
#         return np.zeros((ndim,1))
