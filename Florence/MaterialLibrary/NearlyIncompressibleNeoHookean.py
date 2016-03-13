import numpy as np
from .MaterialBase import Material
from Florence.Tensor import trace


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

        # INITIALISE STRAIN TENSORS
        from Florence.FiniteElements.ElementalMatrices.KinematicMeasures import KinematicMeasures
        StrainTensors = KinematicMeasures(np.asarray([np.eye(self.ndim,self.ndim)]*2),"Nonlinear")
        self.Hessian(StrainTensors)
        

    def Hessian(self,StrainTensors,elem=0,gcounter=0):
        
        I = StrainTensors['I']
        detF = StrainTensors['J'][gcounter]

        mu2 = self.mu/detF- self.lamb*(detF-1.0)
        lamb2 = self.lamb*(2*detF-1.0) 

        H_Voigt = lamb2*self.vIijIkl+mu2*self.vIikIjl

        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt

    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]

        mu = self.mu
        lamb = self.lamb
            
        return 1.0*mu/J*b + (lamb*(J-1.0)-mu/J)*I












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
