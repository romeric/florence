import numpy as np
from numpy import einsum
from Florence.Tensor import *


class LegendreTransform(object):

    def __init__(self, material_internal_energy=None, material_enthalpy=None,
        newton_raphson_tolerance=1e-12):
        self.material_internal_energy = material_internal_energy
        self.material_enthalpy = material_enthalpy
        self.newton_raphson_tolerance = newton_raphson_tolerance
        self.newton_raphson_convergence = []
        self.newton_raphson_max_iter = 20

    def SetIternalEnergy(self, energy):
        self.material_internal_energy = energy

    def SetEnthalpy(self, enthalpy):
        self.material_enthalpy = enthalpy

    def InternalEnergyToEnthalpy(self,W_dielectric, W_coupling_tensor, W_elasticity, in_voigt=True):
        """Converts the directional derivatives of the free energy of a system to its electric enthalpy.

            in_voigt=True       operates in Voigt form inputs
            in_voigt=False      operates in indicial form inputs


        Note that the coupling tensor should be symmetric with respect to the first two indices.
        Irrespective of the input option (in_voigt), the output is always in Voigt form"""


        # DIELECTRIC TENSOR REMAINS THE SAME IN VOIGT AND INDICIAL NOTATION
        H_dielectric = - np.linalg.inv(W_dielectric)

        # COMPUTE THE CONSTITUTIVE TENSORS OF ENTHALPY BASED ON THOSE ON INTERNAL ENERGY
        if in_voigt:
            # IF GIVEN IN VOIGT FORM
            H_coupling_tensor =  - np.dot(H_dielectric,W_coupling_tensor.T)
            H_elasticity = W_elasticity - np.dot(W_coupling_tensor,H_coupling_tensor)
            # H_elasticity = W_elasticity + np.dot(W_coupling_tensor,H_coupling_tensor) # not right
            H_coupling_tensor = H_coupling_tensor.T

        else: 
            # IF GIVEN IN INDICIAL FORM
            # H_coupling_tensor = - einsum('mi,jkm->jki',H_dielectric,W_coupling_tensor)
            # H_elasticity = Voigt(W_elasticity - einsum('ijm,klm->ijkl',W_coupling_tensor,H_coupling_tensor),1)

            H_coupling_tensor = - einsum('ij,kli->klj',H_dielectric,W_coupling_tensor)
            # H_coupling_tensor = - einsum('im,kli->klm',H_dielectric,W_coupling_tensor)
            # H_elasticity = Voigt(W_elasticity - einsum('ijk,klm->ijlm',W_coupling_tensor,einsum('kij',H_coupling_tensor)),1)
            H_elasticity = Voigt(W_elasticity - einsum('ijk,klm->ijlm',W_coupling_tensor,einsum('kji',H_coupling_tensor)),1)
            # H_elasticity = Voigt(W_elasticity - einsum('ijk,klm',W_coupling_tensor,einsum('kji',H_coupling_tensor)),1)

            # H_coupling_tensor = - einsum('mi,jkm->ijk',H_dielectric,W_coupling_tensor)
            # H_elasticity = Voigt(W_elasticity - einsum('ijm,mkl->ijkl',W_coupling_tensor,H_coupling_tensor),1)
            

            # W_coupling_tensor_Ts = einsum('kij',W_coupling_tensor) 
            # # H_coupling_tensor = -einsum('ij,jkl',H_dielectric,W_coupling_tensor_Ts)
            # # H_coupling_tensor_Ts = einsum('kij',H_coupling_tensor)
            # H_coupling_tensor = -einsum('ij,jkl',H_dielectric,W_coupling_tensor) #
            # H_elasticity = Voigt(W_elasticity - einsum('kij,klm',W_coupling_tensor_Ts,H_coupling_tensor), 1)

            # xx = einsum('kji',H_coupling_tensor)
            # print(xx[0,:,:])
            # print(H_coupling_tensor[:,:,0])
            # print(W_coupling_tensor[:,:,0])
            # print("\n")

            H_coupling_tensor = Voigt(H_coupling_tensor,1)


        return H_dielectric, H_coupling_tensor, H_elasticity


    def GetElectricDisplacement(self, material, StrainTensors, ElectricFieldx, elem=0, gcounter=0):
        """Given electric field and permittivity, computes electric displacement iteratively"""

        norm = np.linalg.norm
        ElectricFieldx = ElectricFieldx.reshape(material.ndim,1)

        # if np.allclose(norm(ElectricFieldx),0):
        #     # BE WARNED THAT THIS MAY NOT ALWAYS BE THE CASE
        #     return np.zeros((material.ndim,1))

        # D = np.copy(ElectricFieldx)
        D = np.zeros((material.ndim,1))
        Residual = -ElectricFieldx
        self.newton_raphson_convergence = []
        self.iter = 0

        norm_forces = norm(ElectricFieldx)
        if np.isclose(norm_forces,0.):
            norm_forces = 1e-14 
        
        # GET INITIAL DIELECTRIC TENSOR    
        material.Hessian(StrainTensors, D, elem, gcounter)

        while np.abs(norm(Residual)/norm_forces) > self.newton_raphson_tolerance:

            # SOLVE THE SYSTEM AND GET ITERATIVE D (deltaD)
            deltaD = np.linalg.solve(material.dielectric_tensor,-Residual)
            # UPDATE ELECTRIC DISPLACEMENT
            D += deltaD
            # UPDATE DIELECTRIC TENSOR 
            material.Hessian(StrainTensors, D, elem, gcounter)
            # RECOMPUTE RESIDUAL
            Residual = np.dot(material.dielectric_tensor,D) - ElectricFieldx
            # STORE CONVERGENCE RESULT
            self.newton_raphson_convergence.append(np.abs(norm(Residual)/norm_forces))
            self.iter += 1

            if self.iter > self.newton_raphson_max_iter:
                raise ValueError('Quadrature point based Newton-Raphson did not converge')

        return D













# def FreeEnergy2Enthalpy(W_Permittivity,W_CoupledTensor,W_Elasticity,opt=0):
#     # Converts the directional derivatives of the free energy of a system to its electric enthalpy.
#     # opt=0, operates in Voigt form inputs
#     # opt=1, operates in indicial form inputs
#     # NOTE THAT THE COUPLED TENSOR SHOULD BE SYMMETRIC WITH RESPECT TO THE LAST TWO INDICES

#     # Irrespective of the input option (opt), the output is always in Voigt form

#     # Permittivity is the same in Voigt and index formats
#     Inverse = np.linalg.inv(W_Permittivity)
#     H_Permittivity = -Inverse

#     if opt==0:
    
#         # H_CoupledTensor = np.dot(Inverse,W_CoupledTensor)
#         # H_Elasticity = W_Elasticity - np.dot(W_CoupledTensor.T,H_CoupledTensor)

#         H_CoupledTensor = np.dot(Inverse,W_CoupledTensor.T)
#         H_Elasticity = W_Elasticity - np.dot(W_CoupledTensor,H_CoupledTensor)
        
#         H_CoupledTensor = H_CoupledTensor.T

#     elif opt==1:
#         # Using Einstein summation (using numpy einsum call)
#         d = np.einsum
#         # Computing directional derivatives of the enthalpy
#         W_CoupledTensor_Ts = d('kij',W_CoupledTensor) 
#         H_CoupledTensor = d('ij,jkl',Inverse,W_CoupledTensor_Ts)
#         H_CoupledTensor_Ts = d('kij',H_CoupledTensor)
#         H_CoupledTensor = d('ij,jkl',Inverse,W_CoupledTensor) #
#         # H_Elasticity = W_Elasticity - Voigt( d('ijk,klm',W_CoupledTensor,H_CoupledTensor) )
#         # H_Elasticity = W_Elasticity - Voigt( d('ijk,klm',W_CoupledTensor_Ts,H_CoupledTensor) )

#         # H_Elasticity = W_Elasticity - Voigt( d('kij,klm',W_CoupledTensor,H_CoupledTensor_Ts) )

#         H_Elasticity = W_Elasticity - Voigt( d('kij,klm',W_CoupledTensor_Ts,H_CoupledTensor) )  #



#         H_CoupledTensor = Voigt(H_CoupledTensor,1)



#     return H_Permittivity, H_CoupledTensor, H_Elasticity



################

# import numpy as np
# import numpy.linalg as la 
# import scipy.linalg as sla 

# def LG_NewtonRaphson(PermittivityW, ElectricFieldx):    
#     # Given electric field and permittivity, computes electric displacement

#     ndim = ElectricFieldx.shape[0]
#     if np.allclose(la.norm(ElectricFieldx),0):
#         # BE WARNED THAT THIS MAY NOT ALWAYS BE THE CASE
#         D = np.zeros((ndim,1))
#     else:
#         # Newton-Raphson scheme to find electric displacement from the free energy
#         tolerance = 1e-13
#         D = np.zeros((ndim,1))
#         # D = np.copy(ElectricFieldx)
#         Residual = -ElectricFieldx
#         ResidualNorm = []

#         while np.abs(la.norm(Residual)/la.norm(ElectricFieldx)) > tolerance:

#             # Update the hessian - depending on the model, extra arguments needs to be passed
#             # PermittivityW = (1.0/varepsilon_1)*d2

#             deltaD = sla.solve(PermittivityW,-Residual)
#             # Update electric displacement
#             D += deltaD
#             # Find residual (first term is equivalent to internal traction)
#             Residual = np.dot(PermittivityW,D) - ElectricFieldx
#             # Save internal tractions
#             ResidualNorm = np.append(ResidualNorm,np.abs(la.norm(Residual)/la.norm(ElectricFieldx)))

#         # print ResidualNorm

#     return D

