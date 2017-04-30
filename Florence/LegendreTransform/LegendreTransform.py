import numpy as np
from numpy import einsum
from Florence.Tensor import *


class LegendreTransform(object):

    def __init__(self, material_internal_energy=None, material_enthalpy=None,
        newton_raphson_tolerance=1e-09):
        self.material_internal_energy = material_internal_energy
        self.material_enthalpy = material_enthalpy
        self.newton_raphson_tolerance = newton_raphson_tolerance
        self.newton_raphson_convergence = []
        self.newton_raphson_max_iter = 20

    def SetIternalEnergy(self, energy):
        self.material_internal_energy = energy

    def SetEnthalpy(self, enthalpy):
        self.material_enthalpy = enthalpy

    def InternalEnergyToEnthalpy(self,W_dielectric, W_coupling_tensor, W_elasticity):
        """Converts the directional derivatives of the internal energy of a system to its electric enthalpy.
        Note that the transformation needs to be performed first and then the tensors need to be transformed
        to Voigt form
        """

        # DIELECTRIC TENSOR REMAINS THE SAME IN VOIGT AND INDICIAL NOTATION
        H_dielectric = - np.linalg.inv(W_dielectric)

        # COMPUTE THE CONSTITUTIVE TENSORS OF ENTHALPY BASED ON THOSE ON INTERNAL ENERGY
        H_coupling_tensor = - einsum('ij,kli->klj',H_dielectric,W_coupling_tensor)
        H_elasticity = Voigt(W_elasticity - einsum('ijk,klm->ijlm',W_coupling_tensor,einsum('kji',H_coupling_tensor)),1)

        # print(H_coupling_tensor[:,:,0])
        # print(W_coupling_tensor[:,:,0])
        # print("\n")
        H_coupling_tensor = Voigt(H_coupling_tensor,1)


        return H_dielectric, H_coupling_tensor, H_elasticity


    def GetElectricDisplacement(self, material, StrainTensors, ElectricFieldx, elem=0, gcounter=0):
        """Given electric field and permittivity, computes electric displacement iteratively"""

        norm = np.linalg.norm
        ElectricFieldx = ElectricFieldx.reshape(material.ndim,1)

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







