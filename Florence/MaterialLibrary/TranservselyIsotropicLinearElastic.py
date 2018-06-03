import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt, UnVoigt


class TranservselyIsotropicLinearElastic(Material):
    """A linear elastic transervely isotropic material model, with 4 material constants
        and 5 independent components in Hessian.

        Note that this assumes N = [0,0,1] as the direction of anisotropy
    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(TranservselyIsotropicLinearElastic, self).__init__(mtype,ndim,**kwargs)
        # MUST BE SET AFTER CALLING BASE __init__
        self.is_transversely_isotropic = True
        self.is_nonisotropic = True
        self.energy_type = "internal_energy"
        self.nature = "linear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        elif self.ndim==2:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        E = self.E
        E_A = self.E_A
        G_A = self.G_A
        v = self.nu

        H_Voigt = np.array([
                [ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A), (E_A**2*(v - 1))/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [                                                      0,                                                      0,                                       0,                  E/(2*(v + 1)),  0,   0],
                [                                                      0,                                                      0,                                       0,                  0,              G_A, 0],
                [                                                      0,                                                      0,                                       0,                  0,              0, G_A]
            ])

        if self.ndim == 2:
            # CAREFUL WITH THIS SLICING AS SOME MATERIAL CONSTANTS WOULD BE REMOVED.
            # ESSENTIALLY IN PLANE STRAIN ANISOTROPY THE BEHAVIOUR OF MATERIAL
            # PERPENDICULAR TO THE PLANE IS LOST

            H_Voigt = H_Voigt[np.array([2,1,-1])[:,None],[2,1,-1]]


        self.H_VoigtSize = H_Voigt.shape[0]


        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        strain = StrainTensors['strain'][gcounter]
        strain_Voigt = Voigt(strain)

        E = self.E
        E_A = self.E_A
        G_A = self.G_A
        v = self.nu

        H_Voigt = np.array([
                [ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A), (E_A**2*(v - 1))/(2*E*v**2 + E_A*v - E_A),              0,              0,   0],
                [                                                      0,                                                      0,                                       0,                  E/(2*(v + 1)),  0,   0],
                [                                                      0,                                                      0,                                       0,                  0,              G_A, 0],
                [                                                      0,                                                      0,                                       0,                  0,              0, G_A]
            ])

        if self.ndim == 2:
            # CAREFUL WITH THIS SLICING AS SOME MATERIAL CONSTANTS WOULD BE REMOVED.
            # ESSENTIALLY IN PLANE STRAIN ANISOTROPY THE BEHAVIOUR OF MATERIAL
            # PERPENDICULAR TO THE PLANE IS LOST

            H_Voigt = H_Voigt[np.array([2,1,-1])[:,None],[2,1,-1]]

        stress = UnVoigt(np.dot(H_Voigt,strain_Voigt))

        return stress



