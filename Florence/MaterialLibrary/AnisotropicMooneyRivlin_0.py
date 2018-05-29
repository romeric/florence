import numpy as np
from numpy import einsum
from .MaterialBase import Material
from Florence.Tensor import trace, Voigt

#####################################################################################################
                                # Anisotropic MooneyRivlin Model
#####################################################################################################


class AnisotropicMooneyRivlin_0(Material):
    """A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
        The energy is given by:

            W(C) =  gamma * ( alpha*(C:I) + beta*(G:I) ) + 
                    eta*(1-alpha)*( (N C N)**2 + N G N) - ut*J + lambda/2*(J-1)**2

            ut = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta  # for the stress to be 
                zero at the origin

        the parameter "gamma" controls the amount of anisotropy and the vector N(ndim,1) is 
        the direction of anisotropy

    """

    def __init__(self, ndim, **kwargs):
        mtype = type(self).__name__
        super(AnisotropicMooneyRivlin_0, self).__init__(mtype, ndim, **kwargs)
        self.nvar = self.ndim
        self.is_transversely_isotropic = True
        self.energy_type = "internal_energy"
        self.nature = "nonlinear"
        self.fields = "mechanics"

        if self.ndim==3:
            self.H_VoigtSize = 6
        else:
            self.H_VoigtSize = 3

        # LOW LEVEL DISPATCHER
        self.has_low_level_dispatcher = False


    def Hessian(self,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

        mu = self.mu
        lamb = self.lamb

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)[:,0]
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        # FIX GAMMA
        gamma = 0.5
        # gamma = 1.0
        alpha = mu/2./gamma 
        beta  = mu/2./gamma
        eta   = mu/3.
        ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
        lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


        H_Voigt = 2.*gamma*beta/J* ( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
                4.*(1-gamma)*eta/J * einsum('i,j,k,l',FN,FN,FN,FN) + \
                4.*(1-gamma)*eta/J * ( innerHN * einsum('ij,kl',I,I) - \
                0.5*innerHN * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) - \
                einsum('ij,k,l',I,HN,HN) - einsum('i,j,kl',HN,HN,I) ) + \
                2.*(1-gamma)*eta/J * ( einsum('il,j,k',I,HN,HN) + einsum('jl,i,k',I,HN,HN) + \
                einsum('ik,j,l',I,HN,HN) + einsum('jk,i,l',I,HN,HN) ) - \
                ut*einsum('ij,kl',I,I) + ut * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
                lamb*(2.*J-1.)*einsum('ij,kl',I,I) - lamb*(J-1.) * ( einsum('ik,jl',I,I) + einsum('il,jk',I,I) )

        H_Voigt = Voigt(H_Voigt ,1)
        
        self.H_VoigtSize = H_Voigt.shape[0]

        return H_Voigt



    def CauchyStress(self,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

        I = StrainTensors['I']
        J = StrainTensors['J'][gcounter]
        b = StrainTensors['b'][gcounter]
        F = StrainTensors['F'][gcounter]
        H = J*np.linalg.inv(F).T
        N = self.anisotropic_orientations[elem][:,None]
        FN = np.dot(F,N)
        HN = np.dot(H,N)[:,0]
        innerHN = einsum('i,i',HN,HN)
        outerHN = einsum('i,j',HN,HN)

        mu = self.mu
        lamb = self.lamb

        # FIX GAMMA
        gamma = 0.5
        # gamma = 1.0
        alpha = mu/2./gamma 
        beta  = mu/2./gamma
        eta   = mu/3.
        ut    = 2.*gamma*(alpha+2.0*beta) + 2.*(1. - gamma)*eta
        lamb  = lamb + 2.*gamma*alpha - 2*(1.- gamma)*eta


        if self.ndim == 3:
            trb = trace(b)
        elif self.ndim == 2:
            trb = trace(b) + 1


        stress = 2.*gamma*alpha/J*b + 2.*gamma*beta/J*(trb*b - np.dot(b,b)) + \
                 2.*(1.- gamma)*eta/J*np.dot(FN.T,FN)[0][0]*np.dot(FN,FN.T) + \
                 2.*(1.- gamma)*eta/J*(innerHN*I - outerHN) - \
                 ut*I + lamb*(J-1.)*I


        # print stress
        return stress

    def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
        ndim = StrainTensors['I'].shape[0]
        return np.zeros((ndim,1))





# ANISOTROPIC MODEL WORKING
#############################################################################
# import numpy as np
# from Core.Supplementary.Tensors import *
# from numpy import einsum

# #####################################################################################################
#                               # Anisotropic MooneyRivlin Model
# #####################################################################################################


# class AnisotropicMooneyRivlin(object):
#   """A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
#       The energy is given by:

#           W(C) =  alpha*(u1/2*(C:I) +u2/2*(G:I)) + 
#                   u3/2(1-alpha)*(N C N + N G N) - ut lnJ + lambda/2*(J-1)**2

#           ut = alpha*u1+2*alpha*u2+u3*(1-alpha) # for the stress to be zero at the origin

#       the parameter "alpha" controls the amount of anisotropy and the vector N(ndim,1) is 
#       the direction of anisotropy

#   """

#   def __init__(self, ndim):
#       super(AnisotropicMooneyRivlin, self).__init__()
#       self.ndim = ndim

#   def Get(self):
#       self.nvar = self.ndim
#       self.modelname = 'AnisotropicMooneyRivlin'
#       return self.nvar, self.modelname

#   def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#       # Get material constants (5 in this case)
#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]
#       F = StrainTensors['F'][gcounter]
#       H = J*np.linalg.inv(F).T
#       N = np.array([-1.,0.]).reshape(2,1)
#       # N = np.array([0.,0.]).reshape(2,1)
#       # V = np.dot(H,N)
#       FN = np.dot(F,N)[:,0]
#       HN = np.dot(H,N)[:,0]
#       # HN = HN[:,0]
#       # innerVV = np.dot(V.T,V)[0][0]
#       innerHN = einsum('i,i',HN,HN)
#       # outerVV = np.dot(V,V.T)
#       outerHN = einsum('i,j',HN,HN)
#       # V = V[:,0]

#       # FIX ALPHA
#       alpha = 0.5
#       u1=mu/2./alpha 
#       u2=mu/2./alpha
#       u3 = mu/3.
#       ut = alpha*(u1+2.0*u2) + (1-alpha)*u3
#       lamb = lamb - alpha*u1 - (1.-alpha)*u3

#       # H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
#       #   ut*einsum('ij,kl',I,I) + ut*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#       #   lamb*(2*J-1)*einsum('ij,kl',I,I) -lamb*(J-1)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#       #   2.*u3*(1.-alpha)/J*(innerHN*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerHN) - einsum('ij,kl',outerHN,I) ) - \
#       #   u3*(1.-alpha)/J*innerHN*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#       #   u3*(1.-alpha)/J* ( einsum('il,j,k',I,HN,HN) + einsum('ik,j,l',I,HN,HN) ) + \
#       #   u3*(1.-alpha)/J* ( einsum('jl,i,k',I,HN,HN) + einsum('jk,i,l',I,HN,HN) ) ##


#       H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
#           ut*einsum('ij,kl',I,I) + ut*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#           lamb*(2*J-1)*einsum('ij,kl',I,I) -lamb*(J-1)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#           2.*u3*(1.-alpha)/J*(innerHN*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerHN) - einsum('ij,kl',outerHN,I) ) - \
#           u3*(1.-alpha)/J*innerHN*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#           u3*(1.-alpha)/J* ( einsum('il,j,k',I,HN,HN) + einsum('ik,j,l',I,HN,HN) ) + \
#           u3*(1.-alpha)/J* ( einsum('jl,i,k',I,HN,HN) + einsum('jk,i,l',I,HN,HN) ) + \
#           2.*u3*(1.-alpha)/J* einsum('i,j,k,l',FN,FN,FN,FN)


#       H_Voigt = Voigt(H_Voigt ,1)
        
#       MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#       return H_Voigt



#   def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]
#       F = StrainTensors['F'][gcounter]
#       H = J*np.linalg.inv(F).T
#       N = np.array([-1.,0.]).reshape(2,1)
#       # N = np.array([0.,0.]).reshape(2,1)
#       FN = np.dot(F,N)
#       HN = np.dot(H,N)[:,0]
#       innerHN = einsum('i,i',HN,HN)
#       outerHN = einsum('i,j',HN,HN)

#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       # FIX ALPHA
#       alpha = 0.5
#       u1=mu/2./alpha 
#       u2=mu/2./alpha
#       u3 = mu/3.
#       ut = alpha*(u1+2.0*u2) + (1-alpha)*u3
#       lamb = lamb - alpha*u1 - (1.-alpha)*u3

#       # stress = alpha*u1/J*b + alpha*u2/J*(trace(b)*b - np.dot(b,b)) - ut*I + lamb*(J-1)*I + \
#       #       u3*(1.-alpha)/J*np.dot(FN,FN.T) + \
#       #       u3*(1.-alpha)/J*(innerHN*I - outerHN)

#       if I.shape[0]==2:
#           trb = trace(b)+1
#       elif I.shape[0]==3:
#           trb = trace(b)

#       # stress = alpha*u1/J*b + alpha*u2/J*(trb*b - np.dot(b,b)) - ut*I + lamb*(J-1)*I + \
#       #       u3*(1.-alpha)/J*np.dot(FN,FN.T) + \
#       #       u3*(1.-alpha)/J*(innerHN*I - outerHN)

#       stress = alpha*u1/J*b + alpha*u2/J*(trb*b - np.dot(b,b)) - ut*I + lamb*(J-1)*I + \
#               u3*(1.-alpha)/J*np.dot(FN.T,FN)[0][0]*np.dot(FN,FN.T) + \
#               u3*(1.-alpha)/J*(innerHN*I - outerHN)


#       # print stress
#       return stress

#   def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
#       ndim = StrainTensors['I'].shape[0]
#       return np.zeros((ndim,1))














# FULL ISOTROPIC VERSION WITH ALPHA
###################################
# import numpy as np
# from Core.Supplementary.Tensors import *
# from numpy import einsum

# #####################################################################################################
#                               # Anisotropic MooneyRivlin Model
# #####################################################################################################


# class AnisotropicMooneyRivlin(object):
#   """A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
#       The energy is given by:

#           W(C) =  alpha*(u1/2*(C:I) +u2/2*(G:I)) + 
#                   u3/2(1-alpha)*(N C N + N G N) - ut lnJ + lambda/2*(J-1)**2

#           ut = alpha*u1+2*alpha*u2+u3*(1-alpha) # for the stress to be zero at the origin

#       the parameter "alpha" controls the amount of anisotropy and the vector N(ndim,1) is 
#       the direction of anisotropy

#   """

#   def __init__(self, ndim):
#       super(AnisotropicMooneyRivlin, self).__init__()
#       self.ndim = ndim

#   def Get(self):
#       self.nvar = self.ndim
#       self.modelname = 'AnisotropicMooneyRivlin'
#       return self.nvar, self.modelname

#   def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#       # Get material constants (5 in this case)
#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]

#       # FIX ALPHA
#       alpha = 0.5
#       u1=mu/2./alpha 
#       u2=mu/2./alpha
#       ut = alpha*(u1+2.0*u2)
#       lamb = lamb - alpha*u1

#       H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
#           ut*einsum('ij,kl',I,I) + ut*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + \
#           lamb*(2*J-1)*einsum('ij,kl',I,I) -lamb*(J-1)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) )

#       H_Voigt = Voigt(H_Voigt ,1)
        
#       MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#       return H_Voigt



#   def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]

#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       # FIX ALPHA
#       alpha = 0.5
#       u1=mu/2./alpha 
#       u2=mu/2./alpha
#       ut = alpha*(u1+2.0*u2)
#       lamb = lamb - alpha*u1

#       stress = alpha*u1/J*b + alpha*u2/J*(trace(b)*b - np.dot(b,b)) - ut*I + lamb*(J-1)*I


#       # print stress
#       return stress

#   def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
#       ndim = StrainTensors['I'].shape[0]
#       return np.zeros((ndim,1))
####################################################################################################










# import numpy as np
# from Core.Supplementary.Tensors import *
# from numpy import einsum

# #####################################################################################################
#                               # Anisotropic MooneyRivlin Model
# #####################################################################################################


# class AnisotropicMooneyRivlin(object):
#   """A compressible transervely isotropic model with the isotropic part being Mooney-Rivlin
#       The energy is given by:

#           W(C) =  alpha*(u1/2*(C:I) +u2/2*(G:I)) + 
#                   u3/2(1-alpha)*(N C N + N G N) - ut lnJ + lambda/2*(J-1)**2

#           ut = alpha*u1+2*alpha*u2+u3*(1-alpha) # for the stress to be zero at the origin

#       the parameter "alpha" controls the amount of anisotropy and the vector N(ndim,1) is 
#       the direction of anisotropy

#   """

#   def __init__(self, ndim):
#       super(AnisotropicMooneyRivlin, self).__init__()
#       self.ndim = ndim

#   def Get(self):
#       self.nvar = self.ndim
#       self.modelname = 'AnisotropicMooneyRivlin'
#       return self.nvar, self.modelname

#   def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

#       # Get material constants (5 in this case)
#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]
        # F = StrainTensors['F'][gcounter]
        # H = J*np.linalg.inv(F).T
        # # g = np.dot(H,H.T)
        # N = np.array([-1.,0.]).reshape(2,1)
        # # N = np.array([0.,0.]).reshape(2,1)
        # V = np.dot(H,N)
        # innerVV = np.dot(V.T,V)[0][0]
        # outerVV = np.dot(V,V.T)
        # V = V[:,0]


#       # FIX ALPHA
#       alpha = 0.5

#       u1=mu/2. 
#       u2=mu/2.
#       # u3=mu/2.
#       u3 = 0.
#       ut = alpha*(u1+2.0*u2)+(1.-alpha)*u3

        
#       # H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
#       #   2.0*(1.-alpha)*u3/J*(innerVV*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerVV) - einsum('ij,kl',outerVV,I) - \
#       #   0.5*innerVV*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + einsum('ik,j,l',I,V,V) + einsum('i,k,jl',V,V,I) ) + \
#       #   ut/J*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + lamb*(2.0*J-1.0)*einsum('ij,kl',I,I) - \
#       #   lamb*(J-1.)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) )

#       # H_Voigt = u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
#       #   ut/J*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + lamb*(2.0*J-1.0)*einsum('ij,kl',I,I) - \
#       #   lamb*(J-1.)*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) )

#       H_Voigt = alpha*u2/J*(2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
#           ut/J*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) 

#       H_Voigt = Voigt(H_Voigt ,1)
        
#       MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#       return H_Voigt



#   def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]
#       F = StrainTensors['F'][gcounter]
#       H = J*np.linalg.inv(F).T
#       N = np.array([-1.,0.]).reshape(2,1)
#       # N = np.array([0.,0.]).reshape(2,1)
#       V = np.dot(H,N)
#       innerVV = np.dot(V.T,V)
#       outerVV = np.dot(V,V.T)
#       FN = np.dot(F,N)

#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb
#       # FIX ALPHA
#       alpha = 0.5

#       u1=mu/2. 
#       u2=mu/2.
#       # u3=mu/2.
#       u3 = 0.
#       ut = alpha*(u1+2.0*u2)+(1.-alpha)*u3

#       # stress = alpha*u1/J*b + alpha*u2/J*(trace(b)*b - np.dot(b,b)) + \
#       #       u3/J*(1.-alpha)*np.dot(FN.T,FN) + u3/J*(1.-alpha)*(innerVV*I-outerVV) - \
#       #       ut/J*I + lamb*(J-1)*I

#       # stress = u1/J*b + u2/J*(trace(b)*b - np.dot(b,b)) - ut/J*I + lamb*(J-1)*I

#       stress = alpha*u1/J*b + alpha*u2/J*(trace(b)*b - np.dot(b,b)) - ut*I

#       # print stress
#       return stress

#   def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
#       ndim = StrainTensors['I'].shape[0]
#       return np.zeros((ndim,1))


# import numpy as np
# from numpy import einsum
# from Core.Supplementary.Tensors import *


# #####################################################################################################
#                               # Isotropic AnisotropicMooneyRivlin Model
# #####################################################################################################


# class AnisotropicMooneyRivlin(object):
#   """ Polyconvex compressible MooneyRivlin material model based on the energy:

#           W = alpha*C:I+beta*G:I+lambda/2*(J-1)**2-4*beta*J-2*alpha*lnJ - (3*alpha-beta)

#       where at the origin (alpha + beta) = mu/2
#       """

#   def __init__(self, ndim):
#       super(AnisotropicMooneyRivlin, self).__init__()
#       self.ndim = ndim

#   def Get(self):
#       self.nvar = self.ndim
#       self.modelname = 'AnisotropicMooneyRivlin'
#       return self.nvar, self.modelname

#   def Hessian(self,MaterialArgs,ndim,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):


#       # GET MATERIAL CONSTANTS 
#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb



#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]
#       F = StrainTensors['F'][gcounter]
#       H = J*np.linalg.inv(F).T
#       N = np.array([-1.,0.]).reshape(2,1)
#       V = np.dot(H,N)
#       innerVV = np.dot(V.T,V)[0][0]
#       outerVV = np.dot(V,V.T)
#       V = V[:,0]

#       # gamma= 0.0
#       # u3=beta/5.


#       # H_Voigt = 2.0*beta/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) + \
#       #   (lamb*(2.0*J-1.0) -4.0*beta)*einsum('ij,kl',I,I) - \
#       #   (lamb*(J-1.0) -4.0*beta -2.0*alpha/J)*( einsum('ik,jl',I,I) + einsum('il,jk',I,I) ) + \
#       #   4.0*(1.-gamma)*u3/J*(innerVV*einsum('ij,kl',I,I) - einsum('ij,kl',I,outerVV) - einsum('ij,kl',outerVV,I) - \
#       #   0.5*innerVV*( einsum('ik,jl',I,I)+einsum('il,jk',I,I) ) + einsum('ik,j,l',I,V,V) + einsum('i,k,jl',V,V,I) )
#       # H_Voigt = Voigt(H_Voigt,1) 

#       u2 = mu
#       ut = mu - u2/2.
#       alpha  = 1.0

#       H_Voigt = alpha*u2/J*( 2.0*einsum('ij,kl',b,b) - einsum('ik,jl',b,b) - einsum('il,jk',b,b) ) - \
#           ut*einsum('ij,kl',I,I) + ut*( einsum('ik,jl',I,I) - einsum('il,jk',I,I) )
#       H_Voigt = Voigt(H_Voigt,1) 

#       MaterialArgs.H_VoigtSize = H_Voigt.shape[0]

#       return H_Voigt



#   def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

#       I = StrainTensors['I']
#       J = StrainTensors['J'][gcounter]
#       b = StrainTensors['b'][gcounter]

#       mu = MaterialArgs.mu
#       lamb = MaterialArgs.lamb

#       F = StrainTensors['F'][gcounter]
#       H = J*np.linalg.inv(F).T
#       N = np.array([-1.,0.]).reshape(2,1)
#       V = np.dot(H,N)
#       innerVV = np.dot(V.T,V)[0][0]
#       outerVV = np.dot(V,V.T)
#       V = V[:,0]
#       FN = np.dot(F,N)
        
#       u2 = mu
#       ut = mu - u2/2.
#       alpha  = 1.0
        
#       # stress = 2.0*alpha/J*b+2.0*beta/J*(trace(b)*b - np.dot(b,b)) + (lamb*(J-1.0)-4.0*beta-2.0*alpha/J)*I  + \
#       #   2.0*u3/J*(1.-gamma)*np.dot(FN.T,FN) + 2.0*u3/J*(1.-gamma)*(innerVV*I-outerVV) - 2.0*u3/J*(1.-gamma)*I

#       stress = alpha*u2/J*(trace(b)*b - np.dot(b,b)) -ut * I
            

#       return stress


#   def ElectricDisplacementx(self,MaterialArgs,StrainTensors,ElectricFieldx):
#       ndim = StrainTensors['I'].shape[0]
#       return np.zeros((ndim,1))

