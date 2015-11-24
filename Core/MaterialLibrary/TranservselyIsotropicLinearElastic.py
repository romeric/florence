import numpy as np
from Core.Supplementary.Tensors import *
from numpy import einsum

#####################################################################################################
								# Anisotropic MooneyRivlin Model
#####################################################################################################


class TranservselyIsotropicLinearElastic(object):
	"""A linear elastic transervely isotropic material model, with 4 material constants
		and 5 independent components in Hessian.

		Note that this assumes N = [0,0,1] as the direction of anisotropy
	"""

	def __init__(self, ndim, gamma=0.5):
		super(TranservselyIsotropicLinearElastic, self).__init__()
		self.ndim = ndim
		self.nvar = self.ndim


	def Hessian(self,MaterialArgs,StrainTensors,ElectricFieldx=0,elem=0,gcounter=0):

		E = MaterialArgs.E
		E_A = MaterialArgs.E_A
		G_A = MaterialArgs.G_A
		v = MaterialArgs.nu

		if self.ndim == 3:

			H_Voigt = np.array([
				[ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A), (E_A**2*(v - 1))/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[                                                      0,                                                      0,                                       0, 					E/(2*(v + 1)),  0,   0],
				[                                                      0,                                                      0,                                       0,             		0, 				G_A, 0],
				[                                                      0,                                                      0,                                       0,             		0,   			0, G_A]
			])

		elif self.ndim == 2:

			# G_A NOT NEEDED FOR PLAIN STRAIN. ESSENTIALLY IN PLANE STRAIN ANISOTROPY 
			# THE BEHAVIOUR OF MATERIAL PERPENDICULAR TO THE PLANE IS LOST  

			H_Voigt = np.array([
				[ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),             0],
				[   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),             0],
				[                                                      0,                                                      0, 	 E/(2*(v + 1))] 
				])
		
		MaterialArgs.H_VoigtSize = H_Voigt.shape[0]


		return H_Voigt



	def CauchyStress(self,MaterialArgs,StrainTensors,ElectricFieldx,elem=0,gcounter=0):

		strain = StrainTensors['strain'][gcounter]
		strain_Voigt = Voigt(strain)

		E = MaterialArgs.E
		E_A = MaterialArgs.E_A
		G_A = MaterialArgs.G_A
		v = MaterialArgs.nu

		if self.ndim == 3:

			H_Voigt = np.array([
				[ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),      -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A),                     -(E_A*E*v)/(2*E*v**2 + E_A*v - E_A), (E_A**2*(v - 1))/(2*E*v**2 + E_A*v - E_A),              0,   			0,   0],
				[                                                      0,                                                      0,                                       0, 					E/(2*(v + 1)),  0,   0],
				[                                                      0,                                                      0,                                       0,             		0, 				G_A, 0],
				[                                                      0,                                                      0,                                       0,             		0,   			0, G_A]
 
			])

			


		elif self.ndim == 2:

			# G_A NOT NEEDED FOR PLAIN STRAIN. ESSENTIALLY IN PLANE STRAIN ANISOTROPY 
			# THE BEHAVIOUR OF MATERIAL PERPENDICULAR TO THE PLANE IS LOST 

			H_Voigt = np.array([
				[ -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),             0],
				[   -(E*v*(E_A + E*v))/((v + 1)*(2*E*v**2 + E_A*v - E_A)), -(E*(- E*v**2 + E_A))/((v + 1)*(2*E*v**2 + E_A*v - E_A)),             0],
				[                                                      0,                                                      0, E/(2*(v + 1))]
 
				])

		stress = UnVoigt(np.dot(H_Voigt,strain_Voigt))

		return stress


	
