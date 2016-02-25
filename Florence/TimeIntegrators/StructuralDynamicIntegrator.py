import numpy as np 
import scipy.linalg as sla 


class StructuralDynamicIntegrators(object):
	"""docstring for StructuralDynamicIntegrators"""
	def __init__(self):
		super(StructuralDynamicIntegrators, self).__init__()
		self.Alpha_alpha = -0.1 	
		self.Alpha_gamma = 0.5
		self.Alpha_delta = 0.
			
	def Alpha(self,K,M,F1,freedof,nstep,dt,napp,alpha,delta,gam):

		# Input/Output Data
		# M - Mass Matrix of the System
	    # K - Stiffness Matrix of the System
	    # F1 - Vector of Dynamic Nodal Forces
	    # freedof - Free Degrees of Freedom
	    # napp - Degrees of Freedom at Which Excitation is Applied
	    # nstep - No of Time Steps
	    # dt - Time Step Size
	    # delta, gam and alpha - 3 Integration parameters
	    #                     With -1/3=<alpha<=0;  delta = 0.5-gam; alpha =
	    #                     0.25*(1-gam)**2

	    # Reference:
	    # Hilber, H.M, Hughes,T.J.R and Talor, R.L. 
	    # "Improved Numerical Dissipation for Time Integration Algorithms in
	    # Structural Dynamics" Earthquake Engineering and Structural Dynamics,
	    # 5:282-292, 1977.


	    # U(n,nstep) - Matrix storing nodal displacements at each step.
	    # V(n,nstep) - Matrix storing nodal Velocities at each step.
	    # A(n,nstep) - Matrix storing nodal Accelerations at each step.
	    # n - is No of DOF's.
		
		# Allocate Space for Vectors and Matrices
		u = np.zeros(K.shape[0]); u = u[freedof]
		u0=u; v = u

		F = np.zeros(K.shape[0]);    F = F[freedof];  F2=F
		M = M[:,freedof][freedof,:]; K = K[:,freedof][freedof,:]
		U = np.zeros((u.shape[0],nstep)); V=U; A=U

		# Initial Calculations
		U[:,0] = u.reshape(u.shape[0])
		A[:,0] =  sla.solve(M,(F - np.dot(K,u0)))
		V[:,0] = v


		# Initialize The Algorithm (Compute Displacements, Vel's and Accel's)
		for istep in range(0,nstep-1):
			if istep==nstep-1:
				break

			# print nstep, istep

			# F[napp] = F1[istep]
			# F2[napp] = F1[istep+1]
			# U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
		 #        np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
			# A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
			# V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])

			F[napp] = F1[istep]
			F2[napp] = F1[istep+1]
			U[:,istep+1] = sla.solve((1/dt**2/gam*M+(1.0+alpha)*K),((1+alpha)*F2-alpha*F +\
		        np.dot((1.0/dt**2/gam*M+alpha*K),U[:,istep])+ 1.0/dt/gam*np.dot(M,V[:,istep])+(1/2/gam-1)*np.dot(M,A[:,istep])))
			A[:,istep+1] = 1.0/dt**2/gam*(U[:,istep+1]-U[:,istep]) - 1.0/dt/gam*V[:,istep]+(1-1.0/2.0/gam)*A[:,istep]
			V[:,istep+1] = V[:,istep] + dt*(delta*A[:,istep+1]+(1-delta)*A[:,istep])


		return U, V, A 





import matplotlib.pyplot as plt

dyn = StructuralDynamicIntegrators()
n = 500
stiffness = np.random.rand(n,n)
mass = np.random.rand(n,n)
# mass = np.eye(n,n)
alpha = 0.2
delta=0.5
gamma = 0.4
freedof = np.arange(0,10)
nstep = 2*n
F = np.random.rand(nstep,1)
napp=8
dt = 1.0/nstep
U, A, V = dyn.Alpha(stiffness,mass,F,freedof,nstep,dt,napp,alpha,delta,gamma)

plt.plot(U[2,:])
plt.show()