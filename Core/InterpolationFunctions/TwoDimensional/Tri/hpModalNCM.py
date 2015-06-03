import numpy as np 
from Core.JacobiPolynomials.JacobiPolynomials import *


def hpBases(C,x,y):

	# Size of these bases is (p+1)*(p+2)/2. where p is the polynomial degree (here p is denoted by r)

	r = C+1
	i=r-2
	j=r-2

	nsize = int((r+1.)*(r+2.)/2.)
	Bases = np.zeros(nsize)

	l0 = 1.-x-y
	l1 = x
	l2 = y

	Bases[:3] = np.array([l0,l1,l2])
	# print Bases

	if C >0:
		# Edges
		e1 = l0*l1*JacobiPolynomials(i,l1-l0,1.,1.)[:,0]
		e2 = l1*l2*JacobiPolynomials(i,l2-l1,1.,1.)[:,0]
		e3 = l2*l0*JacobiPolynomials(i,l0-l2,1.,1.)[:,0]

		Bases[3:3+C] = e1; Bases[3+C:3+2*C] = e2; Bases[3+2*C:3+3*C] = e3
		# print Bases

	if C>1:
		# Interiors
		interior = []
		for i in range(0,r-2):
			for j in range(0,r-2):
				if i+j <= r-3:
					interior = np.append(interior,l0*l1*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1])
					# print i, j

		Bases[3+3*C:] = interior

		# print interior
		# print Bases


	return Bases


def GradhpBases(C,x,y):

	# Size of these bases is (p+1)*(p+2)/2. where p is the polynomial degree (here p is denoted by r)

	r = C+1
	i=r-2
	j=r-2

	nsize = int((r+1.)*(r+2.)/2.)
	GradBases = np.zeros((nsize,2))

	l0 = 1.-x-y
	l1 = x
	l2 = y

	dl0dx = -1.
	dl1dx = 1.
	dl2dx = 0.

	dl0dy = -1.
	dl1dy = 0.
	dl2dy = 1.

	GradBases[:3,:] = np.array([
		[dl0dx,dl1dx,dl2dx],
		[dl0dy,dl1dy,dl2dy]
		]).T
	# print GradBases

	if C >0:
		# Edges
		# l1-l0 = 2.*x+y-1. 		# d(l1-l0)/dx = 2. and d(l1-l0)/dy = 1. 
		# l2-l1 = y-x 				# d(l2-l1)/dx = -1. and d(l2-l1)/dy = 1.
		# l0-l2 = 1.-x-2.*y 		# d(l0-l2)/dx = -1. and d(l0-l2)/dx = -2.

		dl1l0dx = 2.;	 dl1l0dy = 1. 
		dl2l1dx = -1.;	 dl2l1dy = 1.
		dl0l2dx = -1.;	 dl0l2dy = -2.

		de1dx = (-1.)*l1*JacobiPolynomials(i,l1-l0,1.,1.) + l0*(1.)*JacobiPolynomials(i,l1-l0,1.,1.) + l0*l1*DiffJacobiPolynomials(i,l1-l0,1.,1.,1)*dl1l0dx
		de2dx = (1.)*l2*JacobiPolynomials(i,l2-l1,1.,1.) + l1*l2*DiffJacobiPolynomials(i,l2-l1,1.,1.,1)*dl2l1dx
		de3dx = l2*(-1.)*JacobiPolynomials(i,l0-l2,1.,1.) + l2*l0*JacobiPolynomials(i,l0-l2,1.,1.)*dl0l2dx

		de1dy = (-1.)*l1*JacobiPolynomials(i,l1-l0,1.,1.) + l0*l1*DiffJacobiPolynomials(i,l1-l0,1.,1.,1)*dl1l0dy
		de2dy = l1*(1.)*JacobiPolynomials(i,l2-l1,1.,1.) + l1*l2*DiffJacobiPolynomials(i,l2-l1,1.,1.,1)*dl2l1dy
		de3dy = (1.)*l0*JacobiPolynomials(i,l0-l2,1.,1.) + l2*(-1.)*JacobiPolynomials(i,l0-l2,1.,1.) + l2*l0*DiffJacobiPolynomials(i,l0-l2,1.,1.,1)*dl2l1dy


		GradBases[3:3+C,0] = de1dx[:,0]; GradBases[3+C:3+2*C,0] = de2dx[:,0]; GradBases[3+2*C:3+3*C,0] = de3dx[:,0]
		GradBases[3:3+C,1] = de1dy[:,0]; GradBases[3+C:3+2*C,1] = de2dy[:,0]; GradBases[3+2*C:3+3*C,1] = de3dy[:,0]
		# print GradBases

	if C>1:
		# Interiors
		dinteriordx = []; dinteriordy = []
		for i in range(0,r-2):
			for j in range(0,r-2):
				if i+j <= r-3:
					didx = (-1.)*l1*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*(1.)*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*l1*l2*DiffJacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.,1)[-1]*(2./(1.-y))*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1]

					dinteriordx = np.append(dinteriordx,didx)

					didy = (-1.)*l1*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*l1*(1.)*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*l1*l2*DiffJacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.,1)[-1]*(2.*x/(1.-y)**2)*(1-y)**i*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*l1*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*i*(1-y)**(i-1)*(-1.)*JacobiPolynomials(j,2.*y-1.,2.*i+1,1)[-1] +\
					l0*l1*l2*JacobiPolynomials(i,2.*x/(1.-y)-1.,1.,1.)[-1]*(1-y)**i*DiffJacobiPolynomials(j,2.*y-1.,2.*i+1.,1.,1)[-1]*(2.)

					dinteriordy = np.append(dinteriordy,didy)

		GradBases[3+3*C:,0] = dinteriordx
		GradBases[3+3*C:,1] = dinteriordy
		# print GradBases


	return GradBases