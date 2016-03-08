import os, imp
import numpy as np
from Florence.FunctionSpace.JacobiPolynomials import *


def hpBases(C,r,s):

	order = -1

	P1=C+1
	P2=C+1 
	# Size of bases is (for equal order interpolation)
	nsize = int((P1+1.)*(P1+2.)/2.)

	p = P1-1
	q = P2-1

	Bases = np.zeros(nsize)

	a = 2.*(1.+r)/(1.-s) - 1.
	b = s


	# Vertices
	va = ((1.-a)/2.)*((1.-b)/2.)
	vb = ((1.+a)/2.)*((1.-b)/2.)
	vc = ((1.+b)/2.)

	Bases[:3] = np.array([va,vb,vc])

	if C>0:
		# Edges
		e1 = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*((1.-b)/2.)**(p+1)
		e2 = ((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]
		e3 = ((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]

		Bases[3:3+C] = e1; Bases[3+C:3+2*C] = e2; Bases[3+2*C:3+3*C] = e3
		# print Bases

		# Interior
		interior = []
		for p in range(1,P1):
			for q in range(1,P2):
				if p+q < P2:
					interior = np.append(interior,((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[order]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order])
					# print p-1,q-1

		# print interior
		Bases[3+3*C:] = interior
		# Bases = np.array([e1,e2,e3,i])

	elif C<0 or isinstance(C,float):
		raise ValueError('Order of interpolation degree should a non-negative integer')
	
	return Bases






def GradhpBases(C,r,s):

	order = -1

	P1=C+1
	P2=C+1 
	# Size of bases is (for equal order interpolation)
	nsize = int((P1+1.)*(P1+2.)/2.)

	p = P1-1
	q = P2-1

	GradBases = np.zeros((nsize,2))

	a = 2.*(1.+r)/(1.-s) - 1.
	b = s


	# Vertices
	dvadx = -0.5*((1.-b)/2.)
	dvbdx = 0.5*((1.-b)/2.)
	dvcdx = 0.

	dvady = -0.5*((1.-a)/2.)
	dvbdy = -0.5*((1.+a)/2.)
	dvcdy = 0.5

	GradBases[:3,:] = np.array([
		[dvadx,dvbdx,dvcdx],
		[dvady,dvbdy,dvcdy]
		]).T

	if C>0:
		# Edges

		# dN/dx = dN/da (a being the triangular coordinate)
		de1dx = -0.5*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)*((1.-b)/2.)**(p+1) +\
		((1.-a)/2.)*0.5*JacobiPolynomials(p-1,a,1.,1.)*((1.-b)/2.)**(p+1) +\
		((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)*((1.-b)/2.)**(p+1)

		de2dx = -0.5*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.) 

		de3dx = 0.5*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)

		# dN/dy = dN/db (b being the triangular coordinate)
		de1dy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)*(p+1)*((1.-b)/2.)**p*(-0.5)

		de2dy = ((1.-a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.) +\
		((1.-a)/2.)*((1.-b)/2.)*0.5*JacobiPolynomials(q-1,b,1.,1.) +\
		((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)

		de3dy = ((1.+a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.) +\
		((1.+a)/2.)*((1.-b)/2.)*0.5*JacobiPolynomials(q-1,b,1.,1.) +\
		((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)

		GradBases[3:3+C,0] = de1dx[:,0]; GradBases[3+C:3+2*C,0] = de2dx[:,0]; GradBases[3+2*C:3+3*C,0] = de3dx[:,0]
		GradBases[3:3+C,1] = de1dy[:,0]; GradBases[3+C:3+2*C,1] = de2dy[:,0]; GradBases[3+2*C:3+3*C,1] = de3dy[:,0]


		# Interior
		dinteriordx = []; dinteriordy = []
		for p in range(1,P1):
			for q in range(1,P2):
				if p+q < P2:
					# dN/dx = dN/da (a being the triangular coordinate)
					didx = -0.5*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[order]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order] +\
					((1.-a)/2.)*0.5*JacobiPolynomials(p-1,a,1.,1.)[order]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order] +\
					((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)[order]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order] 

					dinteriordx = np.append(dinteriordx,didx)

					# dN/dy = dN/db (b being the triangular coordinate)
					didy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[order]*(p+1)*((1.-b)/2.)**p*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order] +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[order]*((1.-b)/2.)**(p+1)*0.5*JacobiPolynomials(q-1,b,2.*p+1.,1.)[order] +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[order]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,2.*p+1.,1.,1)[order]

					dinteriordy = np.append(dinteriordy,didy)

		GradBases[3+3*C:,0] = dinteriordx
		GradBases[3+3*C:,1] = dinteriordy


	elif C<0 or isinstance(C,float):
		raise ValueError('Order of interpolation degree should a non-negative integer')
	
	return GradBases