import imp, os
import numpy as np 
from Florence.FunctionSpace.JacobiPolynomials import *


def hpBases(C,r0,s,t):

	# The input argument r is changed to r0, because r is used as the polynomial degree in the 3rd (z) direction	
	# Coordinate transformation for tetrahedrals
	a = 2.0*(1.+r0)/(-s-t) -1.
	b = 2.0*(1.+s)/(1.-t) - 1.
	c = t

	order = -1

	P1=C+1
	P2=C+1 
	P3=C+1
	# Size of bases is (for equal order interpolation)
	nsize = int((P1+1.)*(P1+2.)*(P1+3.)/6.)
	# Vertex based bases size
	vsize = 4
	# Edge based bases size
	esize = 6*C
	# Face based bases size
	fsize = 2*C*(C-1)
	# Interior base bases size
	isize = int(C*(C-1)*(C-2)/6.)

	# Allocate
	Bases = np.zeros(nsize)


	# Vertices
	va = ((1.-a)/2.)*((1.-b)/2.)*((1.-c)/2.)
	vb = ((1.+a)/2.)*((1.-b)/2.)*((1.-c)/2.)
	vc = ((1.-a)/2.)*((1.+b)/2.)*((1.-c)/2.)  # vc = ((1.+b)/2.)*((1.-c)/2.) 
	vd = (1.+c)/2.

	Bases[:4] = np.array([va,vb,vc,vd])

	
	if C > 0:
		p = P1-1; 	q = P2-1; 	r = P3-1
		# Edges
		e1 = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)
		e2 = ((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1)
		e3 = ((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1)
		e4 = ((1.-a)/2.)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]
		e5 = ((1.+a)/2.)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]
		e6 = ((1.+b)/2.)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]

		Bases[4:4+C] = e1; Bases[4+C:4+2*C] = e2; Bases[4+2*C:4+3*C] = e3; Bases[4+3*C:4+4*C] = e4; Bases[4+4*C:4+5*C] = e5; Bases[4+5*C:4+6*C] = e6

		# Faces
		f1 = []; f2 = []; f3 = []; f4 = []
		for p in range(1,P1):
			for q in range(1,P2):
				if p+q < P2:
					f1 = np.append(f1,((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1))
		for p in range(1,P1):
			for r in range(1,P3):
				if p+r < P3:
					f2 = np.append(f2,((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1])
		for q in range(1,P2):
			for r in range(1,P3):
				if q+r < P3:
					f3 = np.append(f3,((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1])
					f4 = np.append(f4,((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1])

		Bases[4+6*C:4+6*C+2*C*(C-1)] = np.append(np.append(np.append(f1,f2),f3),f4) # 2*C*(C-1) is the total number of bases on the faces (fsize)

		# Interior
		interior = []
		for p in range(1,P1):
			for q in range(1,P2):
				for r in range(1,P3):
					if p+q+r < P3:
						interior = np.append(interior,((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1])

		Bases[4+6*C+2*C*(C-1):4+6*C+2*C*(C-1)+isize] = interior

	return Bases, np.array([nsize,vsize,esize,fsize,isize])




def GradhpBases(C,r0,s,t):


	# The input argument r is changed to r0, because r is used as the polynomial degree in the 3rd (z) direction	
	# Coordinate transformation for tetrahedrals
	a = 2.0*(1.+r0)/(-s-t) -1.
	b = 2.0*(1.+s)/(1.-t) - 1.
	c = t

	order = -1

	P1=C+1
	P2=C+1 
	P3=C+1
	# Size of bases is (for equal order interpolation)
	nsize = int((P1+1.)*(P1+2.)*(P1+3.)/6.); 
	vsize = 4; esize = 6*C; fsize = 2*C*(C-1); isize = int(C*(C-1)*(C-2)/6.)

	# Allocate
	GradBases = np.zeros((nsize,3))


	# Vertices
	# dN/dx = dN/da (a being the tetrahedral coordinate)
	dvadx = (-0.5)*((1.-b)/2.)*((1.-c)/2.)
	dvbdx = (0.5)*((1.-b)/2.)*((1.-c)/2.)
	dvcdx = (-0.5)*((1.+b)/2.)*((1.-c)/2.)  # dvcdx = 0.    # The commented one is if we follow Sherwin's 95 paper
	dvddx = 0.

	# dN/dy = dN/db (b being the tetrahedral coordinate)
	dvady = ((1.-a)/2.)*(-0.5)*((1.-c)/2.)
	dvbdy = ((1.+a)/2.)*(-0.5)*((1.-c)/2.)
	dvcdy = ((1.-a)/2.)*(0.5)*((1.-c)/2.)  # dvcdx = (0.5)*((1.-c)/2.) 
	dvddy = 0.

	# dN/dz = dN/dc (c being the tetrahedral coordinate)
	dvadz = ((1.-a)/2.)*((1.-b)/2.)*(-0.5)
	dvbdz = ((1.+a)/2.)*((1.-b)/2.)*(-0.5)
	dvcdz = ((1.-a)/2.)*((1.+b)/2.)*(-0.5)  # dvcdx = ((1.+b)/2.)*(-0.5) 
	dvddz = 0.5

	GradBases[:4,:] = np.array([
		[dvadx,dvbdx,dvcdx,dvddx],
		[dvady,dvbdy,dvcdy,dvddy],
		[dvadz,dvbdz,dvcdz,dvddz]
		]).T

	if C > 0:
		p = P1-1; 	q = P2-1; 	r = P3-1
		# Edges

		# dN/dx = dN/da (a being the tetrahedral coordinate)
		de1dx = (-0.5)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1) +\
		((1.-a)/2.)*(0.5)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1) +\
		((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)[:,0]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)		

		de2dx = (-0.5)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1) 

		de3dx = (0.5)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1)

		de4dx = (-0.5)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]

		de5dx = (0.5)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]

		de6dx = 0.


		# dN/dy = dN/db (b being the tetrahedral coordinate)
		de1dy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*(p+1)*((1.-b)/2.)**(p)*(-0.5)*((1.-c)/2.)**(p+1)

		de2dy = ((1.-a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1) +\
		((1.-a)/2.)*((1.-b)/2.)*(0.5)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1) +\
		((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)[:,0]*((1.-c)/2.)**(q+1)

		de3dy = ((1.+a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1) +\
		((1.+a)/2.)*((1.-b)/2.)*(0.5)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*((1.-c)/2.)**(q+1) +\
		((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)[:,0]*((1.-c)/2.)**(q+1)

		de4dy = ((1.-a)/2.)*(-0.5)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]

		de5dy = ((1.+a)/2.)*(-0.5)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]

		de6dy = (0.5)*((1.-c)/2.)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0]


		# dN/dz = dN/dc (c being the tetrahedral coordinate)
		de1dz = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[:,0]*((1.-b)/2.)**(p+1)*(p+1)*((1.-c)/2.)**(p)*(-0.5)

		de2dz = ((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*(q+1)*((1.-c)/2.)**(q)*(-0.5)

		de3dz = ((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[:,0]*(q+1)*((1.-c)/2.)**(q)*(-0.5)

		de4dz = ((1.-a)/2.)*((1.-b)/2.)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.-a)/2.)*((1.-b)/2.)*((1.-c)/2.)*(0.5)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.-a)/2.)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,1.,1.,1)[:,0]

		de5dz = ((1.+a)/2.)*((1.-b)/2.)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.+a)/2.)*((1.-b)/2.)*((1.-c)/2.)*(0.5)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.+a)/2.)*((1.-b)/2.)*((1.-c)/2.)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,1.,1.,1)[:,0]

		de6dz = ((1.+b)/2.)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.+b)/2.)*((1.-c)/2.)*(0.5)*JacobiPolynomials(r-1,c,1.,1.)[:,0] +\
		((1.+b)/2.)*((1.-c)/2.)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,1.,1.,1)[:,0]


		GradBases[4:4+C,0] = de1dx; GradBases[4+C:4+2*C,0] = de2dx; GradBases[4+2*C:4+3*C,0] = de3dx; GradBases[4+3*C:4+4*C,0] = de4dx; GradBases[4+4*C:4+5*C,0] = de5dx; GradBases[4+5*C:4+6*C,0] = de6dx
		GradBases[4:4+C,1] = de1dy; GradBases[4+C:4+2*C,1] = de2dy; GradBases[4+2*C:4+3*C,1] = de3dy; GradBases[4+3*C:4+4*C,1] = de4dy; GradBases[4+4*C:4+5*C,1] = de5dy; GradBases[4+5*C:4+6*C,1] = de6dy
		GradBases[4:4+C,2] = de1dy; GradBases[4+C:4+2*C,2] = de2dz; GradBases[4+2*C:4+3*C,2] = de3dz; GradBases[4+3*C:4+4*C,2] = de4dz; GradBases[4+4*C:4+5*C,2] = de5dz; GradBases[4+5*C:4+6*C,2] = de6dz


		# Faces
		dface1dx = []; dface1dy = []; dface1dz = []
		for p in range(1,P1):
			for q in range(1,P2):
				if p+q < P2:
					df1dx = (-0.5)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1) +\
					((1.-a)/2.)*(0.5)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1) +\
					((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)

					dface1dx = np.append(dface1dx,df1dx)

					df1dy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*(p+1)*((1.-b)/2.)**(p)*(0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1) +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*(0.5)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1) +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,2.*p+1.,1.,1)[-1]*((1.-c)/2.)**(p+q+1)

					dface1dy = np.append(dface1dy,df1dy)

					df1dz = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*(p+q+1)*((1.-c)/2.)**(p+q)*(-0.5)

					dface1dz = np.append(dface1dz,df1dz)

		dface2dx = []; dface2dy = []; dface2dz = []
		for p in range(1,P1):
			for r in range(1,P3):
				if p+r < P3:
					df2dx = (-0.5)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1] +\
					((1.-a)/2.)*(0.5)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1] +\
					((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1]

					dface2dx = np.append(dface2dx,df2dx)

					df2dy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*(p+1)*((1.-b)/2.)**(p)*(-0.5)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1] 

					dface2dy = np.append(dface2dy,df2dy)

					df2dz = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*(p+1)*((1.-c)/2.)**(p)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1] +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*(0.5)*JacobiPolynomials(r-1,c,2.*p+1.,1.)[-1] +\
					((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.-c)/2.)**(p+1)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,2.*p+1.,1.,1)[-1]

					dface2dz = np.append(dface2dz,df2dz)

		dface3dx = []; dface3dy = []; dface3dz = []
		dface4dx = []; dface4dy = []; dface4dz = []
		for q in range(1,P2):
			for r in range(1,P3):
				if q+r < P3:
					df3dx = (-0.5)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] 

					dface3dx = np.append(dface3dx,df3dx)

					df3dy = ((1.-a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.-a)/2.)*((1.-b)/2.)*(0.5)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1]

					dface3dy = np.append(dface3dy,df3dy)

					df3dz = ((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*(q+1)*((1.-c)/2.)**(q)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*(0.5)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.-a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,2.*q+1.,1.,1)[-1]

					dface3dz = np.append(dface3dz,df3dz)


					df4dx = (0.5)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] 

					dface4dx = np.append(dface4dx,df4dx)

					df4dy = ((1.+a)/2.)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.+a)/2.)*((1.-b)/2.)*(0.5)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,1.,1.,1)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1]

					dface4dy = np.append(dface4dy,df4dy)

					df4dz = ((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*(q+1)*((1.-c)/2.)**(q)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*(0.5)*JacobiPolynomials(r-1,c,2.*q+1.,1.)[-1] +\
					((1.+a)/2.)*((1.-b)/2.)*((1.+b)/2.)*JacobiPolynomials(q-1,b,1.,1.)[-1]*((1.-c)/2.)**(q+1)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,2.*q+1.,1.,1)[-1]

					dface4dz = np.append(dface4dz,df4dz)


		GradBases[4+6*C:4+6*C+2*C*(C-1),0] = np.append(np.append(np.append(dface1dx,dface2dx),dface3dx),dface4dx)
		GradBases[4+6*C:4+6*C+2*C*(C-1),1] = np.append(np.append(np.append(dface1dy,dface2dy),dface3dy),dface4dy)
		GradBases[4+6*C:4+6*C+2*C*(C-1),2] = np.append(np.append(np.append(dface1dz,dface2dz),dface3dz),dface4dz)



		# Interior
		dinteriordx = []; dinteriordy = []; dinteriordz = []
		for p in range(1,P1):
			for q in range(1,P2):
				for r in range(1,P3):
					if p+q+r < P3:
						didx = (-0.5)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*(0.5)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*((1.+a)/2.)*DiffJacobiPolynomials(p-1,a,1.,1.,1)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1]

						dinteriordx = np.append(dinteriordx,didx)

						didy = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*(p+1)*((1.-b)/2.)**(p)*(-0.5)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*(0.5)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*DiffJacobiPolynomials(q-1,b,2.*p+1.,1.,1)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1]

						dinteriordy = np.append(dinteriordy,didy)

						didz = ((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*(p+q+1)*((1.-c)/2.)**(p+q)*(-0.5)*((1.+c)/2.)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*(0.5)*JacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.)[-1] +\
						((1.-a)/2.)*((1.+a)/2.)*JacobiPolynomials(p-1,a,1.,1.)[-1]*((1.-b)/2.)**(p+1)*((1.+b)/2.)*JacobiPolynomials(q-1,b,2.*p+1.,1.)[-1]*((1.-c)/2.)**(p+q+1)*((1.+c)/2.)*DiffJacobiPolynomials(r-1,c,2.*p+2.*q+1.,1.,1)[-1]	

						dinteriordz = np.append(dinteriordz,didz)


		GradBases[4+6*C+2*C*(C-1):4+6*C+2*C*(C-1)+isize,0] = dinteriordx
		GradBases[4+6*C+2*C*(C-1):4+6*C+2*C*(C-1)+isize,1] = dinteriordy
		GradBases[4+6*C+2*C*(C-1):4+6*C+2*C*(C-1)+isize,2] = dinteriordz

	# Build the Jacobian to take you from a,b,c to r,s,t  (Recently changed fro r to r0)
	Jacobian = np.array([
		[-2./(s+t), 2.*(1.+r0)/(s+t)**2, 2.*(1.+r0)/(s+t)**2],
		[0., 2.0/(1.-t), 2.*(1.+s)/(1.-t)**2],
		[0., 0., 1.]
		])

	return GradBases, Jacobian
