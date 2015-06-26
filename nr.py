import numpy as np 
import matplotlib.pyplot as plt
import scipy.io as io
import imp, os, sys, time, cProfile	

# from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool

def nr():

	# y = np.sqrt(x**3+8.0)
	# dy = 3.0*x**2/2.0/np.sqrt(x**3+8)

	tol = 1e-14

	x0=7
	r=0.1
	xiter =[]
	while r > tol:

		# y = np.sqrt(x0**3+8.0)
		# dy = 3.0*x0**2/2.0/np.sqrt(x0**3+8)
		# y = x0**3-8.0
		# dy = 3.0*x0**2
		# y = np.log(np.sqrt(x0))-1
		# dy = 1./2./x0
		y = x0+1
		dy = 1.
		
		x1 = x0-y/dy

		r = np.abs(x0-x1)/np.abs(x1)
		x0=x1

		xiter = np.append(xiter,r)
		print x0

	plt.figure()
	y = np.linspace(0,len(xiter),len(xiter))
	plt.semilogy(y,xiter,'-o')

	# plt.figure()
	# x = np.linspace(2,5,100)
	# y = x**3-8
	# plt.plot(x,y)
	plt.show()




def check(zeta,eta,beta):

	m = 1/8
	n1 = m*(1-zeta)*(1-eta)*(1-beta)
	n2 = m*(1+zeta)*(1-eta)*(1-beta)
	n3 = m*(1+zeta)*(1+eta)*(1-beta)
	n4 = m*(1-zeta)*(1+eta)*(1-beta)
	n5 = m*(1-zeta)*(1-eta)*(1+beta)
	n6 = m*(1+zeta)*(1-eta)*(1+beta)
	n7 = m*(1+zeta)*(1+eta)*(1+beta)
	n8 = m*(1-zeta)*(1+eta)*(1+beta)

	N = np.array([n1,n2,n3,n4,n5,n6,n7,n8])
	print N


def nr2():

	alpha = 2.5
	A = alpha*np.eye(3,3)

	x = np.zeros((3,1))
	y = np.array([[0.,0.,1.]]).T
	tol = 1e-13

	r = -y
	while np.linalg.norm(r)/np.linalg.norm(y) > tol:
		# # y = np.dot(A,x)
		# dx = - np.dot(np.linalg.inv(A),r)
		# x += dx
		# r = x - y
		# print np.linalg.norm(r)
		# # print x

		# y = np.dot(A,x)
		dx = - np.dot(np.linalg.inv(A),r)
		x1 = x - dx
		r = x - y
		print np.linalg.norm(r)
		# print x


from Core.MeshGeneration.ReadSalomeMesh import ReadMesh
# from Core.MeshGeneration.PythonMeshScripts import vtk_writer

def dumm():

	# vmesh = ReadHexMesh('/home/roman/Partition_3.dat')
	# # print mesh.points.shape
	# # print mesh.elements.shape
	# # print mesh.edges.shape
	# # print mesh.faces.shape
	# vdata = vmesh.points**2
	# # print vdata.shape
	# vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, pdata=vdata, fname='cylcheck.vtu')

	# vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Half_Bent_Pipe.dat')
	# vmesh = ReadTetMesh('/home/roman/Dropbox/Today/Circular_Holes.dat')
	# vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Partition_3.dat')
	# vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Ellipse_Cylinder_4000.dat')
	vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Ellipse_Cylinder_32000.dat')

	# print mesh.points.shape
	# print mesh.elements.shape
	# print mesh.edges.shape
	# print mesh.faces.shape
	vdata = np.sin(vmesh.points)**2
	# print vdata.shape
	# vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, pdata=vdata, fname='Half_Bent_Pipe.vtu')
	# vtk_writer.write_vtu(Verts=vmesh.points, Cells={10:vmesh.elements}, pdata=vdata, fname='Circular_Holes.vtu')
	# vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, pdata=vdata, fname='Ellipse_Cylinder_32000.vtu')

def dummdyn():

	vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Ellipse_Cylinder_4000.dat')
	# vmesh = ReadHexMesh('/home/roman/Dropbox/Today/Ellipse_Cylinder_32000.dat')

	m=0
	points=np.copy(vmesh.points)
	# print np.max(vmesh.points[:,m])
	for i in range(0,100):
		vmesh.points=np.copy(points)
		vdata = vmesh.points+np.array([[0.1*i,0.2*i,i]])
		# vdata = np.sqrt(i)*vmesh.points/(i+10)
		# vmesh.points[:,0] += vdata[:,0]
		# vmesh.points[:,1] += vdata[:,1]
		# vmesh.points[:,2] += vdata[:,2]
		vmesh.points[:,0] = vdata[:,0]
		vmesh.points[:,1] = vdata[:,1]
		vmesh.points[:,2] = vdata[:,2]
		vtk_writer.write_vtu(Verts=vmesh.points, Cells={12:vmesh.elements}, pdata=vdata,
		fname='/home/roman/Dropbox/Today/VTKs/EllipseDyn/Ellipse_Cylinder_'+str(i)+'.vtu')
		# print np.max(vmesh.points[:,m])
		# print np.max(vdata[:,m])



def convertimage():
	# n=30
	# for i in range(0,n):
	# 	if i<n-1:
	# 		print 'convert expansion.00'+str(i)+'.jpg -crop 1580x870+110+0 expansion.00'+str(i)+'.png' + ' &&', 
	# 	elif i==n-1:
	# 		print 'convert expansion.00'+str(i)+'.jpg -crop 1580x870+110+0 expansion.00'+str(i)+'.png',


	# n=30
	# for i in range(0,n):
	# 	if i<n-1:
	# 		print 'convert expansion.00'+str(i)+'.png -crop 1580x870+80+300 expansion.00'+str(i)+'.png' + ' &&', 
	# 	elif i==n-1:
	# 		print 'convert expansion.00'+str(i)+'.png -crop 1580x870+110+0 expansion.00'+str(i)+'.png',

	n=30
	for i in range(0,n):
		if i<n-1:
			print 'convert expansion_2.00'+str(i)+'.jpg -crop 1580x500+0+230 expansion_2.00'+str(i)+'.png' + ' &&', 
		elif i==n-1:
			print 'convert expansion_2.00'+str(i)+'.jpg -crop 1580x500+0+230 expansion_2.00'+str(i)+'.png',

			# convert expansion_2.000.jpg -crop 1580x500+0+230 expansion_2.0000.jpg


def asmble(elem):
	n=30
	A = np.random.rand(n,n)
	B = np.random.rand(n,n)
	A.dot(B)


def vtkWriter1():
	C=1
	# fname = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_80.dat'
	fname = '/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_16.dat'
	class MeshInfo(object):
		MeshType = 'tri'
		Nature = 'straight'
			
	mesh = ReadMesh(fname,'tri',0)
	nmesh = HighOrderMeshTri(C,mesh,MeshInfo)
	# print nmesh.points.shape

	pdata1 = mesh.points**2

	# vtk_writer.write_vtu(Verts=nmesh.points, Cells={5:nmesh.elements}, pdata=pdata1, pvdata=None, cdata=None, cvdata=None, fname='/home/roman/Desktop/test.vtu')
	# print mesh.elements	
	# print nmesh.elements.reshape(nmesh.elements.shape[0]*nmesh.elements.shape[1])
	dp= np.concatenate(( nmesh.points, np.zeros((nmesh.points.shape[0],1)) ),axis=1)
	dp =  dp.reshape(dp.shape[0]*dp.shape[1])
	# for i in range(dp.shape[0]):
		# print dp[i],
	# print mesh.points
	# print nmesh.points[:,0]**2,

	# for i in range(0,mesh.elements.shape[0]):
	# 	print (i+1)*3
	# for i in range(0,mesh.elements.shape[0]):
		# print (i+1)*5,

	print nmesh.elements.shape, nmesh.points.shape



from Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal import *

def shapes(x,y):

	N = np.array([
		[(1-x)*(1-y)/4],
		[(1+x)*(1-y)/4],
		[(1+y)/2]
		]).reshape(3)
 
 
	dN = np.array([
		[-1./4.*(1-y), -1./4.*(1-x)],
        [1./4.*(1-y),  -1./4.*(1+x)],
        [0.0,           1./2.      ]
        ])

	return N, dN 


def checktri():

	C=0
	# N,dN=hpBases(C,-1,1,1)

	# N,dN=hpBases(C,-1,-1)
	# N,dN=hpBases(C,1,-1)

	# print N
	# print 
	# print dN
	x=np.linspace(-1,1,100);
	N = np.zeros((100,3)); Ns = np.zeros((100,3))
	dNx = np.zeros((100,3)); dNsx = np.zeros((100,3))
	dNy = np.zeros((100,3)); dNsy = np.zeros((100,3))
	tt = -0.9
	for i in range(0,len(x)):
		# N[i,:]=hpBases(C,x[i],tt)[0]
		# Ns[i,:]=shapes(x[i],tt)[0]

		dNx[i,:]=hpBases(C,x[i],tt)[1][:,0]
		dNsx[i,:]=shapes(x[i],tt)[1][:,0]

		dNy[i,:]=hpBases(C,x[i],tt)[1][:,0]
		dNsy[i,:]=shapes(x[i],tt)[1][:,0]


	# print N 
	# m=2
	# plt.plot(x,N[:,m],'o',x,Ns[:,m],'-.')
	# plt.show()

	m=0
	plt.plot(x,dNx[:,m],'o',x,dNsx[:,m],'-.')
	plt.show()

	print np.linalg.norm(dNx - dNsx)
	# print np.linalg.norm(N-Ns) 




def write_mesh():
	# mesh = ReadMesh('/home/roman/Desktop/check_tri_fem/Mesh_2.dat','quad',0)
	# mesh = ReadMesh('/home/roman/Desktop/check_tri_fem/Mesh_100.dat','quad',0)
	# print mesh.elements
	# np.savetxt('/home/roman/Desktop/check_tri_fem/elements_quad.txt', mesh.elements)
	# np.savetxt('/home/roman/Desktop/check_tri_fem/points_quad.txt', mesh.points)

	mesh = ReadMesh('/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_16.dat','tri',0)
	# mesh = ReadMesh('/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_80.dat','tri',0)
	# mesh = ReadMesh('/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_1838.dat','tri',0)
	# mesh = ReadMesh('/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Sqaure_Tri_7748.dat','tri',0)

	np.savetxt('/home/roman/Desktop/check_tri_fem/elements.txt', mesh.elements)
	np.savetxt('/home/roman/Desktop/check_tri_fem/points.txt', mesh.points)




def FindIndices(A):
	rows=np.zeros(A.shape[0]*A.shape[1], dtype=int); cols = np.copy(rows); 
	coeff = np.zeros(A.shape[0]*A.shape[1])
	counter = 0
	for i in range(0,A.shape[0]):
		for j in range(0,A.shape[1]):
			if np.abs(A[i,j]) < 1e-15:
				A[i,j]=0
			rows[counter] = i
			cols[counter] = j
			coeff[counter] = A[i,j]
			counter += 1

	# Faster method hopefully
	print coeff
	print rows
	print cols
	# print

	# print A.reshape(rows.shape[0]) 
	# print np.repeat(np.linspace(0,A.shape[0]-1,A.shape[0]),A.shape[0],axis=0).astype(int)
	# print np.linspace(0,A.shape[0]-1,A.shape[0]).astype(int)

	print A.ravel()
	print np.repeat(np.arange(0,A.shape[0]),A.shape[0],axis=0)
	print np.tile(np.arange(0,A.shape[0]),A.shape[0])



	return rows, cols, coeff




from Core.Supplementary.BasesPlot.ShapeFunctionsPlot import *
def meshplots():
	# PlotQuads()
	PlotHexs()
	pass


# from Core.NumericalIntegration import *
def quadtri():

	C=7
	z,w= GaussQuadrature(C+2,-1,1)

	# Triangular Gauss quadrature based on 1D Gauss quadrature
	allGauss=[]
	for i in range(0,w.shape[0]):
		for j in range(0,w.shape[0]):
			allGauss = np.append(allGauss,w[i]*w[j]*(1-z[j])/2. )



	# Test triangular quadrature
	from Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal import hpBases, GradhpBases
	Area = 0
	nsize = int((C+2)*(C+3)/2.)

	for i in range(0,allGauss.shape[0]):
		Area += allGauss[i]

	Bases = np.zeros((nsize,allGauss.shape[0]))
	gBasesx = np.zeros((nsize,allGauss.shape[0]))
	gBasesy = np.zeros((nsize,allGauss.shape[0]))
	g=0

	
	for i in range(0,w.shape[0]):
		for j in range(0,w.shape[0]):
			ndum, dum = hpBases(C,z[i],z[j])
			Bases[:,g] = ndum
			gBasesx[:,g] = dum[:,0]
			gBasesy[:,g] = dum[:,1]

			g+=1

	# print Bases
	# print gBasesy
	# print gBasesx

	# print z
	# vechpBases = np.vectorize(hpBases,excluded=['C'],otypes=[np.float])
	# print vechpBases(C,[0.57735027, -0.57735027], [0.57735027, -0.57735027])

	print np.sum(allGauss)
	print np.sum(Bases,axis=0)
	print np.sum(gBasesx,axis=0)
	print np.sum(gBasesy,axis=0)



def NurbsStuff():

	# from Nurbs import Crv
	# crv = Crv.Circle(5., [1., 2.])


	from igakit.cad import circle
	from igakit.plot import plt as iplt
	from igakit.cad import ruled

	c1 = circle(angle=(0,np.pi/2.))
	# iplt.plot(c1,color='b')

	# print c1.control
	# print c1.knots
	print dir(c1.points)

	# plt.plot(c1.points[:,1])
	# iplt.show()

	
	# srf = ruled(c1,c2)
	# iplt.plot(srf)
	# iplt.show()


def plotelasticcurves():
	
	C=3
	dum1=[]
	for i in range(0,C):
		dum1=np.append(dum1,i+3)
	# print dum1

	dum2=[]
	for i in range(0,C):
		dum2 = np.append(dum2, 2*C+3 +i*C -i*(i-1)/2 )
	# print dum2

	dum3 = []
	for i in range(0,C):
		dum3 = np.append(dum3,C+3 +i*(C+1) -i*(i-1)/2 )
	# print dum3

	dum =( np.append(np.append(np.append(np.append(np.append(np.append(0,dum1),1),dum2),2),np.fliplr(dum3.reshape(1,dum3.shape[0]))),0) ).astype(np.int32)
	print dum
	# 2 
	# 3



from matplotlib import animation
def animationss():
	fig = plt.figure()
	fig.set_dpi(100)
	fig.set_size_inches(7, 6.5)

	ax = plt.axes(xlim=(0, 10), ylim=(0, 10))
	patch = plt.Circle((5, -5), 0.75, fc='y')

	def init():
	    patch.center = (5, 5)
	    ax.add_patch(patch)
	    return patch,

	def animate(i):
	    x, y = patch.center
	    x = 5 + 3 * np.sin(np.radians(i))
	    y = 5 + 3 * np.cos(np.radians(i))
	    patch.center = (x, y)
	    return patch,

	anim = animation.FuncAnimation(fig, animate, 
	                               init_func=init, 
	                               frames=360, 
	                               interval=20,
	                               blit=True)

	plt.show()


# import time
# import sys

 
# def do_task():
#     time.sleep(0.1)

# def example_1(n):
#     steps = n/10
#     for i in range(n):
#         do_task()
#         if i%steps == 0:
#             print '\b.',
#             sys.stdout.flush()
#     print '\b]  Done!',
# print 'Starting [',
# # print '\b'*12,
# sys.stdout.flush()
# example_1(100)

from scipy.stats import itemfreq
def unique_floats():
	# A=np.random.rand(10,6)
	A=np.arange(1,11)
	A[[1,5]]=1
	A[[6,-1]]=9
	print A
	a,b,c=np.unique(A,return_index=True,return_inverse=True)
	print a
	print b 
	print c 
	print np.bincount(c)
	# print a[c]
	print itemfreq(A)

	print len(A.reshape(2,5).shape)
	# print np.size(A)



def migakit():
	from igakit.cad import circle
	from igakit.plot import plt as iplt

	circle = circle(radius=1, center=None, angle=None)
	print circle.knots
	# print circle.reverse
	print circle.weights
	print circle.points
	print circle.array

	help(circle)

	# plt.plot(circle.array[:,0],circle.array[:,1] )
	# plt.show()

	# iplt.plot(circle,color='b')
	# iplt.plot(circle,color='g')
	# iplt.show()


# from Core.MeshGeneration.HigherOrderMeshing.HigherOrderMeshingTet import duplicate
# a=np.arange(10).reshape(5,2).astype(np.float64)
# a[2,0]=8.0; a[2,1]=9.00001
# a[1,0]=8; a[1,1]=9
# # a[4,1]=8
# print a
# # print np.round(a,decimals=10)
# print duplicate(a,decimals=5)





# import scipy.io as sio 
# xx={}
# sio.loadmat('/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.mat',mdict=xx)
# print xx['nurbs']['Pw'].shape
# print xx['nurbs']['U'].shape
# print xx['nurbs']['iniParam'].shape
# print xx['nurbs']['endParam'].shape

# print xx['nurbs']['U']


import sys
from OCC.gp import gp_Pnt, gp_XOY
from OCC.GeomAPI import GeomAPI_ProjectPointOnCurve
from OCC.Geom import Geom_Circle, Geom_Curve, Geom_BSplineCurve
from OCC.IGESControl import IGESControl_Reader
from OCC.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.TopoDS import TopoDS_Shape, TopoDS_Edge, TopoDS_Builder
from OCC.BRep import BRep_Builder, BRep_Tool
from OCC.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert
from OCC.BRepAdaptor import BRepAdaptor_Curve
# import OCC.TopoDS as TopoDS
from OCC.TopoDS import *
from OCC.TopExp import TopExp_Explorer
from OCC.TopAbs import TopAbs_VERTEX, TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from Core import Mesh

from OCC.TColStd import TColStd_HSequenceOfTransient,  TColStd_Array1OfReal, TColStd_Array1OfInteger
from OCC.gp import gp_Vec, gp_Pnt2d, gp_Pln, gp_Dir
from OCC.TColgp import TColgp_Array1OfPnt2d, TColgp_Array1OfPnt
from Core.Supplementary.Tensors import itemfreq_py
def pythoncc_test():



	# load the data manually and build Geom_BSplineCurve
	U = np.array([ 0.,    0.,    0.,    0.25,  0.25,  0.5,   0.5,   0.75,  0.75,  1.,    1.,    1.  ])
	Pw = np.array([[-1.,          0.,          0.,          1.,        ],
				 [-0.70710678,  0.70710678,  0.,          0.70710678],
				 [ 0.        ,  1.        ,  0.,          1.        ],
				 [ 0.70710678,  0.70710678,  0.,          0.70710678],
				 [ 1.        ,  0.        ,  0.,          1.        ],
				 [ 0.70710678, -0.70710678,  0.,          0.70710678],
				 [ 0.        , -1.        ,  0.,          1.        ],
				 [-0.70710678, -0.70710678,  0.,          0.70710678],
				 [-1.        ,  0.        ,  0.,          1.        ]])

	# Allocate array
	array_Pw = TColgp_Array1OfPnt(1, Pw.shape[0])
	# print dir(array)
	# array.SetValue(1, gp_Pnt(0, 0,0 ) )
	for i in range(Pw.shape[0]):
		array_Pw.SetValue(i+1, gp_Pnt(Pw[i,0], Pw[i,1], Pw[i,2] ) )
		# print array_Pw.Value(i+1).X()

	array_U = TColStd_Array1OfReal(1, U.shape[0])
	for i in range(U.shape[0]):
		array_U.SetValue(i+1, U[i] )
		# print array_U.Value(i+1)

	mult = itemfreq_py(U)
	array_M = TColStd_Array1OfInteger(1,mult.shape[0])
	for i in range(mult.shape[0]):
		dum= int(mult[i,1])
		# array_M.SetValue(int(mult[i,1]))
		# array_M.SetValue(int(mult[i,1]))
		# print array_M.Value(i+1)

	# bspCrv = Geom_BSplineCurve(array_Pw,array_U)
		
	# print array.Length() 
	

	iges_reader = IGESControl_Reader()
	status = iges_reader.ReadFile('/home/roman/Dropbox/2015_HighOrderMeshing/examples/rae2822.igs')
	if status == IFSelect_RetDone:  # check status
	    failsonly = False
	    iges_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
	    iges_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
	    ok = iges_reader.TransferRoots()
	    _nbs = iges_reader.NbShapes()
	    aResShape = iges_reader.Shape(2)



	    mlist = iges_reader.GiveList("some_curve")
	    nbtrans = iges_reader.TransferList(mlist)
	    aResShape1 = iges_reader.OneShape()
	    # OCC.TopoDS.Edge
	    # print dir(OCC.TopoDS)
	    # print TopoDS_Edge()
	    # aResShape = iges_reader.edge(2)
	    # xx = iges_reader.IGESModel()	
	    # print dir(xx)
	else:
	    raise NameError('Could not read the .iges/.igs file')

	# OCC.TopoDS(aResShape)
	# print dir(iges_reader)
	# print dir(mlist)
	# print dir(nbtrans)
	# aa = TopoDS_Edge()
	# b = BRep_Tool.Curve(aa)

	# builder = BRep_Builder()
	# a = BRep_Builder().MakeEdge(aa)
	# print 
	# a = TopoDS_Builder()
	# print dir(a)
	# print a 
	# print dir(a)

	a = BRepBuilderAPI_NurbsConvert()
	a.Perform(aResShape)
	# print dir(a)


	# print a.IsDone()
	# b = a.ModifiedShape(aResShape)
	# print b.Modified()
	# b= a.Generated(aResShape)
	# print dir(b)
	# print b[1]








	C=1

	 
	mesh = Mesh()
	mesh.ReadSeparate('/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/elements_circle.dat',
		'/home/roman/Dropbox/Python/Problems/FiniteElements/Annular_Circle_Nurbs/points_circle.dat','tri',',',',')
	mesh.GetBoundaryEdgesTri()
	mesh.GetHighOrderMesh(C,Parallel=True,nCPU=16)

	radius = 1.
	circle = Geom_Circle(gp_XOY(), radius)
	# print dir(circle)
	for iface in range(mesh.edges.shape[0]):
		for j in range(0,mesh.edges.shape[1]):
			# print mesh.points[mesh.edges[iface,j],:]
			point_to_project = mesh.points[mesh.edges[iface,j],:]
			x = point_to_project[0]
			y = point_to_project[1]
			# x = mesh.points[mesh.edges[iface,j],0]
			# y = mesh.points[mesh.edges[iface,j],1]
			# print x,y
			# gp_Pnt(mesh.points[mesh.edges[iface,j],:])
			point_to_project = gp_Pnt(x, y, 0.)
			projection = GeomAPI_ProjectPointOnCurve(point_to_project,circle.GetHandle())
			projected_point = projection.NearestPoint()
			# the number of possible results
			nb_results = projection.NbPoints()
			# print("NbResults : %i" % nb_results)
			# print projected_point
	# print dir(point_to_project)

	# xyz = point_to_project.XYZ
	# print dir(xyz)
	# print dir(projected_point)
	x = projected_point.X()
	y = projected_point.Y()
	z = projected_point.Z()
	print np.sqrt(x**2+y**2)
	d = projection.LowerDistance()
	dd = projection.LowerDistanceParameter()

	pp = gp_Pnt(0, 0, 0.)
	circle.D0(dd,pp)
	print x, pp.X()


	from OCC.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
	from OCC.TopAbs import TopAbs_FORWARD
	from OCC.TopLoc import TopLoc_Location
	# from OCC.TopoDS import TopoDS_Edge
	# from OCC.gp import gp_Pnt
	# from OCC.BRep import BRep_Tool


	line = BRepBuilderAPI_MakeEdge(gp_Pnt(), gp_Pnt(100,0,0)).Edge()


	class InheritEdge(TopoDS_Edge):
	    def __init__(self, edge):
	        # just returns an uninitialized TopoDS_Edge...
	        super(InheritEdge, self).__init__()
	        # copy a reference to the new TopoDS_Edge instance
	        # I would assume this is more sharing a pointer, than duplicating the underlying data in `edge`
	        self.TShape(edge.TShape())
	        self.Location(edge.Location())
	        self.Orientation(edge.Orientation())

	    def get_curve(self):
	        return BRep_Tool.Curve(self)

	ie = InheritEdge(line)
	# print ie.get_curve()
	# print circle.GetHandle()
	# print dir(ie.get_curve())	
	# print dir(circle.GetHandle())

	# projection = GeomAPI_ProjectPointOnCurve(point_to_project,ie.get_curve())
	
	# print dir(iges_reader)
	# mc = Geom_Curve()
	# print dir(aResShape)
	# print aResShape.Located()
	# print aResShape.()
	# print aResShape.TShape()

	# mm = BRepBuilderAPI_NurbsConvert()
	# mm.Perform(aResShape)

	# exp = TopExp_Explorer(aResShape,TopAbs_EDGE)
	# print TopoDS.TopoDS().Edge(exp.CURRENT())
	# print TopoDS_Edge()
	# print dir(TopoDS)
	# print aResShape.Edge
	# print dir(exp) 
	# print exp.Depth()
	# ss =  TopoDS_Edge()
	# print dir(ss)
	# print ss.ShapeType()


	# edges = []
	# explorer = TopExp_Explorer(aResShape, TopAbs_EDGE)
	# explorer.ReInit()
	# while explorer.More():
	# 	# edge = TopoDS().Edge(explorer.Current())
	# 	# edge = TopoDS.Edge(explorer.Current())
	# 	edge1 = Edge(edge)
	# 	edges.append(edge1)
	# 	explorer.Next()
	# return edges

	# print dir(TopoDS)
	# TopoDS_Edge(aResShape)

	# mm  = BRepAdaptor_Curve
	# edge  = TopoDS_Edge()
	# print dir(edge)


	# mm().Initialize(aResShape)
	# print dir(mm)
	# print aResShape.TShape()
	# print dir(aResShape)

	# aResEdge = TopoDS_Edge()
	# print dir(aResEdge)
	# print dir(TopoDS)
	# print dir(TopoDS.TopoDS_Edge())
	# print dir(aResShape)


	# print dir(mm)
	# print dir(iges_reader)
	# print dir(iges_reader.OneShape())
	# mm2 =  mm.Shape()
	# print dir(mm2)
	# print mm.Generated(mm)

	# print _nbs
	for i in range(aResShape.ShapeType()):
		# print dir(aResShape.Location())
		# print dir(aResShape.TShape)
		pass
	# xdd = Geom_Curve.Geom_Curve_D0(dd)

	# from pyocc import OCC_shape


	# a = OCC_shape(aResShape)
	# print a.get_edges()
	# print dir(TopoDS)
	# print a() 
	# print d,dd

	# print dd, x,y
	# help(circle)

	# help(projected_point)
	# help(projection)
	# x=[]
	# x.append(point_to_project)
	# print x
			# print 
	# help(point_to_project)
	# print point_to_project.Coord

	# print point_to_project

	# aa = Geom_Curve()
	# print aa



if __name__ == '__main__':

	pythoncc_test()


	# migakit()

	# unique_floats()

	# plotelasticcurves()
	# animationss()


	# quadtri()

	# NurbsStuff()

	# meshplots()
	


	

	# n = 5
	# elem = 1000
	# A = np.random.rand(n,n)
	# # for i in range(0,elem):
	# # 	FindIndices(A)

	# FindIndices(A)
	#----------------

	# write_mesh()
	# checktri()

	# vtkWriter1()
	
	# nr()
	# nr2()

	########################################

	# dumm()
	# dummdyn()
	# convertimage()

	# p=Pool(2)
	# p.map(asmble, ( (elem,) range(0,100) ) )





	# Nested vs one big for loop - for numpy stuff it seems to pay off but not much - recheck
	# n=20
	# A = np.random.rand(n,n)
	# B = np.random.rand(n,n)
	# C = np.random.rand(n,n)
	# nn=30
	# test = 0
	# t1 = time.time() 
	# for i in range(0,nn):
	# 	for j in range(0,nn):
	# 		for k in range(0,nn):
	# 			test +=1
	# 			C.dot(A.dot(B))
	# 			A+B+C 
	# 			if i>10:
	# 				A-B
	# 			else:
	# 				A+B
	# print time.time()-t1

	# t2=time.time()
	# for i in range(0,nn**3):
	# 	test +=1
	# 	C.dot(A.dot(B))
	# 	A+B+C 
	# 	if i>10:
	# 		A-B
	# 	else:
	# 		A+B
	# print time.time()-t2

	#---------------------------------------------------------------------------------
	# 3 
	# 3 4 
	# 3 4 5

	# 5
	# 7 9
	# 9 12 14
	# 11 15 18 20 

	#  

	# 4 
	# 5 8
	# 6 10 13 
	# 7 12 16 19

	# 2
	# 19 20
	# 16 17 18
	# 12 13 14 15
	# 7 8  9  10  11
	# 0  3  4  5  6  1


	# C = 2
	# Edge0 = np.linspace(3,3+(C-1),C).astype(int)
	# Edge1 = np.array([C+3])
	# Edge2 = np.array([2*C+3])
	# for i in range(1,C):
	# 	Edge1 =  np.append(Edge1, Edge1[-1]-i+C+2 )
	# 	Edge2 =  np.append(Edge2, Edge2[-1]-i+C+1 )
	# print Edge2

	# np.array()
	# x = np.array([2,3])
	# print x
	# print np.fliplr(x.reshape(1,x.shape[0]))



	#---------------------------------------
	# class MeshInfo(object):
		# MeshType = 'tet'
			# 
	# mesh = ReadMesh('/home/roman/Dropbox/Python/Problems/FiniteElements/Hollow_Arc_Tri/Mesh_Cube_Tet_12.dat',MeshInfo.MeshType,0)
	# vdata = mesh.points**2

	# vtk_writer.write_vtu(Verts=mesh.points, Cells={10:mesh.elements}, pdata=vdata, fname='/home/roman/Desktop/cube_tet.vtu')
	#---------------------------------------



	# from Core.NumericalIntegration.FeketePointsTri import *
	# from Core.NumericalIntegration.FeketePointsTet import *
	# C=4
	# print FeketePointsTri(C)
	# print
	# print FeketePointsTet(C)

	# x = 0.16331353;  y = 0.53697847
	# print x*np.sin(y), y*np.sin(x)

	pass
