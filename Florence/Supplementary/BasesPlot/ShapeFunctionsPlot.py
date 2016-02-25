import imp, os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import cm

import Core.InterpolationFunctions.TwoDimensional.Tri.hpNodal as Tri
import Core.InterpolationFunctions.ThreeDimensional.Tetrahedral.hpNodal as Tet
import Core.InterpolationFunctions.TwoDimensional.Quad.QuadLagrangeGaussLobatto as Quad
import Core.InterpolationFunctions.ThreeDimensional.Hexahedral.HexLagrangeGaussLobatto as Hex
from Core.MeshGeneration.ReadSalomeMesh import ReadMesh
from scipy import io 


def PlotTris():

	pwd = os.path.dirname(os.path.realpath('__file__'))

	C=1
	# Get the mesh
	mesh = ReadMesh(pwd+'/Core/Supplementary/BasesPlot/TriMesh.dat','tri',0)
	# mesh = ReadMesh(pwd+'/Core/Supplementary/BasesPlot/TriMeshCoarse.dat','tri',0)

	points   = mesh.points
	elements = mesh.elements 
	# edges    = mesh.edges

	nsize = int((C+2)*(C+3)/2.)

	Bases = np.zeros((points.shape[0],nsize))
	for i in range(0,points.shape[0]):
		Bases[i,:] = Tri.hpBases(C,points[i,0],points[i,1])[0]			

	plt.triplot(points[:, 0], points[:, 1], elements)
	plt.tricontourf(points[:,0], points[:,1], elements, Bases[:,3], 100, cmap=cm.coolwarm)
	plt.colorbar()
	plt.show()

	# mydict = {'points':points,'elements':elements,'Bases':Bases}
	# io.savemat('/home/roman/Desktop/orthopoly/bases_p7.mat',mydict)


def PlotTets():

	# Get the mesh
	from meshpy.tet import MeshInfo, build

	mesh_info = MeshInfo()
	mesh_info.set_points([
	    (-1,-1,-1), (1,-1,-1), (-1,1,-1), (-1,-1,1)
	    ])
	mesh_info.set_facets([
	    [0,2,1],
	    [0,1,3],
	    [0,3,2],
	    [2,1,3]
	    ])
	mesh = build(mesh_info,max_volume=0.0002)
	points = np.asarray(mesh.points)
	elements = np.asarray(mesh.elements)
	faces = np.asarray(mesh.faces)

	# mesh.save_faces('/home/roman/Desktop/test.dat')

	C=2
	nsize = int((C+2)*(C+3)*(C+4)/6.)

	Bases = np.zeros((points.shape[0],nsize))
	for i in range(0,points.shape[0]):
		Bases[i,:] = Tet.hpBases(C,points[i,0],points[i,1],points[i,2])[0]	

	mydict = {'points':points,'elements':elements,'faces':faces,'Bases':Bases}
	io.savemat('/home/roman/Desktop/orthopoly/bases_tet_p'+str(C+1)+'.mat',mydict)



def PlotQuads():
	# THIS FUNCTION JUST WRITES TO THE DICTIONARY

	pwd = os.path.dirname(os.path.realpath('__file__'))
	mesh = ReadMesh(pwd+'/Core/Supplementary/BasesPlot/QuadMesh.dat','quad',0)
	points   = mesh.points
	elements = mesh.elements 

	C=4
	nsize = (C+2)**2
	Bases = np.zeros((points.shape[0],nsize))
	for i in range(0,points.shape[0]):
		Bases[i,:] = Quad.LagrangeGaussLobatto(C,points[i,0],points[i,1])[0].reshape(Bases.shape[1])

	mydict = {'points':points,'elements':elements,'Bases':Bases}
	io.savemat('/home/roman/Desktop/orthopoly/bases_quad_p'+str(C+1)+'.mat',mydict)


def PlotHexs():
	# THIS FUNCTION JUST WRITES TO THE DICTIONARY

	pwd = os.path.dirname(os.path.realpath('__file__'))
	mesh = ReadMesh(pwd+'/Core/Supplementary/BasesPlot/HexMesh.dat','hex',0)
	points   = mesh.points
	elements = mesh.elements 

	C=3
	nsize = (C+2)**3
	Bases = np.zeros((points.shape[0],nsize))
	for i in range(0,points.shape[0]):
		Bases[i,:] = Hex.LagrangeGaussLobatto(C,points[i,0],points[i,1],points[i,2])[0].reshape(Bases.shape[1])

	mydict = {'points':points,'elements':elements,'Bases':Bases}
	io.savemat('/home/roman/Desktop/orthopoly/bases_hex_p'+str(C+1)+'.mat',mydict)