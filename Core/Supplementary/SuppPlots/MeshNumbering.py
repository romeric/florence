import numpy as np 
import matplotlib.pyplot as plt

def PlotMeshNumbering(mesh):
	# TRIANGULAR MESH PLOTS

	fig = plt.figure()
	# ax = fig.axes()
	# ax = fig.add_axes([0.0,0.0,0.8,0.8])
	plt.triplot(mesh.points[:,0],mesh.points[:,1], mesh.elements[:,:3])
	plt.tricontourf(mesh.points[:,0], mesh.points[:,1], mesh.elements[:,:3], np.ones(mesh.points.shape[0]), 100,alpha=0.3)

	for i in range(0,mesh.elements.shape[0]):
		coord = mesh.points[mesh.elements[i,:],:]
		x_avg = np.sum(coord[:,0])/mesh.elements.shape[1]
		y_avg = np.sum(coord[:,1])/mesh.elements.shape[1]
		plt.text(x_avg,y_avg,str(i),backgroundcolor='#F88379',ha='center')

	for i in range(0,mesh.points.shape[0]):
		plt.text(mesh.points[i,0],mesh.points[i,1],str(i),backgroundcolor='#0087BD',ha='center')

	# xmin = np.min(mesh.points[:,0]); xmax = np.max(mesh.points[:,0])
	# ymin = np.min(mesh.points[:,1]); ymax = np.max(mesh.points[:,1])
	# xmargin = 1.0*xmax/10.
	# ymargin = 1.0*ymax/10.
	# plt.xlim(xmin - xmargin, xmax + xmargin)
	# plt.ylim(ymin - ymargin, ymax + ymargin)

	# ax = fig.add_axes([xmin - xmargin, ymin - ymargin, xmax + xmargin, ymax + ymargin])



