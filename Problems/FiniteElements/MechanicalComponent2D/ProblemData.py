import numpy as np 
import os, imp


def ProblemData(MainData):

	# ndim - Dimension of the problem - 1D, 2D, 3D
	MainData.ndim = 2
	# Type of formulation - Displacement-based/ mixed etc
		# 1 - Displacement approach (for electromechanics, it is displacement-electric potential approach)
		# 2 - x, J approach
	
	MainData.Fields = 'Mechanics'
	# MainData.Fields = 'ElectroMechanics'
	
	MainData.Formulation = 1 	# Displacement-Potential based formulation
	MainData.Analysis = 'Static'
	# MainData.Analysis = 'Dynamic'
	MainData.AnalysisType = 'Linear'
	# MainData.AnalysisType = 'Nonlinear'

	class MaterialArgs(object):
		"""docstring for MaterialArgs"""
		# Type = 'Steinmann'
		# Type = 'LinearisedElectromechanics'
		# Type = 'LinearModel'
		Type = 'Incrementally_Linearised_NeoHookean'
		# Type = 'AnisotropicMooneyRivlin_1'
		# Type = 'NearlyIncompressibleNeoHookean'
		# Type = 'NeoHookean_1'
		
		
		

		E = 1.0e1
		# nu = 0.4
		nu=0.35

		E = MainData.E 
		nu = MainData.nu 


		# GET LAME CONSTANTS
		lamb = E*nu/(1.+nu)/(1.-2.0*nu)
		mu = E/2./(1+nu)
		# mu=0
		# lamb=0

		# mu = 1.
		# lamb  = 2.
		# mu    = 0.3571
		# lamb  = 1.4286
		# lamb = lamb - mu
		# mu = 2*mu
		# lamb = lamb + mu

		# mu    = 0.090571
		# lamb  = 1.4286
		# mu    = 0.5
		# lamb  = 0.6
		# mu    = 0.5
		# lamb  = 0.5
		rho   = 7.5*10e-6
		eps_1 = 1.0
		c1    = 0.
		c2    = 0.

	# print (MaterialArgs.lamb)/2./(MaterialArgs.lamb+MaterialArgs.mu)

		# mu = 23.3*1000   # N/mm^2
		# lamb = 79.4*1000 # N/mm^2
		# eps_1 = 1.5*10e-11  # C/mm^2

	MainData.MaterialArgs = MaterialArgs

	ProblemPath = os.path.dirname(os.path.realpath(__file__))
	class MeshInfo(object):
		MeshType = 'tri'
		Nature = 'straight'
		Reader = 'Read'

		# FileName = ProblemPath + '/MechanicalComponent2D_664.dat'
		FileName = ProblemPath + '/MechanicalComponent2D_192.dat'
		# FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_2672.dat'
		# FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_321.dat'
		# FileName = ProblemPath + '/MechanicalComponent2D_NonSmooth_236.dat'
		


	class BoundaryData(object):
		# NURBS/NON-NURBS TYPE BOUNDARY CONDITION
		Type = 'nurbs'

		# scale = 1000.
		# condition = 1.0e10
		scale = 1.
		condition = 1e10

		IGES_File = ProblemPath + '/mechanical2D.iges'
		# IGES_File = ProblemPath + '/mechanical2d.igs' # non-smooth

		class DirichArgs(object):
			node = 0
			Applied_at = 'node' 
									

		class NeuArgs(object):
			points=0
			node = 0
			# Applied_at = 'face'
			Applied_at = 'node'
			#--------------------------------------------------------------------------------------------------------------------------#
			# The condition upon which Neumann is applied 
			# - tuple (first is the coordinate direction x=0,y=1,z=2 and second is value of coordinate in that direction e.g. x, y or z) 
			cond = np.array([[1,2.]])
			Loads = np.array([
				[0.2,0.,0.],
				])
			# Number of nodes is necessary
			no_nodes = 0.
			#--------------------------------------------------------------------------------------------------------------------------#


		# Dynamic Data
		nstep = 100
		dt = 1./nstep
		drange = np.linspace(0.,60.,nstep)
		Amp = 10000.0
		DynLoad = Amp*np.sin(drange)
				

		def DirichletCriterion(self,DirichArgs):
			
			node = DirichArgs.node 
			points = DirichArgs.points 

			# REMOVE THIS
			#----------------------------------
			edges = DirichArgs.edges 
			unedges = np.unique(edges)
			inode = DirichArgs.inode

			r  = 0.5
			rn = np.sqrt(node[0]**2+node[1]**2)
			tol_radius = 0.1
			
			if rn < 0.5+tol_radius and rn > 0.5 - tol_radius:

				# print node[0], node[1]
				theta = np.arctan(node[1]/node[0])

				# Is this node on the edge
				p = np.where(unedges==inode)[0]
				if p.shape[0]!=0:
					# Now we are on the edge
					# x = rn*np.cos(theta)
					# y = rn*np.sin(theta)
					x=node[0]
					y=node[1]
					Lx = 1.0*r/rn*x
					Ly = 1.0*r/rn*y
					# print x, np.sign(x)
					ux = np.sign(x)*abs(abs(Lx)-abs(x))
					uy = np.sign(y)*abs(abs(Ly)-abs(y))

					b = np.array([ux,uy])
				else: 
					b = [None,None]	
			elif rn < 2.0+tol_radius and rn > 2.0 - tol_radius:

				# print node[0], node[1]
				theta = np.arctan(node[1]/node[0])

				# Is this node on the edge
				p = np.where(unedges==inode)[0]
				if p.shape[0]!=0:
					# Now we are on the edge
					b = np.array([0.,0.])
				else: 
					b = [None,None]	
			else:	
				b = [None,None]	
			
		
			return b


		def NURBSParameterisation(self):
			import scipy.io as spio 
			dummy_nurbs, nurbs={}, {}
			spio.loadmat('/home/roman/Dropbox/2015_HighOrderMeshing/examples/mechanical2d.mat',mdict=dummy_nurbs)
			nurbs['Pw'] = dummy_nurbs['nurbs']['Pw'][0]
			nurbs['U'] = dummy_nurbs['nurbs']['U'][0]
			nurbs['start'] = dummy_nurbs['nurbs']['iniParam'][0]
			nurbs['end'] = dummy_nurbs['nurbs']['endParam'][0]

			# print len(nurbs['Pw'])
			# print nurbs['Pw'].size
			fnurbs=[]
			from Core.Supplementary.Tensors import itemfreq_py
			for i in range(0,nurbs['Pw'].shape[0]):
				degree = np.int64(itemfreq_py(nurbs['U'][i])[-1,-1] - 1)
				# print degree
				fnurbs.append(({'U':nurbs['U'][i],'Pw':nurbs['Pw'][i],'start':nurbs['start'][i][0][0],'end':nurbs['end'][i][0][0],'degree':degree}))

			# print len(fnurbs) 
			# print fnurbs[0]['start']
			# print dummy_nurbs['nurbs']['iniParam'][0][1][0][0]
			# import sys;sys.exit(0)
			return fnurbs 

		def NURBSCondition(self,x):
			return np.sqrt(x[:,0]**2 + x[:,1]**2) < 0.1
			

		def ProjectionCriteria(self,mesh):
			projection_edges = np.zeros((mesh.edges.shape[0],1),dtype=np.uint64)
			num = mesh.edges.shape[1]
			for iedge in range(mesh.edges.shape[0]):
				x = np.sum(mesh.points[mesh.edges[iedge,:],0])/num
				y = np.sum(mesh.points[mesh.edges[iedge,:],1])/num
				x *= self.scale
				y *= self.scale 
				if np.sqrt(x*x+y*y)< self.condition:
					projection_edges[iedge]=1
			
			return projection_edges


		
		def NeumannCriterion(self,NeuArgs,Analysis=0,Step=0):
			# USING THIS APPROACH YOU EITHER NEED TO APPLY FORCE (N) OR YOU SHOULD KNOW THE VALUE OF AREA (M^2)
			node = NeuArgs.node
			# Area should be specified for as many physical (no meshed faces i.e. not mesh.faces) as Neumann is applied 
			area = 1.0*np.array([4.,4.,100.])

			t=[]
			for i in range(0,len(NeuArgs.cond)):
				no_nodes = 1.0*NeuArgs.no_nodes[i] 
				if Analysis != 'Static':
					if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
						t = np.array([0.,0.,self.DynLoad[Step],0.])*area[i]/no_nodes
					else:
						t = [[],[],[],[]]

				# Static Analysis 
				if Analysis=='Static':
					if np.allclose(node[NeuArgs.cond[i,0]],NeuArgs.cond[i,1]):
						t = NeuArgs.Loads[i,:]*area[i]/no_nodes
					else:
						t = [[],[],[],[]]

			return t


		# class DynamicData(object):
			# nstep = 100
			# dt = 1./nstep
			# drange = np.linspace(0.,60.,nstep)
			# Amp = 100.0
			# DynLoad = Amp*np.sin(drange)





	class AnalyticalSolution(object):
		class Args(object):
			node = 0
			points = 0




		def Get(self,Args):
			node = Args.node
			ndim = 2
			nvar = 3

			m=2
			if node.size==2:
				x = node[0]
				y = node[1]
				# sol = np.array([ 0.0,x*np.sin(y),0.0])
				sol = np.array([ 0.0,0.1*y**m])
			else:
				x = node[:,0]
				y = node[:,1]
				ux = y*0.0
				# uy = x*np.sin(y)
				uy = 0.1*y**m
				sol = np.zeros((node.shape[0],nvar))
				sol[:,0] = ux
				sol[:,1] = uy

			return sol 

			
	# PLACE THEM ALL INSIDE THE MAIN CLASS
	MainData.BoundaryData = BoundaryData
	MainData.AnalyticalSolution = AnalyticalSolution
	MainData.MeshInfo = MeshInfo

	# return MainData, MeshInfo, AnalyticalSolution 