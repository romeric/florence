import numpy as np 
# import scipy as sp 
import numpy.linalg as la
from scipy.sparse.linalg import spsolve  
from scipy import sparse



def StaticCondensationGlobal(stiffness,F,nvar,nmesh,columns_in,columns_out):

	# The input is in terms of Reduced stiffness and forces (i.e. after Dirichlet is applied) 

	TotalSize = nmesh.points.shape[0]*nvar
	TotalNodes = nmesh.points.shape[0]

	x = []; y = []; z = []; t = [];
	for i in range(0,TotalNodes):
		x = np.append(x,np.where(columns_in==nvar*i)[0]).astype(int) #nvar*i
		y = np.append(y,np.where(columns_in==nvar*i+1)[0]).astype(int) #nvar*i
		z = np.append(z,np.where(columns_in==nvar*i+2)[0]).astype(int) #nvar*i
		t = np.append(t,np.where(columns_in==nvar*i+3)[0]).astype(int) #nvar*i
	x = columns_in[x]
	y = columns_in[y]
	z = columns_in[z]
	t = columns_in[t]

	uu = np.append(np.append(x,y),z); uu=np.sort(uu)

	stiffness_uu = stiffness[uu,:][:,uu]
	stiffness_up = stiffness[uu,:][:,t]		# Note that if coupling blocks are zero, the print function does not display them
	stiffness_pu = stiffness[t,:][:,uu]		# Note that if coupling blocks are zero, the print function does not display them
	stiffness_pp = stiffness[t,:][:,t]
	F_u = F[uu,0]
	F_p = F[t,0]
	# F_u = F[uu]
	# F_p = F[t]


	# solve for inversable block ------ to do this perform: AX=I 
	inv_stiffness_pp = spsolve( stiffness_pp , sparse.csr_matrix(np.eye(t.shape[0],t.shape[0])))

	# Get equivalent stiffness
	stiffness_eqv = stiffness_uu - stiffness_up.dot(inv_stiffness_pp.dot(stiffness_pu))
	# Get equivalent force
	F_eqv = F_u - stiffness_up.dot(inv_stiffness_pp.dot(F_p)) 


	return stiffness_eqv, F_eqv, stiffness_uu, stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u, uu, t



def StaticCondensationGlobalPost(sol,uu,t, fsize, stiffness_up, stiffness_pu, stiffness_pp, inv_stiffness_pp, F_p, F_u):


	p = inv_stiffness_pp.dot(F_p - stiffness_pu.dot(sol))
	total_sol_incomplete = np.zeros((fsize,1))

	total_sol_incomplete[t,0] = p
	total_sol_incomplete[uu,0] = sol 


	return total_sol_incomplete

	# return stiffness_eqv, F_eqv