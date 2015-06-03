import numpy as np 

# MAPPING FROM QUAD TO TRI
def MapRS2XiEta(r,s):
	# Given a map (r,s) in [-1,1]^2 quad, computes the mapped (xi,eta)  

	xi = 0.5*(1.+r)*(1.-s) - 1.
	eta = 1.0*s
	return xi, eta
	

# MAPPING FROM HEX TO TET
def MapRST2XiEtaZeta(r,s,t):
	# Given a map (r,s,t) in [-1,1]^3 hex, computes the mapped (xi,eta,zeta)  
	
	zeta = 1.0*t 
	eta = 0.5*(1.+s)*(1.-t) - 1.
	xi = -0.5*(1.+r)*(eta+zeta) - 1

	return xi, eta, zeta 



# MAPPING FROM TRI TO QUAD
def MapXiEta2RS(xi,eta):

	if 'array' in str(type(eta)):
		xi = xi.reshape(xi.size); eta = eta.reshape(eta.size)
		xi = np.array([xi]); eta = np.array([eta])
	else:
		xi = np.array([[xi]]); eta = np.array([[eta]])
	xi=xi.reshape(xi.shape[1])
	eta=eta.reshape(eta.shape[1])

	dum = np.linspace(0,eta.shape[0]-1,eta.shape[0])

	posSingular = np.where(eta>0.999)[0]
	posNoSingular = np.delete(dum, posSingular)
	posNoSingular = posNoSingular.astype(int)

	# Allocate
	r = np.zeros(dum.shape[0]); s = np.copy(r)
	# Fill now
	r[posSingular] = -1.
	r[posNoSingular] = 2.*(1.+xi[posNoSingular] )/(1.- eta[posNoSingular]) - 1.
	s = eta

	return r, s


# MAPPING FROM TET TO HEX
def MapXiEtaZeta2RST(xi,eta,zeta):

	if 'array' in str(type(eta)):
		xi = xi.reshape(xi.size); eta = eta.reshape(eta.size); zeta = zeta.reshape(zeta.size); 
		xi = np.array([xi]); eta = np.array([eta]); zeta = np.array([zeta]); 
	else:
		xi = np.array([[xi]]); eta = np.array([[eta]]); zeta = np.array([[zeta]])
	xi=xi.reshape(xi.shape[1])
	eta=eta.reshape(eta.shape[1])
	zeta=zeta.reshape(zeta.shape[1])

	dum = np.linspace(0,eta.shape[0]-1,eta.shape[0])


	posSingular1 = np.where(np.abs(eta+zeta)<0.001)
	posSingular2 = np.where(zeta>0.999)
	posNoSingular1 = np.delete(dum, posSingular1)
	posNoSingular1 = posNoSingular1.astype(int)
	posNoSingular2 = np.delete(dum, posSingular2)
	posNoSingular2 = posNoSingular2.astype(int)

	r = np.zeros(dum.shape[0]); s = np.copy(r); t=np.copy(r)
	# Fill now
	r[posSingular1] = -1.
	s[posSingular2] = 1.
	r[posNoSingular1] = -2.*(1.+xi[posNoSingular1])/(eta[posNoSingular1]+zeta[posNoSingular1]) - 1.
	s[posNoSingular2] = 2.*(1.+eta[posNoSingular2])/(1.-zeta[posNoSingular2]) - 1.
	t = zeta

	return r, s, t 