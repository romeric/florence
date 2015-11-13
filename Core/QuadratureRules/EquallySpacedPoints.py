import numpy as np

def EquallySpacedPoints(ndim=2,C=0):
	"""Produce equally spaced points in (ndim-1) dimension, for the boundaries
		of the mesh. For ndim=2 this is the region is enclosed in [-1,1] range

		input: 				[int] order of polynomial interpolation
		Returns:			[ndarray] array of equally spaced points 

		"""

	assert ndim<4

	if ndim==2:
		# FOR 2-DIMENSION
		return np.linspace(-1,1,C+2)[:,None]
	elif ndim==3:
		return 
